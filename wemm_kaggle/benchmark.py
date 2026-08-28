from __future__ import annotations

from dataclasses import dataclass
import gc
from statistics import fmean
import time
from typing import Any, Callable, Sequence

from .embedding import check_embedding
from .evidence import process_snapshot, torch_cuda_memory_snapshot


def percentile(samples: Sequence[float], q: float) -> float:
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("samples must not be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile must be between 0 and 1")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def summarize_latencies(samples: Sequence[float], batch_size: int) -> dict[str, float | int]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("samples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_seconds = sum(values)
    total_items = batch_size * len(values)
    return {
        "min": min(values),
        "mean": fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "total_seconds": total_seconds,
        "total_items": total_items,
        "items_per_second": total_items / total_seconds,
    }


def is_cuda_oom(exc: BaseException, torch_module=None) -> bool:
    if torch_module is not None:
        oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def run_batch_progression(batches, run_one, on_record=None):
    planned = [int(batch) for batch in batches]
    if not planned or any(batch <= 0 for batch in planned):
        raise ValueError("batches must contain positive integers")
    records = []
    stopped_by = None
    stop_context = {}
    for batch in planned:
        if stopped_by is not None:
            skipped = {
                **stop_context,
                "batch": batch,
                "status": "SKIPPED_AFTER_OOM",
                "skipped_because_batch": stopped_by,
            }
            records.append(skipped)
            if on_record is not None:
                on_record(skipped)
            continue
        record = dict(run_one(batch))
        record.setdefault("batch", batch)
        records.append(record)
        if on_record is not None:
            on_record(record)
        if record.get("status") == "OOM":
            stopped_by = batch
            stop_context = {
                key: record[key]
                for key in ("workload", "dimension")
                if key in record
            }
    return records


def build_safe_envelope(records):
    groups: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (str(record["workload"]), int(record["dimension"]))
        groups.setdefault(key, []).append(record)

    envelope = []
    for (workload, dimension), rows in groups.items():
        rows = sorted(rows, key=lambda row: int(row["batch"]))
        planned = [int(row["batch"]) for row in rows]
        passed = [int(row["batch"]) for row in rows if row.get("status") == "PASS"]
        oom_batches = [int(row["batch"]) for row in rows if row.get("status") == "OOM"]
        envelope.append(
            {
                "workload": workload,
                "dimension": dimension,
                "planned_batches": planned,
                "passed_batches": passed,
                "first_oom_batch": min(oom_batches) if oom_batches else None,
                "max_tested_passing_batch": max(passed) if passed else None,
                "all_planned_batches_passed": bool(passed) and len(passed) == len(planned),
                "oom_boundary_observed": bool(oom_batches),
            }
        )
    return envelope


@dataclass(frozen=True)
class BenchmarkConfig:
    workloads: tuple[str, ...] = ("text", "image", "image_text")
    dimensions: tuple[int, ...] = (4096, 1024)
    batches: tuple[int, ...] = (1, 2, 4)
    warmup_iterations: int = 2
    measured_iterations: int = 20

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("workloads must not be empty")
        if not self.dimensions or any(int(value) <= 0 for value in self.dimensions):
            raise ValueError("dimensions must contain positive integers")
        if not self.batches or any(int(value) <= 0 for value in self.batches):
            raise ValueError("batches must contain positive integers")
        normalized_batches = tuple(int(value) for value in self.batches)
        if any(left >= right for left, right in zip(normalized_batches, normalized_batches[1:])):
            raise ValueError("batches must be strictly increasing")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be positive")


def _synchronize_cuda(torch_module: Any) -> None:
    for index in range(int(torch_module.cuda.device_count())):
        torch_module.cuda.synchronize(index)


def _reset_peak_memory(torch_module: Any) -> None:
    for index in range(int(torch_module.cuda.device_count())):
        torch_module.cuda.reset_peak_memory_stats(index)


def _cleanup_after_oom(torch_module: Any) -> None:
    gc.collect()
    empty_cache = getattr(torch_module.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _gpu_peak_mib(snapshot: Sequence[dict[str, int]]) -> dict[str, float]:
    return {
        f"gpu{int(row['index'])}": float(row["max_allocated_bytes"]) / (1024.0 * 1024.0)
        for row in snapshot
    }


def _process_memory_summary(snapshot: dict[str, int]) -> dict[str, float]:
    return {
        "rss_mib": float(snapshot["rss_bytes"]) / (1024.0 * 1024.0),
        "peak_rss_mib": float(snapshot["peak_rss_bytes"]) / (1024.0 * 1024.0),
    }


def _validate_embedding(embedding: Any, batch_size: int, dimension: int) -> dict[str, Any]:
    report = check_embedding(embedding, dimension)
    if report["shape"][0] != batch_size:
        raise RuntimeError(
            f"Unexpected benchmark batch shape {report['shape']}; expected [{batch_size}, {dimension}]"
        )
    if not 0.98 <= report["norm_min"] <= 1.02 or not 0.98 <= report["norm_max"] <= 1.02:
        raise RuntimeError(f"Benchmark embedding is not L2-normalized: {report}")
    return report


def _invoke_workload(runtime: Any, workload: str, payload: Any, dimension: int):
    runtime_dimension = None if dimension == 4096 else dimension
    if workload == "text":
        return runtime.embed_texts(payload, dimension=runtime_dimension)
    if workload == "image":
        return runtime.embed_images(payload, dimension=runtime_dimension)
    if workload == "image_text":
        return runtime.embed_image_texts(payload, dimension=runtime_dimension)
    raise ValueError(f"Unsupported benchmark workload: {workload}")


def _capture_completion_memory(runtime: Any, record: dict[str, Any]) -> None:
    process_after = process_snapshot()
    gpu_after = torch_cuda_memory_snapshot(runtime.torch)
    record["process_after"] = process_after
    record["torch_cuda_memory_after"] = gpu_after
    record["gpu_peak_mib"] = _gpu_peak_mib(gpu_after)
    record["memory"] = _process_memory_summary(process_after)


def measure_series(
    runtime: Any,
    *,
    workload: str,
    batch_size: int,
    dimension: int,
    warmup_iterations: int,
    measured_iterations: int,
    payload_factory: Callable[[str, int], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measured_iterations <= 0:
        raise ValueError("measured_iterations must be positive")

    payload = payload_factory(workload, batch_size)
    if len(payload) != batch_size:
        raise ValueError(f"payload_factory returned {len(payload)} items for batch {batch_size}")

    record: dict[str, Any] = {
        "workload": workload,
        "batch": batch_size,
        "dimension": dimension,
        "warmup_iterations_requested": warmup_iterations,
        "warmup_iterations_completed": 0,
        "measured_iterations_requested": measured_iterations,
        "measured_iterations_completed": 0,
        "latency_seconds": [],
        "summary": None,
        "embedding": None,
        "oom": None,
        "status": "RUNNING",
    }

    phase = "warmup"
    try:
        for _ in range(warmup_iterations):
            embedding = _invoke_workload(runtime, workload, payload, dimension)
            _synchronize_cuda(runtime.torch)
            record["embedding"] = _validate_embedding(embedding, batch_size, dimension)
            record["warmup_iterations_completed"] += 1

        _synchronize_cuda(runtime.torch)
        _reset_peak_memory(runtime.torch)
        record["process_before"] = process_snapshot()
        record["torch_cuda_memory_before"] = torch_cuda_memory_snapshot(runtime.torch)

        phase = "measurement"
        for _ in range(measured_iterations):
            _synchronize_cuda(runtime.torch)
            started = clock()
            embedding = _invoke_workload(runtime, workload, payload, dimension)
            _synchronize_cuda(runtime.torch)
            elapsed = clock() - started
            report = _validate_embedding(embedding, batch_size, dimension)
            record["latency_seconds"].append(float(elapsed))
            record["embedding"] = report
            record["measured_iterations_completed"] += 1

        record["summary"] = summarize_latencies(record["latency_seconds"], batch_size)
        record["status"] = "PASS"
        _capture_completion_memory(runtime, record)
        return record
    except BaseException as exc:
        if not is_cuda_oom(exc, runtime.torch):
            raise
        if record["latency_seconds"]:
            record["summary"] = summarize_latencies(record["latency_seconds"], batch_size)
        record["status"] = "OOM"
        record["oom"] = {
            "phase": phase,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        _capture_completion_memory(runtime, record)
        _cleanup_after_oom(runtime.torch)
        return record


def run_matrix(
    runtime: Any,
    config: BenchmarkConfig,
    payload_factory: Callable[[str, int], Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for workload in config.workloads:
        for dimension in config.dimensions:
            progression = run_batch_progression(
                config.batches,
                lambda batch, workload=workload, dimension=dimension: measure_series(
                    runtime,
                    workload=workload,
                    batch_size=batch,
                    dimension=dimension,
                    warmup_iterations=config.warmup_iterations,
                    measured_iterations=config.measured_iterations,
                    payload_factory=payload_factory,
                    clock=clock,
                ),
                on_record=on_record,
            )
            records.extend(progression)
    return records, build_safe_envelope(records)
