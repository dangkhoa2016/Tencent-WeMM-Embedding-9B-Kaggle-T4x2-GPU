from __future__ import annotations

import pytest
import torch


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.sync_calls = []
        self.reset_calls = []
        self.empty_cache_calls = 0

    def device_count(self):
        return 2

    def synchronize(self, index):
        self.sync_calls.append(index)

    def reset_peak_memory_stats(self, index):
        self.reset_calls.append(index)

    def empty_cache(self):
        self.empty_cache_calls += 1

    def memory_allocated(self, index):
        return (index + 1) * 100 * 1024 * 1024

    def memory_reserved(self, index):
        return (index + 1) * 120 * 1024 * 1024

    def max_memory_allocated(self, index):
        return (index + 1) * 150 * 1024 * 1024

    def max_memory_reserved(self, index):
        return (index + 1) * 180 * 1024 * 1024


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class FakeRuntime:
    def __init__(self, fail_call=None):
        self.torch = FakeTorch()
        self.calls = []
        self.fail_call = fail_call

    def _return(self, workload, payload, dimension):
        self.calls.append((workload, len(payload), dimension))
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise self.torch.cuda.OutOfMemoryError("CUDA out of memory in fake runtime")
        effective_dimension = dimension or 4096
        values = torch.arange(1, effective_dimension + 1, dtype=torch.float32).repeat(len(payload), 1)
        return torch.nn.functional.normalize(values, dim=-1)

    def embed_texts(self, payload, dimension=None):
        return self._return("text", payload, dimension)

    def embed_images(self, payload, dimension=None):
        return self._return("image", payload, dimension)

    def embed_image_texts(self, payload, dimension=None):
        return self._return("image_text", payload, dimension)


def payload_factory(workload, batch):
    if workload == "image_text":
        return [(object(), f"text-{i}") for i in range(batch)]
    return [f"item-{i}" for i in range(batch)]


def test_measure_series_excludes_warmups_and_synchronizes_timed_iterations():
    from wemm_kaggle.benchmark import measure_series

    runtime = FakeRuntime()
    times = iter([10.0, 10.1, 20.0, 20.2, 30.0, 30.3])

    record = measure_series(
        runtime,
        workload="text",
        batch_size=2,
        dimension=1024,
        warmup_iterations=2,
        measured_iterations=3,
        payload_factory=payload_factory,
        clock=lambda: next(times),
    )

    assert record["status"] == "PASS"
    assert record["warmup_iterations_completed"] == 2
    assert record["measured_iterations_completed"] == 3
    assert record["latency_seconds"] == pytest.approx([0.1, 0.2, 0.3])
    assert len(runtime.calls) == 5
    assert runtime.torch.cuda.reset_calls == [0, 1]
    # two synchronizations around each timed iteration, each across both GPUs
    assert runtime.torch.cuda.sync_calls[-12:] == [0, 1] * 6
    assert record["summary"]["total_items"] == 6
    assert record["embedding"]["shape"] == [2, 1024]
    assert record["gpu_peak_mib"] == {"gpu0": 150.0, "gpu1": 300.0}
    assert record["memory"]["rss_mib"] > 0
    assert record["memory"]["peak_rss_mib"] > 0


def test_measure_series_records_oom_phase_and_preserves_completed_samples():
    from wemm_kaggle.benchmark import measure_series

    runtime = FakeRuntime(fail_call=3)  # warmup succeeds, first measured succeeds, second measured OOMs
    times = iter([1.0, 1.2, 2.0])

    record = measure_series(
        runtime,
        workload="image",
        batch_size=4,
        dimension=4096,
        warmup_iterations=1,
        measured_iterations=3,
        payload_factory=payload_factory,
        clock=lambda: next(times),
    )

    assert record["status"] == "OOM"
    assert record["warmup_iterations_completed"] == 1
    assert record["measured_iterations_completed"] == 1
    assert record["latency_seconds"] == pytest.approx([0.2])
    assert record["summary"]["total_items"] == 4
    assert record["oom"]["phase"] == "measurement"
    assert record["oom"]["exception_type"] == "OutOfMemoryError"
    assert "CUDA out of memory" in record["oom"]["message"]
    assert runtime.torch.cuda.empty_cache_calls == 1


def test_run_matrix_applies_fail_stop_per_workload_dimension_group():
    from wemm_kaggle.benchmark import BenchmarkConfig, run_matrix

    class MatrixRuntime(FakeRuntime):
        def _return(self, workload, payload, dimension):
            if workload == "text" and dimension is None and len(payload) == 2:
                raise self.torch.cuda.OutOfMemoryError("CUDA out of memory at text B2")
            return super()._return(workload, payload, dimension)

    runtime = MatrixRuntime()
    config = BenchmarkConfig(
        workloads=("text", "image"),
        dimensions=(4096,),
        batches=(1, 2, 4),
        warmup_iterations=0,
        measured_iterations=1,
    )
    times = iter(float(i) / 10 for i in range(100))

    records, envelope = run_matrix(
        runtime,
        config,
        payload_factory,
        clock=lambda: next(times),
    )

    text = [row for row in records if row["workload"] == "text"]
    image = [row for row in records if row["workload"] == "image"]
    assert [row["status"] for row in text] == ["PASS", "OOM", "SKIPPED_AFTER_OOM"]
    assert [row["status"] for row in image] == ["PASS", "PASS", "PASS"]
    text_env = next(row for row in envelope if row["workload"] == "text")
    assert text_env["max_tested_passing_batch"] == 1
    assert text_env["first_oom_batch"] == 2
    assert text_env["all_planned_batches_passed"] is False
    assert text_env["oom_boundary_observed"] is True


def test_full_4096_uses_native_embedding_without_mrl_truncation():
    from wemm_kaggle.benchmark import measure_series

    runtime = FakeRuntime()
    times = iter([1.0, 1.1])
    record = measure_series(
        runtime,
        workload="text",
        batch_size=1,
        dimension=4096,
        warmup_iterations=0,
        measured_iterations=1,
        payload_factory=payload_factory,
        clock=lambda: next(times),
    )

    assert record["status"] == "PASS"
    assert runtime.calls == [("text", 1, None)]
    assert record["embedding"]["shape"] == [1, 4096]


def test_benchmark_config_requires_strictly_increasing_batch_progression():
    from wemm_kaggle.benchmark import BenchmarkConfig

    with pytest.raises(ValueError, match="strictly increasing"):
        BenchmarkConfig(batches=(1, 4, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        BenchmarkConfig(batches=(1, 2, 2))
