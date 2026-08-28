#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from wemm_kaggle.benchmark import BenchmarkConfig, run_matrix
from wemm_kaggle.evidence import (
    atomic_write_json,
    nvidia_smi_snapshot,
    package_evidence_zip,
    process_snapshot,
    torch_cuda_memory_snapshot,
)
from wemm_kaggle.gpu import cuda_preflight
from wemm_kaggle.model_resolver import resolve_model_dir, validate_model_dir
from wemm_kaggle.offline import enforce_offline
from wemm_kaggle.runtime import load_local_runtime


TEXT_FIXTURES = (
    "A small red square with a yellow circle on a white background.",
    "Một hình vuông nhỏ màu đỏ có vòng tròn màu vàng trên nền trắng.",
    "A bicycle is parked beside a tree on a quiet street.",
    "Một chiếc xe đạp dựng cạnh một cái cây trên con phố yên tĩnh.",
)


class OperatorLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return parsed


def generated_benchmark_image(index: int) -> Image.Image:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    offset = (index % 4) * 8
    draw.rectangle((56 + offset, 64, 176 + offset, 184), fill="red")
    draw.ellipse((96, 96 + offset, 152, 152 + offset), fill="yellow")
    return image


def make_payload_factory():
    def payload_factory(workload: str, batch_size: int):
        if workload == "text":
            return [TEXT_FIXTURES[index % len(TEXT_FIXTURES)] for index in range(batch_size)]
        images = [generated_benchmark_image(index) for index in range(batch_size)]
        if workload == "image":
            return images
        if workload == "image_text":
            return [
                (image, f"Represent this image for multimodal retrieval, sample {index}.")
                for index, image in enumerate(images)
            ]
        raise ValueError(f"Unsupported benchmark workload: {workload}")

    return payload_factory


def source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    version_path = root / "VERSION"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else None
    unavailable = {
        "version": version,
        "git_available": False,
        "git_commit": None,
        "git_branch": None,
        "git_tree": None,
        "git_dirty": None,
        "git_status_porcelain": [],
    }
    if not (root / ".git").exists():
        return unavailable
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return unavailable
        porcelain = [line for line in proc.stdout.splitlines() if line]
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return unavailable
        head_lines = proc.stdout.splitlines()
        if len(head_lines) < 2:
            return unavailable
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return unavailable
        return {
            "version": version,
            "git_available": True,
            "git_commit": head_lines[0].strip(),
            "git_tree": head_lines[1].strip(),
            "git_branch": proc.stdout.strip() or None,
            "git_dirty": bool(porcelain),
            "git_status_porcelain": porcelain,
        }
    except (OSError, subprocess.TimeoutExpired):
        return unavailable


def assert_clean_source(source: Mapping[str, Any]) -> None:
    if source.get("git_available") and source.get("git_dirty"):
        raise RuntimeError(
            "Benchmark source tree is dirty; commit or restore source changes before collecting release evidence."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark WeMM-Embedding-9B on the frozen Kaggle T4x2 runtime")
    parser.add_argument("--per-gpu-mib", type=int, default=int(os.environ.get("WEMM_PER_GPU_MIB", "14200")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-non-t4", action="store_true")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--dimensions", default="4096,1024")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=20)
    args = parser.parse_args()

    batches = parse_int_csv(args.batches, name="batches")
    dimensions = parse_int_csv(args.dimensions, name="dimensions")
    config = BenchmarkConfig(
        batches=batches,
        dimensions=dimensions,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
    )

    enforce_offline()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(os.environ.get("WEMM_BENCHMARK_ROOT", "/kaggle/working/wemm-embedding-9b-t4x2-benchmarks"))
    run_dir = args.output_dir or run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "benchmark.json"
    operator = OperatorLog(run_dir / "operator.log")
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "RUNNING",
        "source": source_identity(),
        "config": asdict(config),
        "records": [],
        "safe_envelope": [],
        "stages": {},
    }
    atomic_write_json(evidence_path, evidence)
    operator.emit("benchmark_run_start", run_id=run_id, config=evidence["config"])

    try:
        assert_clean_source(evidence["source"])
        evidence["stages"]["process_before"] = process_snapshot()
        evidence["stages"]["nvidia_smi_before"] = nvidia_smi_snapshot()
        gpu = cuda_preflight(2)
        evidence["gpu_preflight"] = gpu
        names = [row["name"] for row in gpu["devices"][:2]]
        allow_non_t4 = args.allow_non_t4 or os.environ.get("ALLOW_NON_T4") == "1"
        if not allow_non_t4 and not all("T4" in name.upper() for name in names):
            raise RuntimeError(f"Expected Kaggle T4 x2; observed {names}")
        operator.emit("gpu_preflight_pass", devices=names)

        model_dir = resolve_model_dir(Path("/kaggle/input"), os.environ.get("KAGGLE_MODEL_DIR"))
        identity = validate_model_dir(model_dir)
        evidence["model_source"] = "kaggle_input"
        evidence["model"] = identity.to_dict()
        evidence["per_gpu_mib"] = args.per_gpu_mib
        atomic_write_json(evidence_path, evidence)
        operator.emit("model_resolved", path=str(model_dir), weight_bytes=identity.weight_bytes)

        load_started = time.perf_counter()
        runtime = load_local_runtime(model_dir, per_gpu_mib=args.per_gpu_mib)
        load_seconds = time.perf_counter() - load_started
        evidence["load"] = {
            "seconds": load_seconds,
            "input_device": runtime.input_device,
            "device_map": runtime.device_map_report.to_dict(),
            "torch_cuda_memory": torch_cuda_memory_snapshot(runtime.torch),
            "process": process_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
        }
        atomic_write_json(evidence_path, evidence)
        operator.emit(
            "model_load_pass",
            seconds=load_seconds,
            input_device=runtime.input_device,
            module_counts=runtime.device_map_report.module_counts,
        )

        planned_series = len(config.workloads) * len(config.dimensions) * len(config.batches)
        operator.emit("benchmark_matrix_start", planned_series=planned_series)

        def on_record(record: dict[str, Any]) -> None:
            evidence["records"].append(record)
            atomic_write_json(evidence_path, evidence)
            operator.emit(
                "benchmark_series_complete",
                workload=record.get("workload"),
                dimension=record.get("dimension"),
                batch=record.get("batch"),
                status=record.get("status"),
                summary=record.get("summary"),
                oom=record.get("oom"),
            )

        records, envelope = run_matrix(
            runtime,
            config,
            make_payload_factory(),
            on_record=on_record,
        )
        evidence["records"] = records
        evidence["safe_envelope"] = envelope
        oom_count = sum(1 for row in records if row.get("status") == "OOM")
        if oom_count:
            evidence["status"] = "PARTIAL"
            evidence["result"] = "WEMM_T4X2_BENCHMARK_PARTIAL"
        else:
            evidence["status"] = "PASS"
            evidence["result"] = "WEMM_T4X2_BENCHMARK_PASS"
        evidence["stages"]["process_after"] = process_snapshot()
        evidence["stages"]["nvidia_smi_after"] = nvidia_smi_snapshot()
        evidence["stages"]["torch_cuda_memory_after"] = torch_cuda_memory_snapshot(runtime.torch)
        atomic_write_json(evidence_path, evidence)
        operator.emit(
            "benchmark_matrix_complete",
            status=evidence["status"],
            oom_count=oom_count,
            safe_envelope=envelope,
        )

        zip_path = run_dir.parent / f"wemm-embedding-9b-t4x2-benchmark-{run_id}.zip"
        operator.emit("evidence_package_start", zip_path=str(zip_path))
        package = package_evidence_zip(run_dir, zip_path)
        sha256_path = package["sha256_path"]
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "run_dir": str(run_dir),
                    "evidence": str(evidence_path),
                    "sha256_path": sha256_path,
                    "artifact": package,
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        evidence["status"] = "FAIL"
        evidence["result"] = "WEMM_T4X2_BENCHMARK_FAIL"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["stages"]["process_failure"] = process_snapshot()
        evidence["stages"]["nvidia_smi_failure"] = nvidia_smi_snapshot()
        atomic_write_json(evidence_path, evidence)
        operator.emit("benchmark_run_fail", error_type=type(exc).__name__, error=str(exc))
        package = None
        try:
            zip_path = run_dir.parent / f"wemm-embedding-9b-t4x2-benchmark-{run_id}.zip"
            package = package_evidence_zip(run_dir, zip_path)
        except Exception as package_exc:
            operator.emit("evidence_package_fail", error_type=type(package_exc).__name__, error=str(package_exc))
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "run_dir": str(run_dir),
                    "evidence": str(evidence_path),
                    "artifact": package,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
