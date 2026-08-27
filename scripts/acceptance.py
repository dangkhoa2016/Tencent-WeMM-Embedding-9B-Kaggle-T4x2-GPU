#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from wemm_kaggle.embedding import check_embedding
from wemm_kaggle.evidence import atomic_write_json, nvidia_smi_snapshot, process_snapshot, torch_cuda_memory_snapshot
from wemm_kaggle.gpu import cuda_preflight
from wemm_kaggle.model_resolver import resolve_model_dir, validate_model_dir
from wemm_kaggle.offline import enforce_offline
from wemm_kaggle.runtime import load_local_runtime


def generated_test_image() -> Image.Image:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((72, 72, 184, 184), fill="red")
    draw.ellipse((104, 104, 152, 152), fill="yellow")
    return image


def timed_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def synchronize_two(torch_module) -> None:
    for index in (0, 1):
        torch_module.cuda.synchronize(index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-gpu-mib", type=int, default=int(os.environ.get("WEMM_PER_GPU_MIB", "14200")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-non-t4", action="store_true")
    args = parser.parse_args()

    enforce_offline()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or Path(os.environ.get("WEMM_RUN_ROOT", "/kaggle/working/wemm-embedding-9b-t4x2-runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "acceptance.json"
    evidence: dict = {"run_id": run_id, "status": "RUNNING", "stages": {}}

    try:
        evidence["stages"]["process_before"] = process_snapshot()
        evidence["stages"]["nvidia_smi_before"] = nvidia_smi_snapshot()
        gpu = cuda_preflight(2)
        evidence["gpu_preflight"] = gpu
        names = [row["name"] for row in gpu["devices"][:2]]
        allow_non_t4 = args.allow_non_t4 or os.environ.get("ALLOW_NON_T4") == "1"
        if not allow_non_t4 and not all("T4" in name.upper() for name in names):
            raise RuntimeError(f"Expected Kaggle T4 x2; observed {names}")

        model_dir = resolve_model_dir(Path("/kaggle/input"), os.environ.get("KAGGLE_MODEL_DIR"))
        identity = validate_model_dir(model_dir)
        evidence["model_source"] = "kaggle_input"
        evidence["model"] = identity.to_dict()
        evidence["per_gpu_mib"] = args.per_gpu_mib
        atomic_write_json(evidence_path, evidence)

        load_start = time.perf_counter()
        runtime = load_local_runtime(model_dir, per_gpu_mib=args.per_gpu_mib)
        load_seconds = time.perf_counter() - load_start
        evidence["load"] = {
            "seconds": load_seconds,
            "input_device": runtime.input_device,
            "device_map": runtime.device_map_report.to_dict(),
            "torch_cuda_memory": torch_cuda_memory_snapshot(runtime.torch),
            "process": process_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
        }
        atomic_write_json(evidence_path, evidence)

        image = generated_test_image()
        text_en = "A small red square with a yellow circle on a white background."
        text_vi = "Một hình vuông nhỏ màu đỏ có vòng tròn màu vàng trên nền trắng."

        runtime.torch.cuda.reset_peak_memory_stats(0)
        runtime.torch.cuda.reset_peak_memory_stats(1)

        en, en_s = timed_call(lambda: runtime.embed_text(text_en))
        vi, vi_s = timed_call(lambda: runtime.embed_text(text_vi))
        image_emb, image_s = timed_call(lambda: runtime.embed_image(image))
        image_text_emb, image_text_s = timed_call(lambda: runtime.embed_image_text(image, "Represent this image."))
        mrl, mrl_s = timed_call(lambda: runtime.embed_text(text_en, dimension=1024))
        warm, warm_s = timed_call(lambda: runtime.embed_text(text_en))
        synchronize_two(runtime.torch)

        checks = {
            "text_en": check_embedding(en, 4096),
            "text_vi": check_embedding(vi, 4096),
            "image": check_embedding(image_emb, 4096),
            "image_text": check_embedding(image_text_emb, 4096),
            "mrl_1024": check_embedding(mrl, 1024),
            "warm_repeat": check_embedding(warm, 4096),
        }
        for name, report in checks.items():
            if not 0.98 <= report["norm_min"] <= 1.02 or not 0.98 <= report["norm_max"] <= 1.02:
                raise RuntimeError(f"{name} embedding is not L2-normalized: {report}")

        cosine = runtime.torch.nn.functional.cosine_similarity
        evidence["inference"] = {
            "checks": checks,
            "latency_seconds": {
                "text_en": en_s,
                "text_vi": vi_s,
                "image": image_s,
                "image_text": image_text_s,
                "mrl_1024": mrl_s,
                "warm_repeat": warm_s,
            },
            "similarities": {
                "en_vi": float(cosine(en.float(), vi.float()).item()),
                "image_image_text": float(cosine(image_emb.float(), image_text_emb.float()).item()),
            },
            "torch_cuda_memory": torch_cuda_memory_snapshot(runtime.torch),
            "process": process_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
        }
        evidence["status"] = "PASS"
        evidence["result"] = "WEMM_T4X2_ACCEPTANCE_PASS"
        atomic_write_json(evidence_path, evidence)
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "evidence": str(evidence_path)}, indent=2))
        return 0
    except Exception as exc:
        evidence["status"] = "FAIL"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["stages"]["process_failure"] = process_snapshot()
        evidence["stages"]["nvidia_smi_failure"] = nvidia_smi_snapshot()
        atomic_write_json(evidence_path, evidence)
        print(json.dumps({"status": "FAIL", "run_dir": str(run_dir), "evidence": str(evidence_path), "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
