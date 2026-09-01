from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import RuntimeConfig
from .model import load_runtime


def _load_config(path: Path) -> RuntimeConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuntimeConfig(
        model_path=Path(raw["model_path"]),
        device_map=raw.get("device_map", "balanced"),
        gpu_ids=tuple(raw.get("gpu_ids", [0, 1])),
        per_gpu_mib=raw.get("per_gpu_mib", 14200),
        cpu_memory_mib=int(raw.get("cpu_memory_mib", 2048)),
        dtype=raw.get("dtype", "float16"),
        attn_implementation=raw.get("attn_implementation", "sdpa"),
        local_files_only=bool(raw.get("local_files_only", True)),
        offline=bool(raw.get("offline", True)),
        allow_cpu_offload=bool(raw.get("allow_cpu_offload", False)),
        allowed_dimensions=tuple(raw.get("allowed_dimensions", [4096, 1024])),
    )


def emit(payload) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        runtime = load_runtime(_load_config(Path(args.config)))
        emit(
            {
                "ok": True,
                "type": "ready",
                "pid": __import__("os").getpid(),
                "input_device": runtime.input_device,
                "gpu_ids": list(runtime.device_map_report.gpu_ids),
                "module_counts": runtime.device_map_report.module_counts,
                "offload_targets": list(runtime.device_map_report.offload_targets),
                "allowed_dimensions": list(runtime.allowed_dimensions),
                "model_identity": runtime.identity.to_dict(),
                "dtype": str(next(runtime.model.parameters()).dtype).replace("torch.", ""),
            }
        )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            op = request.get("op")
            if op == "shutdown":
                emit({"ok": True, "type": "shutdown"})
                return 0
            if op == "embed_text":
                tensor = runtime.embed_text(
                    str(request["text"]),
                    dimension=int(request.get("dimension", 4096)),
                )
            elif op == "embed_image":
                from PIL import Image

                with Image.open(request["image_path"]) as source:
                    image = source.convert("RGB")
                tensor = runtime.embed_image(
                    image,
                    dimension=int(request.get("dimension", 4096)),
                )
            else:
                raise ValueError(f"Unknown worker operation: {op!r}")

            vector = tensor[0].detach().float().cpu().tolist()
            emit(
                {
                    "ok": True,
                    "type": "embedding",
                    "dimension": len(vector),
                    "vector": vector,
                }
            )
    except BaseException as exc:
        emit(
            {
                "ok": False,
                "type": "error",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
