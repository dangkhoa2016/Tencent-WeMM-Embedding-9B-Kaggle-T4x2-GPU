from __future__ import annotations

import os
from pathlib import Path

from .config import RuntimeConfig
from .model import load_runtime


def runtime_config_from_env() -> RuntimeConfig:
    model_path = os.environ.get("WEMM_MODEL_PATH")
    if not model_path:
        raise RuntimeError("WEMM_MODEL_PATH is required")
    gpu_ids = tuple(
        int(x.strip())
        for x in os.environ.get("WEMM_GPU_IDS", "0,1").split(",")
        if x.strip()
    )
    per_gpu_raw = os.environ.get("WEMM_PER_GPU_MIB", "14200").strip().lower()
    per_gpu_mib = None if per_gpu_raw in {"", "none", "auto"} else int(per_gpu_raw)
    return RuntimeConfig(
        model_path=Path(model_path),
        device_map=os.environ.get("WEMM_DEVICE_MAP", "balanced"),
        gpu_ids=gpu_ids,
        per_gpu_mib=per_gpu_mib,
        cpu_memory_mib=int(os.environ.get("WEMM_CPU_MEMORY_MIB", "2048")),
        dtype=os.environ.get("WEMM_DTYPE", "float16"),
        allow_cpu_offload=os.environ.get("WEMM_ALLOW_CPU_OFFLOAD", "0") == "1",
        offline=os.environ.get("WEMM_OFFLINE", "1") != "0",
    )


def load_runtime_from_env():
    return load_runtime(runtime_config_from_env())
