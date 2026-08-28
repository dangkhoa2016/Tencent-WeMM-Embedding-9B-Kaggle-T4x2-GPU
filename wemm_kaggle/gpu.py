from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DeviceMapReport:
    gpu_ids: tuple[int, ...]
    offload_targets: tuple[str, ...]
    module_counts: dict[int, int]
    raw: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def build_max_memory(per_gpu_mib: int = 14200) -> dict[Any, str]:
    if per_gpu_mib < 1024:
        raise ValueError("per_gpu_mib is implausibly small")
    return {0: f"{per_gpu_mib}MiB", 1: f"{per_gpu_mib}MiB", "cpu": "2GiB"}


def _normalize_target(value: Any) -> str:
    text = str(value).lower()
    if isinstance(value, int):
        return f"cuda:{value}"
    if text.isdigit():
        return f"cuda:{int(text)}"
    if text.startswith("cuda:"):
        return text
    return text


def validate_device_map(mapping: Mapping[str, Any]) -> DeviceMapReport:
    if not mapping:
        raise RuntimeError("Model has no hf_device_map; cannot prove dual-GPU dispatch")
    module_counts = {0: 0, 1: 0}
    offload: set[str] = set()
    normalized: dict[str, str] = {}
    gpu_ids: set[int] = set()
    for module, target in mapping.items():
        target_name = _normalize_target(target)
        normalized[str(module)] = target_name
        if target_name in {"cpu", "disk"}:
            offload.add(target_name)
            continue
        if target_name.startswith("cuda:"):
            try:
                gpu = int(target_name.split(":", 1)[1])
            except ValueError:
                offload.add(target_name)
                continue
            gpu_ids.add(gpu)
            if gpu in module_counts:
                module_counts[gpu] += 1
        else:
            offload.add(target_name)
    if offload:
        raise RuntimeError(f"Model device map contains forbidden offload/unsupported targets: {sorted(offload)}")
    if not {0, 1}.issubset(gpu_ids):
        raise RuntimeError(f"Model must place modules on both GPU 0 and GPU 1; observed GPUs={sorted(gpu_ids)}")
    return DeviceMapReport(
        gpu_ids=tuple(sorted(gpu_ids)),
        offload_targets=tuple(sorted(offload)),
        module_counts=module_counts,
        raw=normalized,
    )


def cuda_preflight(required: int = 2) -> dict:
    import torch

    count = torch.cuda.device_count()
    if count < required:
        raise RuntimeError(f"Need at least {required} CUDA GPUs; torch reports {count}")
    devices = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": count,
        "devices": devices,
    }
