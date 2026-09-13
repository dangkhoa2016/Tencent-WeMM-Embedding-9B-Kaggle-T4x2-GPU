from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DeviceMapReport:
    gpu_ids: tuple[int, ...]
    offload_targets: tuple[str, ...]
    module_counts: dict[int, int]
    raw: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def build_max_memory(
    gpu_ids: Iterable[int] = (0, 1),
    per_gpu_mib: int = 14200,
    cpu_memory_mib: int = 2048,
) -> dict[Any, str]:
    if per_gpu_mib < 1024:
        raise ValueError("per_gpu_mib is implausibly small")
    result: dict[Any, str] = {
        int(gpu): f"{int(per_gpu_mib)}MiB" for gpu in tuple(gpu_ids)
    }
    if cpu_memory_mib > 0:
        result["cpu"] = f"{int(cpu_memory_mib)}MiB"
    return result


def _normalize_target(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value).lower()
    if text.isdigit():
        return f"cuda:{int(text)}"
    return text


def validate_device_map(
    mapping: Mapping[str, Any],
    *,
    required_gpu_ids: Iterable[int] = (0, 1),
    allow_cpu_offload: bool = False,
) -> DeviceMapReport:
    if not mapping:
        raise RuntimeError("Model has no hf_device_map; cannot prove CUDA placement")

    required = tuple(sorted({int(x) for x in required_gpu_ids}))
    module_counts = {gpu: 0 for gpu in required}
    gpu_ids: set[int] = set()
    offload: set[str] = set()
    normalized: dict[str, str] = {}

    for module, target in mapping.items():
        target_name = _normalize_target(target)
        normalized[str(module)] = target_name
        if target_name in {"cpu", "disk"}:
            if target_name == "cpu" and allow_cpu_offload:
                continue
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
        raise RuntimeError(
            "Model device map contains forbidden offload/unsupported targets: "
            f"{sorted(offload)}"
        )
    if not set(required).issubset(gpu_ids):
        raise RuntimeError(
            f"Model must place modules on GPUs {required}; observed GPUs={sorted(gpu_ids)}"
        )
    return DeviceMapReport(
        gpu_ids=tuple(sorted(gpu_ids)),
        offload_targets=tuple(sorted(offload)),
        module_counts=module_counts,
        raw=normalized,
    )


def cuda_preflight(required_gpu_ids: Iterable[int] = (0, 1)) -> dict:
    import torch

    required = tuple(sorted({int(x) for x in required_gpu_ids}))
    count = torch.cuda.device_count()
    if required and max(required) >= count:
        raise RuntimeError(
            f"Required CUDA GPU ids {required}; torch reports device_count={count}"
        )
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
