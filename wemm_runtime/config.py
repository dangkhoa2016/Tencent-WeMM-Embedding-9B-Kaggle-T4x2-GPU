from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RuntimeConfig:
    """Provider-neutral WeMM runtime configuration.

    No Kaggle paths are assumed. The caller supplies a local model path and the
    desired CUDA placement policy. The same object can be used from Kaggle,
    Docker, a VPS, Modal, Lightning, or another provider.
    """

    model_path: Path
    device_map: str = "balanced"
    gpu_ids: tuple[int, ...] = (0, 1)
    per_gpu_mib: int | None = 14200
    cpu_memory_mib: int = 2048
    dtype: str = "float16"
    attn_implementation: str = "sdpa"
    local_files_only: bool = True
    offline: bool = True
    allow_cpu_offload: bool = False
    allowed_dimensions: tuple[int, ...] = (4096, 1024)

    def normalized_model_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    def max_memory(self) -> Mapping[object, str] | None:
        if self.per_gpu_mib is None:
            return None
        if self.per_gpu_mib < 1024:
            raise ValueError("per_gpu_mib is implausibly small")
        result: dict[object, str] = {
            int(gpu): f"{int(self.per_gpu_mib)}MiB" for gpu in self.gpu_ids
        }
        if self.cpu_memory_mib > 0:
            result["cpu"] = f"{int(self.cpu_memory_mib)}MiB"
        return result


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://127.0.0.1:6333"
    collections: Mapping[int, str] = field(
        default_factory=lambda: {
            4096: "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1",
            1024: "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1",
        }
    )
    timeout_s: float = 30.0

    def collection(self, dimension: int) -> str:
        try:
            return str(self.collections[int(dimension)])
        except KeyError as exc:
            raise ValueError(
                f"No Qdrant collection configured for dimension={dimension}"
            ) from exc
