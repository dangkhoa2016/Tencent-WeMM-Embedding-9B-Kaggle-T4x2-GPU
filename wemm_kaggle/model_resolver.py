from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import offline

REQUIRED_FILES = (
    "config.json",
    "modeling_wemm_embedding.py",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ModelIdentity:
    path: str
    matryoshka_dimensions: tuple[int, ...]
    weight_files: tuple[str, ...]
    weight_bytes: int
    config_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _weight_files(path: Path) -> list[Path]:
    monolithic = path / "model.safetensors"
    if monolithic.is_file():
        return [monolithic]
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return []
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    shards = [path / name for name in names]
    if not names or not all(item.is_file() for item in shards):
        return []
    return [index, *shards]


def validate_model_dir(path: Path) -> ModelIdentity:
    path = Path(path).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete WeMM model directory {path}; missing: {', '.join(missing)}")
    weights = _weight_files(path)
    if not weights:
        raise ValueError(f"No complete safetensors weights found under {path}")
    try:
        config_bytes = (path / "config.json").read_bytes()
        config = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid config.json under {path}: {exc}") from exc
    dims = tuple(int(x) for x in config.get("matryoshka_dimensions", []))
    if 4096 not in dims:
        raise ValueError(f"WeMM-Embedding-9B config must advertise Matryoshka dimension 4096: {path}")
    auto_model = str(config.get("auto_map", {}).get("AutoModel", ""))
    if "modeling_wemm_embedding" not in auto_model:
        raise ValueError(f"config.json does not map AutoModel to WeMM custom code: {path}")
    return ModelIdentity(
        path=str(path),
        matryoshka_dimensions=dims,
        weight_files=tuple(item.name for item in weights),
        weight_bytes=sum(item.stat().st_size for item in weights if item.suffix == ".safetensors"),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def _candidate_dirs(root: Path) -> list[Path]:
    found: list[Path] = []
    for config in root.rglob("config.json"):
        candidate = config.parent
        try:
            validate_model_dir(candidate)
        except ValueError:
            continue
        found.append(candidate.resolve())
    return sorted(set(found))


def resolve_model_dir(root: Path = Path("/kaggle/input"), explicit: str | None = None) -> Path:
    root = Path(root).expanduser().resolve()
    if explicit:
        path = offline.assert_kaggle_input_path(Path(explicit))
        validate_model_dir(path)
        return path
    candidates = _candidate_dirs(root)
    preferred = [p for p in candidates if "wemm-embedding-9b" in str(p).lower().replace("_", "-")]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        candidates = preferred
    if not candidates:
        raise FileNotFoundError(
            "No complete WeMM-Embedding-9B model found under /kaggle/input. Attach the Kaggle Model first."
        )
    if len(candidates) > 1:
        rendered = "\n  - ".join(str(p) for p in candidates)
        raise RuntimeError(
            "Multiple complete WeMM model candidates found; set KAGGLE_MODEL_DIR to one exact mounted path:\n  - "
            + rendered
        )
    return candidates[0]
