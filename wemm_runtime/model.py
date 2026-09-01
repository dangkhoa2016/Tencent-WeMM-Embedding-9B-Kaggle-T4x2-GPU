from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .embedding import (
    build_messages,
    emit_padding_evidence,
    ensure_right_padding,
    pool_eos_embedding,
    truncate_mrl,
)
from .gpu import DeviceMapReport, validate_device_map

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
        raise ValueError(
            f"Incomplete Tencent WeMM-Embedding-9B model directory {path}; missing: {', '.join(missing)}"
        )
    weights = _weight_files(path)
    if not weights:
        raise ValueError(f"No complete safetensors weights found under {path}")
    config_bytes = (path / "config.json").read_bytes()
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config.json under {path}: {exc}") from exc
    dims = tuple(int(x) for x in config.get("matryoshka_dimensions", []))
    if 4096 not in dims:
        raise ValueError(
            f"Tencent WeMM-Embedding-9B config must advertise Matryoshka dimension 4096: {path}"
        )
    auto_model = str(config.get("auto_map", {}).get("AutoModel", ""))
    if "modeling_wemm_embedding" not in auto_model:
        raise ValueError(
            f"config.json does not map AutoModel to WeMM custom code: {path}"
        )
    return ModelIdentity(
        path=str(path),
        matryoshka_dimensions=dims,
        weight_files=tuple(item.name for item in weights),
        weight_bytes=sum(
            item.stat().st_size for item in weights if item.suffix == ".safetensors"
        ),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def enforce_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def install_embedding_dispatch_compat(model: Any, use_it: bool = True) -> None:
    if not use_it:
        return

    def _embedding(input_ids=None, attention_mask=None, **kwargs):
        model.model.rope_deltas = None
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return pool_eos_embedding(outputs.last_hidden_state, attention_mask)

    model._wemm_dispatch_compat = True
    model.embedding = _embedding


def _target_name(target: Any) -> str:
    if isinstance(target, int):
        return f"cuda:{target}"
    text = str(target).lower()
    if text.isdigit():
        return f"cuda:{int(text)}"
    return text


def _preferred_input_device(mapping: dict[str, Any]) -> str:
    for key, target in mapping.items():
        if "embed_tokens" in key:
            name = _target_name(target)
            if name.startswith("cuda:"):
                return name
    for target in mapping.values():
        name = _target_name(target)
        if name.startswith("cuda:"):
            return name
    raise RuntimeError("No CUDA execution device found in hf_device_map")


@dataclass
class WeMMRuntime:
    config: RuntimeConfig
    identity: ModelIdentity
    processor: Any
    model: Any
    process_vision_info: Any
    torch: Any
    device_map_report: DeviceMapReport
    input_device: str
    allowed_dimensions: tuple[int, ...]

    def _encode_conversations(
        self,
        conversations: list[list[dict[str, Any]]],
        dimension: int | None = None,
    ):
        if not conversations:
            raise ValueError("Embedding batch must not be empty")
        ensure_right_padding(self.processor)
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in conversations
        ]
        images, videos, video_kwargs = self.process_vision_info(
            [list(conversation) for conversation in conversations],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.input_device)
        else:
            inputs = {
                key: value.to(self.input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with self.torch.inference_mode():
            embedding = self.model.embedding(**inputs)
        if dimension is not None:
            embedding = truncate_mrl(
                embedding,
                int(dimension),
                self.allowed_dimensions,
            )
        return embedding

    def embed_texts(self, texts: list[str], dimension: int | None = None):
        return self._encode_conversations(
            [build_messages(text=text) for text in texts],
            dimension,
        )

    def embed_images(self, images: list[Any], dimension: int | None = None):
        return self._encode_conversations(
            [build_messages(image=image) for image in images],
            dimension,
        )

    def embed_image_texts(
        self,
        items: list[tuple[Any, str]],
        dimension: int | None = None,
    ):
        return self._encode_conversations(
            [build_messages(image=image, text=text) for image, text in items],
            dimension,
        )

    def embed_text(self, text: str, dimension: int | None = None):
        return self.embed_texts([text], dimension)

    def embed_image(self, image: Any, dimension: int | None = None):
        return self.embed_images([image], dimension)


def load_runtime(config: RuntimeConfig) -> WeMMRuntime:
    model_dir = config.normalized_model_path()
    if config.offline:
        enforce_offline()
    identity = validate_model_dir(model_dir)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModel, AutoProcessor

    dtype = getattr(torch, config.dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    emit_padding_evidence(processor)
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
        "dtype": dtype,
        "device_map": config.device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": config.attn_implementation,
        "use_safetensors": True,
    }
    max_memory = config.max_memory()
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    model = AutoModel.from_pretrained(str(model_dir), **kwargs).eval()
    mapping = dict(getattr(model, "hf_device_map", {}) or {})
    report = validate_device_map(
        mapping,
        required_gpu_ids=config.gpu_ids,
        allow_cpu_offload=config.allow_cpu_offload,
    )
    install_embedding_dispatch_compat(model, use_it=len(report.gpu_ids) > 1)
    input_device = _preferred_input_device(mapping)
    allowed = tuple(
        int(x)
        for x in getattr(
            model.config,
            "matryoshka_dimensions",
            identity.matryoshka_dimensions,
        )
    )
    missing = [
        dim for dim in config.allowed_dimensions if int(dim) not in allowed
    ]
    if missing:
        raise RuntimeError(
            f"Loaded model does not advertise required dimensions {missing}; allowed={allowed}"
        )
    return WeMMRuntime(
        config=config,
        identity=identity,
        processor=processor,
        model=model,
        process_vision_info=process_vision_info,
        torch=torch,
        device_map_report=report,
        input_device=input_device,
        allowed_dimensions=allowed,
    )
