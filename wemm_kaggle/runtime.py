from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embedding import (
    build_messages,
    emit_padding_evidence,
    ensure_right_padding,
    pool_eos_embedding,
    truncate_mrl,
)
from .gpu import DeviceMapReport, build_max_memory, validate_device_map
from .model_resolver import ModelIdentity, validate_model_dir
from .offline import assert_kaggle_input_path, enforce_offline


def install_embedding_dispatch_compat(model: Any, use_it: bool = True) -> None:
    """Adapt WeMMEmbedding.embedding for a device map that splits the model across GPUs.

    The Kaggle Input mirror's embedding() pools by indexing last_hidden_state with
    eos_positions derived from attention_mask. When dispatch splits the model,
    last_hidden_state can be produced on a different GPU than the mask. This wrapper
    replicates the upstream pooling exactly but relies on pool_eos_embedding to place
    the position indices on the hidden-state device.
    """
    if not use_it:
        return

    def _embedding(input_ids=None, attention_mask=None, **kwargs):
        model.model.rope_deltas = None
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return pool_eos_embedding(outputs.last_hidden_state, attention_mask)

    model._wemm_dispatch_compat = True
    model.embedding = _embedding


def _imports():
    """Import heavyweight runtime dependencies only after offline mode is set."""
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModel, AutoProcessor

    return torch, AutoModel, AutoProcessor, process_vision_info


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
    model_dir: Path
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
            embedding = truncate_mrl(embedding, dimension, self.allowed_dimensions)
        return embedding

    def embed_texts(self, texts: list[str], dimension: int | None = None):
        if not texts:
            raise ValueError("Embedding batch must not be empty")
        conversations = [build_messages(text=text) for text in texts]
        return self._encode_conversations(conversations, dimension)

    def embed_images(self, images: list[Any], dimension: int | None = None):
        if not images:
            raise ValueError("Embedding batch must not be empty")
        conversations = [build_messages(image=image) for image in images]
        return self._encode_conversations(conversations, dimension)

    def embed_image_texts(self, items: list[tuple[Any, str]], dimension: int | None = None):
        if not items:
            raise ValueError("Embedding batch must not be empty")
        conversations = [build_messages(image=image, text=text) for image, text in items]
        return self._encode_conversations(conversations, dimension)

    def embed_text(self, text: str, dimension: int | None = None):
        return self.embed_texts([text], dimension)

    def embed_image(self, image: Any, dimension: int | None = None):
        return self.embed_images([image], dimension)

    def embed_image_text(self, image: Any, text: str, dimension: int | None = None):
        return self.embed_image_texts([(image, text)], dimension)


def load_local_runtime(model_dir: Path, per_gpu_mib: int = 14200) -> WeMMRuntime:
    enforce_offline()
    model_dir = assert_kaggle_input_path(Path(model_dir))
    identity = validate_model_dir(model_dir)
    torch, AutoModel, AutoProcessor, process_vision_info = _imports()

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    emit_padding_evidence(processor)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16,
        device_map="balanced",
        max_memory=build_max_memory(per_gpu_mib),
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_safetensors=True,
    ).eval()

    mapping = dict(getattr(model, "hf_device_map", {}) or {})
    report = validate_device_map(mapping)
    install_embedding_dispatch_compat(model, use_it=len(report.gpu_ids) > 1)
    input_device = _preferred_input_device(mapping)
    allowed = tuple(int(x) for x in getattr(model.config, "matryoshka_dimensions", identity.matryoshka_dimensions))
    if 4096 not in allowed:
        raise RuntimeError(f"Loaded model does not advertise 4096-dimensional embeddings: {allowed}")
    return WeMMRuntime(
        model_dir=model_dir,
        identity=identity,
        processor=processor,
        model=model,
        process_vision_info=process_vision_info,
        torch=torch,
        device_map_report=report,
        input_device=input_device,
        allowed_dimensions=allowed,
    )
