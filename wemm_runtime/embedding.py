from __future__ import annotations

from typing import Any, Iterable


def build_messages(*, text: str | None = None, image: Any | None = None) -> list[dict[str, Any]]:
    if text is None and image is None:
        raise ValueError("At least text or image must be provided")
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def pool_eos_embedding(last_hidden_state, attention_mask):
    import torch
    import torch.nn.functional as F

    if attention_mask is None:
        positions = torch.full(
            (last_hidden_state.shape[0],),
            last_hidden_state.shape[1] - 1,
            device=last_hidden_state.device,
        )
    else:
        positions = attention_mask.sum(dim=1) - 1
        positions = positions.to(last_hidden_state.device)
    positions = positions.clamp(min=0)
    batch = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return F.normalize(last_hidden_state[batch, positions], dim=-1)


def truncate_mrl(embedding, dimension: int, allowed: Iterable[int]):
    import torch.nn.functional as F

    allowed_dims = tuple(int(x) for x in allowed)
    if int(dimension) not in allowed_dims:
        raise ValueError(
            f"Matryoshka dimension {dimension} is not supported; allowed={allowed_dims}"
        )
    if embedding.ndim != 2 or embedding.shape[-1] < int(dimension):
        raise ValueError(
            f"Cannot truncate embedding shape={tuple(embedding.shape)} to dimension={dimension}"
        )
    return F.normalize(embedding[..., : int(dimension)], dim=-1)


def check_embedding(embedding, expected_dim: int) -> dict[str, object]:
    import torch

    shape = list(embedding.shape)
    if embedding.ndim != 2 or embedding.shape[0] < 1 or embedding.shape[-1] != expected_dim:
        raise RuntimeError(f"Unexpected embedding shape {shape}; expected [N, {expected_dim}]")
    if not torch.isfinite(embedding).all().item():
        raise RuntimeError("Embedding contains non-finite values")
    norms = torch.linalg.vector_norm(embedding.float(), dim=-1)
    return {
        "shape": shape,
        "dtype": str(embedding.dtype).replace("torch.", ""),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "finite": True,
    }
