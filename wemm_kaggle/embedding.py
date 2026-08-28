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


def ensure_right_padding(processor) -> None:
    """Enforce the right-padding contract that EOS pooling depends on.

    ``pool_eos_embedding`` locates the final active token with
    ``attention_mask.sum(dim=1) - 1``, which is only correct when padding is
    appended to the right of each sequence. Batching must therefore never be
    configured for left padding.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    side = getattr(tokenizer, "padding_side", None)
    if side is not None and str(side).lower() != "right":
        raise RuntimeError(
            f"WeMM EOS pooling requires a right-padding tokenizer; "
            f"found padding_side={side!r}"
        )


def emit_padding_evidence(processor) -> str:
    """Emit a deterministic tokenizer padding-side evidence line at model load.

    The static guard proves the contract only on CPU/fake processors; this marker
    records what a real qualified processor exposes so the Kaggle evidence trail is
    explicit. ``UNKNOWN``/``NOT_PROVEN`` must never be treated as qualified.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    side = getattr(tokenizer, "padding_side", None)
    if side is None:
        side_value = "UNKNOWN"
        contract = "NOT_PROVEN"
    else:
        side_value = str(side)
        contract = "PASS" if side_value.lower() == "right" else "FAIL"
    line = (
        f"WEMM_TOKENIZER_PADDING_SIDE={side_value}\n"
        f"WEMM_TOKENIZER_PADDING_CONTRACT={contract}"
    )
    print(line, flush=True)
    return line


def pool_eos_embedding(last_hidden_state, attention_mask):
    """Pool last-token embeddings like the upstream WeMM embedding() method.

    The Kaggle Input mirror computes eos_positions from attention_mask and indexes
    last_hidden_state with it. Under a balanced multi-GPU device map the mask can stay
    on the input device while last_hidden_state is produced by the top layer on another
    GPU; moving the position indices onto last_hidden_state.device keeps the pooling
    valid across the device split and matches upstream semantics on a single device.
    """
    import torch
    import torch.nn.functional as F

    positions = None
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
    if dimension not in allowed_dims:
        raise ValueError(f"Matryoshka dimension {dimension} is not supported; allowed={allowed_dims}")
    if embedding.ndim != 2 or embedding.shape[-1] < dimension:
        raise ValueError(f"Cannot truncate embedding shape={tuple(embedding.shape)} to dimension={dimension}")
    return F.normalize(embedding[..., :dimension], dim=-1)


def check_embedding(embedding, expected_dim: int) -> dict[str, Any]:
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
