import torch
import pytest


def _naive_upstream_pool(hidden, mask):
    """Mirror of modeling_wemm_embedding.WeMMEmbedding.embedding pooling.

    The Kaggle Input mirror computes eos_positions from attention_mask and indexes
    last_hidden_state directly. Under a balanced dual-GPU device map the mask keeps
    the input device while last_hidden_state comes out of the top layer on the other
    GPU, so torch raises a cross-device indexing error.
    """
    import torch.nn.functional as F

    if mask is not None:
        positions = mask.sum(dim=1) - 1
    else:
        positions = torch.full((hidden.size(0),), hidden.size(1) - 1, device=hidden.device)
    positions = positions.clamp(min=0)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return F.normalize(hidden[batch, positions], dim=-1)


NEEDS_TWO_GPUS = "requires two CUDA GPUs to reproduce the T4 x2 dispatch split"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason=NEEDS_TWO_GPUS)
def test_upstream_cross_device_pooling_fails_without_compat():
    hidden = torch.randn(3, 8, 4, device="cuda:1")
    mask = torch.ones(3, 8, dtype=torch.long, device="cuda:0")
    with pytest.raises(RuntimeError, match="same device"):
        _naive_upstream_pool(hidden, mask)
