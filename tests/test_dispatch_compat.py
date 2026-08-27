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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason=NEEDS_TWO_GPUS)
def test_pool_eos_embedding_cross_device_places_indices_on_hidden_device():
    from wemm_kaggle.embedding import pool_eos_embedding

    hidden = torch.randn(3, 8, 4, device="cuda:1")
    mask = torch.ones(3, 8, dtype=torch.long, device="cuda:0")
    out = pool_eos_embedding(hidden, mask)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all().item()
    norms = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(3, device="cuda:1"), atol=1e-6)


def test_pool_eos_embedding_matches_upstream_semantics():
    from wemm_kaggle.embedding import pool_eos_embedding

    hidden = torch.randn(3, 6, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    got = pool_eos_embedding(hidden, mask)
    import torch.nn.functional as F

    expected = F.normalize(hidden[torch.arange(3), mask.sum(dim=1) - 1], dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)


class _FakeEmbeddingModel:
    def embedding(self):
        return "upstream"


def test_install_embedding_dispatch_compat_applies_for_dual_gpu():
    from wemm_kaggle.runtime import install_embedding_dispatch_compat

    model = _FakeEmbeddingModel()
    assert model.embedding() == "upstream"
    install_embedding_dispatch_compat(model, use_it=True)
    assert getattr(model, "_wemm_dispatch_compat", False) is True
    assert model.embedding.__name__ == "_embedding"


def test_install_embedding_dispatch_compat_is_noop_for_single_gpu():
    from wemm_kaggle.runtime import install_embedding_dispatch_compat

    model = _FakeEmbeddingModel()
    install_embedding_dispatch_compat(model, use_it=False)
    assert getattr(model, "_wemm_dispatch_compat", False) is False
    assert model.embedding() == "upstream"