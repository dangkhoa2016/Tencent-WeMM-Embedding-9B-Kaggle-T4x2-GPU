import torch
import pytest


def test_build_messages_places_image_before_text():
    from wemm_kaggle.embedding import build_messages

    image = object()
    messages = build_messages(text="describe", image=image)
    content = messages[0]["content"]
    assert content[0] == {"type": "image", "image": image}
    assert content[1] == {"type": "text", "text": "describe"}


def test_build_messages_rejects_empty_input():
    from wemm_kaggle.embedding import build_messages

    with pytest.raises(ValueError, match="text or image"):
        build_messages()


def test_truncate_mrl_slices_and_renormalizes():
    from wemm_kaggle.embedding import truncate_mrl

    full = torch.zeros((1, 4096), dtype=torch.float32)
    full[0, :1024] = 2.0
    out = truncate_mrl(full, 1024, (64, 128, 256, 512, 1024, 2048, 4096))
    assert out.shape == (1, 1024)
    assert torch.allclose(torch.linalg.vector_norm(out, dim=-1), torch.ones(1), atol=1e-6)


def test_truncate_mrl_rejects_unadvertised_dimension():
    from wemm_kaggle.embedding import truncate_mrl

    with pytest.raises(ValueError, match="not supported"):
        truncate_mrl(torch.ones((1, 4096)), 768, (64, 128, 256, 512, 1024, 2048, 4096))


def test_check_embedding_rejects_non_finite_and_wrong_dimension():
    from wemm_kaggle.embedding import check_embedding

    with pytest.raises(RuntimeError, match="shape"):
        check_embedding(torch.ones((1, 100)), 4096)
    bad = torch.ones((1, 4096))
    bad[0, 10] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        check_embedding(bad, 4096)


def test_check_embedding_reports_norm():
    from wemm_kaggle.embedding import check_embedding

    vector = torch.nn.functional.normalize(torch.arange(1, 4097, dtype=torch.float32)[None, :], dim=-1)
    report = check_embedding(vector, 4096)
    assert report["shape"] == [1, 4096]
    assert abs(report["norm_min"] - 1.0) < 1e-5
    assert abs(report["norm_max"] - 1.0) < 1e-5
