import json
from pathlib import Path

import pytest


def make_model(path: Path, *, dims=(64, 128, 256, 512, 1024, 2048, 4096)) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"matryoshka_dimensions": list(dims), "auto_map": {"AutoModel": "modeling_wemm_embedding.WeMMEmbedding"}}))
    for name in ("modeling_wemm_embedding.py", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}" if name.endswith(".json") else "# custom model\n")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_validate_model_dir_requires_4096_dimension(tmp_path):
    from wemm_kaggle.model_resolver import validate_model_dir

    model = make_model(tmp_path / "wemm", dims=(64, 128, 1024))
    with pytest.raises(ValueError, match="4096"):
        validate_model_dir(model)


def test_resolve_model_dir_returns_unique_complete_candidate(tmp_path, monkeypatch):
    from wemm_kaggle import offline
    from wemm_kaggle.model_resolver import resolve_model_dir

    root = tmp_path / "kaggle" / "input"
    expected = make_model(root / "models" / "publisher" / "wemm-embedding-9b" / "1")
    monkeypatch.setattr(offline, "KAGGLE_INPUT_ROOT", root)
    assert resolve_model_dir(root, None) == expected.resolve()


def test_resolve_model_dir_fails_when_ambiguous(tmp_path, monkeypatch):
    from wemm_kaggle import offline
    from wemm_kaggle.model_resolver import resolve_model_dir

    root = tmp_path / "kaggle" / "input"
    make_model(root / "a" / "wemm-embedding-9b")
    make_model(root / "b" / "wemm-embedding-9b")
    monkeypatch.setattr(offline, "KAGGLE_INPUT_ROOT", root)
    with pytest.raises(RuntimeError, match="Multiple"):
        resolve_model_dir(root, None)


def test_explicit_model_dir_must_be_under_kaggle_input(tmp_path, monkeypatch):
    from wemm_kaggle import offline
    from wemm_kaggle.model_resolver import resolve_model_dir

    root = tmp_path / "kaggle" / "input"
    root.mkdir(parents=True)
    outside = make_model(tmp_path / "outside")
    monkeypatch.setattr(offline, "KAGGLE_INPUT_ROOT", root)
    with pytest.raises(ValueError, match="/kaggle/input"):
        resolve_model_dir(root, str(outside))
