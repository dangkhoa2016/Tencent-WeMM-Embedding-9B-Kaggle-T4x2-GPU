import os
from pathlib import Path

import pytest


def test_enforce_offline_sets_required_environment(monkeypatch):
    from wemm_kaggle.offline import enforce_offline

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(key, raising=False)
    enforce_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_assert_kaggle_input_path_rejects_outside(tmp_path):
    from wemm_kaggle.offline import assert_kaggle_input_path

    with pytest.raises(ValueError, match="/kaggle/input"):
        assert_kaggle_input_path(tmp_path)


def test_assert_kaggle_input_path_accepts_descendant(monkeypatch, tmp_path):
    from wemm_kaggle import offline

    fake_root = tmp_path / "kaggle" / "input"
    model = fake_root / "models" / "wemm"
    model.mkdir(parents=True)
    monkeypatch.setattr(offline, "KAGGLE_INPUT_ROOT", fake_root)
    assert offline.assert_kaggle_input_path(model) == model.resolve()
