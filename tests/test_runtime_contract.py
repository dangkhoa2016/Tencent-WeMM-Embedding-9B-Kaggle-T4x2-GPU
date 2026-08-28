import json
from pathlib import Path
from types import SimpleNamespace


def make_model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({
        "matryoshka_dimensions": [64, 128, 256, 512, 1024, 2048, 4096],
        "auto_map": {"AutoModel": "modeling_wemm_embedding.WeMMEmbedding"},
    }))
    for name in ("modeling_wemm_embedding.py", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}" if name.endswith(".json") else "# code\n")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


class Recorder:
    processor_calls = []
    model_calls = []


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        Recorder.processor_calls.append((path, kwargs))
        return cls()


class FakeModelInstance:
    hf_device_map = {"model.embed_tokens": 0, "model.layers.0": 1}
    config = SimpleNamespace(matryoshka_dimensions=[64, 128, 256, 512, 1024, 2048, 4096])

    def eval(self):
        return self


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        Recorder.model_calls.append((path, kwargs))
        return FakeModelInstance()


def test_loader_uses_local_only_fp16_balanced_dispatch(tmp_path, monkeypatch):
    import torch
    from wemm_kaggle import offline, runtime

    root = tmp_path / "kaggle" / "input"
    model_dir = make_model(root / "wemm-embedding-9b" / "1")
    monkeypatch.setattr(offline, "KAGGLE_INPUT_ROOT", root)
    Recorder.processor_calls.clear()
    Recorder.model_calls.clear()
    monkeypatch.setattr(runtime, "_imports", lambda: (torch, FakeAutoModel, FakeProcessor, lambda *a, **k: (None, None, {})))

    loaded = runtime.load_local_runtime(model_dir, per_gpu_mib=14000)

    p_path, p_kwargs = Recorder.processor_calls[-1]
    m_path, m_kwargs = Recorder.model_calls[-1]
    assert Path(p_path) == model_dir.resolve()
    assert Path(m_path) == model_dir.resolve()
    assert p_kwargs["local_files_only"] is True
    assert p_kwargs["trust_remote_code"] is True
    assert m_kwargs["local_files_only"] is True
    assert m_kwargs["trust_remote_code"] is True
    assert m_kwargs["dtype"] is torch.float16
    assert m_kwargs["device_map"] == "balanced"
    assert m_kwargs["attn_implementation"] == "sdpa"
    assert m_kwargs["max_memory"] == {0: "14000MiB", 1: "14000MiB", "cpu": "2GiB"}
    assert loaded.input_device == "cuda:0"
    assert loaded.allowed_dimensions[-1] == 4096


def test_runtime_source_has_no_model_download_fallbacks():
    from pathlib import Path
    import wemm_kaggle.runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    forbidden = ["snapshot_download(", "hf_hub_download(", "kagglehub.model_download(", 'tencent/WeMM-Embedding-9B']
    assert not [item for item in forbidden if item in source]
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

