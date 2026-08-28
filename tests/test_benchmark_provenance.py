from pathlib import Path

import json
import pytest
import subprocess
import sys

from scripts.benchmark import assert_clean_source, source_identity


@pytest.fixture()
def _restore_offline_env():
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM")
    saved = {key: __import__("os").environ.get(key) for key in keys}
    yield
    import os
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return proc.stdout.strip()


def _init_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Provenance Test")
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def test_clean_committed_repo_reports_clean(tmp_path):
    repo = _init_repo(tmp_path, {"config.json": "{}"})
    info = source_identity(root=repo)
    assert info["git_available"] is True
    assert info["git_dirty"] is False
    assert info["git_status_porcelain"] == []
    assert info["git_branch"]
    assert len(info["git_commit"]) == 40
    assert info["git_tree"]


def test_modified_tracked_file_marks_dirty(tmp_path):
    repo = _init_repo(tmp_path, {"config.json": "{}"})
    (repo / "config.json").write_text('{"x": 1}', encoding="utf-8")
    info = source_identity(root=repo)
    assert info["git_available"] is True
    assert info["git_dirty"] is True
    assert any("config.json" in line for line in info["git_status_porcelain"])


def test_untracked_non_ignored_file_marks_dirty(tmp_path):
    repo = _init_repo(tmp_path, {"config.json": "{}"})
    (repo / "new_file.txt").write_text("x", encoding="utf-8")
    info = source_identity(root=repo)
    assert info["git_available"] is True
    assert info["git_dirty"] is True
    assert any(line.startswith("?? ") for line in info["git_status_porcelain"])


def test_missing_git_reports_unavailable(tmp_path):
    root = tmp_path / "no-git"
    root.mkdir()
    info = source_identity(root=root)
    assert info["git_available"] is False
    assert info["git_commit"] is None
    assert info["git_dirty"] is None
    assert info["git_status_porcelain"] == []


def test_assert_clean_source_allows_clean_and_non_git_sources():
    assert_clean_source({"git_available": True, "git_dirty": False})
    assert_clean_source({"git_available": False, "git_dirty": None})


def test_assert_clean_source_rejects_dirty_git_tree():
    with pytest.raises(RuntimeError, match="dirty"):
        assert_clean_source({"git_available": True, "git_dirty": True})


def test_benchmark_main_fails_closed_on_dirty_source_before_model_load(monkeypatch, tmp_path, _restore_offline_env):
    import scripts.benchmark as mod

    monkeypatch.setenv("WEMM_BENCHMARK_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(
        mod,
        "source_identity",
        lambda: {
            "version": "0.1.3-rc2",
            "git_available": True,
            "git_dirty": True,
            "git_status_porcelain": [" M wemm_kaggle/runtime.py"],
        },
    )

    def fake_cuda_preflight(*args, **kwargs):
        raise AssertionError("benchmark must stop before CUDA preflight on a dirty tree")

    monkeypatch.setattr(mod, "cuda_preflight", fake_cuda_preflight)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--warmup-iterations", "1", "--measured-iterations", "1"])

    rc = mod.main()
    assert rc == 1

    run_root = tmp_path / "runs"
    evidence_files = list(run_root.glob("*/benchmark.json"))
    assert evidence_files, "expected a benchmark.json written on FAIL"
    evidence = json.loads(evidence_files[0].read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL"
    assert "dirty" in evidence["error"].lower()