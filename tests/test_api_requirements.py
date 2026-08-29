from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_api_requirements_are_pinned_without_torch_cuda_specs():
    text = read("requirements-api.txt").lower()
    assert "fastapi==0.141.1" in text
    assert "uvicorn==0.52.4" in text
    assert "python-multipart==0.0.32" in text
    assert "httpx==0.28.1" in text
    forbidden = ("torch", "triton", "nvidia-", "cuda-")
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        assert not clean.startswith(forbidden)


def test_api_setup_reuses_existing_venv_and_runs_torch_guards():
    text = read("kaggle/setup-api.sh")
    assert "requirements-api.txt" in text
    assert "setup_guard.py" in text
    assert "verify-reuse" in text
    assert "scan-local" in text
    assert "rm -rf" not in text
    assert "python -m venv" not in text