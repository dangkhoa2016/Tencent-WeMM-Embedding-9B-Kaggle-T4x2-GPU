from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "kaggle/setup.sh").read_text(encoding="utf-8")
REQ = (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8")


def test_setup_avoids_ensurepip_and_requires_system_site_packages():
    assert "--without-pip --system-site-packages" in SETUP
    assert " -m ensurepip" not in SETUP
    assert "include-system-site-packages" in SETUP


def test_setup_installs_accelerate_without_dependencies():
    assert 'accelerate==1.14.0' in REQ
    assert '--no-deps "$ACCELERATE_SPEC"' in SETUP


def test_requirements_do_not_list_torch_or_cuda_stack():
    active = [line.strip().lower() for line in REQ.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line.startswith("torch") for line in active)
    assert not any(line.startswith("triton") for line in active)
    assert not any(line.startswith("nvidia-") or line.startswith("cuda-") for line in active)


def test_setup_has_before_and_after_torch_identity_gates():
    assert "venv-torch-before-install.json" in SETUP
    assert "venv-torch-after-install.json" in SETUP
    assert SETUP.count("verify-reuse") >= 2
    assert "scan-local" in SETUP


def test_setup_sources_runtime_env_before_torch_identity_capture():
    assert 'source "$ROOT/kaggle/runtime-env.sh"' in SETUP
    assert SETUP.index('source "$ROOT/kaggle/runtime-env.sh"') < SETUP.index("$GUARD\" capture")
