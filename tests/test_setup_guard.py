import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "scripts/setup_guard.py"


def load_guard():
    assert GUARD_PATH.exists(), "v0.1.2 must add scripts/setup_guard.py"
    spec = importlib.util.spec_from_file_location("wemm_setup_guard", GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(torch_file: str, *, version="2.10.0+cu128", cuda="12.8", devices=2):
    return {
        "torch_version": version,
        "torch_file": torch_file,
        "torch_cuda": cuda,
        "cuda_available": True,
        "device_count": devices,
    }


def test_verify_identity_reuse_accepts_same_system_torch_outside_venv(tmp_path: Path):
    guard = load_guard()
    system = identity("/usr/local/lib/python3.12/dist-packages/torch/__init__.py")
    actual = identity("/usr/local/lib/python3.12/dist-packages/torch/__init__.py")
    result = guard.verify_identity_reuse(system, actual, tmp_path / "venv")
    assert result["system_torch_reused"] is True
    assert result["torch_reinstall_detected"] is False


def test_verify_identity_reuse_rejects_same_version_shadowed_inside_venv(tmp_path: Path):
    guard = load_guard()
    venv = tmp_path / "venv"
    system = identity("/usr/local/lib/python3.12/dist-packages/torch/__init__.py")
    actual = identity(str(venv / "lib/python3.12/site-packages/torch/__init__.py"))
    result = guard.verify_identity_reuse(system, actual, venv)
    assert result["system_torch_reused"] is False
    assert result["torch_reinstall_detected"] is True
    assert result["torch_inside_venv"] is True


def test_verify_identity_reuse_rejects_changed_cuda_torch_identity(tmp_path: Path):
    guard = load_guard()
    system = identity("/usr/local/lib/python3.12/dist-packages/torch/__init__.py")
    actual = identity(
        "/opt/other/torch/__init__.py",
        version="2.13.0+cu130",
        cuda="13.0",
    )
    result = guard.verify_identity_reuse(system, actual, tmp_path / "venv")
    assert result["system_torch_reused"] is False
    assert "torch_version" in result["mismatches"]
    assert "torch_cuda" in result["mismatches"]
    assert "torch_file" in result["mismatches"]


def test_check_distribution_requirements_reports_satisfied_for_packaging_runtime():
    guard = load_guard()
    result = guard.check_distribution_requirements("packaging")
    assert result["requirements_satisfied"] is True
    assert result["failures"] == []
