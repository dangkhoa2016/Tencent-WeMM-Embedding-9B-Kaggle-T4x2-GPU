from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

HELPER = ROOT / "kaggle/runtime-env.sh"
NVIDIA_LIB_DIR = "/usr/local/nvidia/lib64"
NVIDIA_SMI_PATH = "/opt/bin/nvidia-smi"
FORBIDDEN_STRINGS = (
    "ldconfig",
    "/etc/environment",
    ".bashrc",
    "apt install",
    "pip install torch",
)
SOURCE_LINE = 'source "$ROOT/kaggle/runtime-env.sh"'


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_runtime_env_helper_exists():
    assert HELPER.is_file(), "kaggle/runtime-env.sh is missing"


def test_runtime_env_helper_knows_nvidia_library_directory():
    text = HELPER.read_text(encoding="utf-8")
    assert NVIDIA_LIB_DIR in text


def test_runtime_env_helper_knows_nvidia_smi_path():
    text = HELPER.read_text(encoding="utf-8")
    assert "nvidia-smi" in text
    assert "NVIDIA_SMI_DIR=/opt/bin" in text


def test_runtime_env_helper_forbids_persistent_system_repair():
    text = HELPER.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_STRINGS:
        assert forbidden not in text, f"forbidden persistent repair token present: {forbidden}"


def test_setup_sources_runtime_env_before_guard_capture():
    text = read("kaggle/setup.sh")
    source_pos = text.index(SOURCE_LINE)
    capture_pos = text.index("$GUARD\" capture")
    assert source_pos < capture_pos


def test_preflight_sources_runtime_env():
    assert SOURCE_LINE in read("kaggle/run-preflight.sh")


def test_acceptance_sources_runtime_env():
    assert SOURCE_LINE in read("kaggle/run-acceptance.sh")


def test_benchmark_sources_runtime_env():
    assert SOURCE_LINE in read("kaggle/run-benchmark.sh")


def test_sanitized_runtime_env_bootstrap_is_idempotent():
    if not HELPER.is_file():
        pytest.skip("runtime-env.sh not yet present")
    clean = os.environ.copy()
    clean.pop("LD_LIBRARY_PATH", None)
    clean["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    script = (
        "source kaggle/runtime-env.sh; "
        "source kaggle/runtime-env.sh; "
        'echo "__LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(ROOT),
        env=clean,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "__LD_LIBRARY_PATH=" in proc.stdout


def test_sanitized_runtime_env_bootstrap_resolves_observed_paths():
    if not HELPER.is_file():
        pytest.skip("runtime-env.sh not yet present")
    has_lib = (Path(NVIDIA_LIB_DIR) / "libcuda.so.1").is_file()
    has_smi = Path(NVIDIA_SMI_PATH).is_file()
    if not has_lib and not has_smi:
        pytest.skip("observed NVIDIA driver paths are not present on this host")
    clean = os.environ.copy()
    clean.pop("LD_LIBRARY_PATH", None)
    clean["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    script = (
        "source kaggle/runtime-env.sh; "
        'echo "__LD=${LD_LIBRARY_PATH:-}"; '
        'echo "__SMI=$(command -v nvidia-smi || true)"'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(ROOT),
        env=clean,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    if has_lib:
        assert NVIDIA_LIB_DIR in proc.stdout
    if has_smi:
        assert "nvidia-smi" in proc.stdout