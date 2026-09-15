"""Bootstrap the frozen Kaggle runtime without moving presentation code into it."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from IPython.display import Markdown, display


REPOSITORY_URL = "https://github.com/dangkhoa2016/Tencent-WeMM-Embedding-9B-Kaggle-T4x2-GPU.git"
RUNTIME_COMMIT = "d04bcd3e601b449b67d09ff1132cab965619d858"
RUNTIME_ROOT = Path("/kaggle/working") / f"wemm-production-runtime-{RUNTIME_COMMIT[:12]}"


def _drop_frozen_runtime_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "wemm_runtime"
            or name.startswith("wemm_runtime.")
            or name == "wemm_kaggle"
            or name.startswith("wemm_kaggle.")
        ):
            del sys.modules[name]


def bootstrap_runtime() -> Path:
    """Fetch, verify and import the frozen runtime authority."""

    display(
        Markdown(
            "## Bootstrap runtime dùng lại / Bootstrap the reusable runtime\n\n"
            "**VI:** Checkout đúng Git commit, cài dependency trong chế độ output sạch, "
            "rồi chứng minh Python import đúng module từ checkout vừa tạo. Presentation/"
            "orchestration code vẫn thuộc public release hiện tại; frozen runtime chỉ cung "
            "cấp science/runtime authority.\n\n"
            "**EN:** Checkout the exact runtime commit, install dependencies with clean "
            "output, and prove imports resolve to that checkout. Presentation/orchestration "
            "code remains owned by the current public release; the frozen checkout supplies "
            "only the science/runtime authority."
        )
    )

    _drop_frozen_runtime_modules()
    sys.path[:] = [
        entry for entry in sys.path if "wemm-production-runtime" not in str(entry)
    ]
    importlib.invalidate_caches()

    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)

    subprocess.run(["git", "init", "-q"], cwd=RUNTIME_ROOT, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", REPOSITORY_URL],
        cwd=RUNTIME_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "fetch", "-q", "--depth", "1", "origin", RUNTIME_COMMIT],
        cwd=RUNTIME_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", "FETCH_HEAD"],
        cwd=RUNTIME_ROOT,
        check=True,
    )

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=RUNTIME_ROOT,
        text=True,
    ).strip()
    if head != RUNTIME_COMMIT:
        raise RuntimeError(
            f"Frozen runtime checkout mismatch: observed={head} expected={RUNTIME_COMMIT}"
        )

    pip_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "-r",
            str(RUNTIME_ROOT / "requirements-kaggle.txt"),
            "-r",
            str(RUNTIME_ROOT / "requirements-demo.txt"),
        ],
        text=True,
        capture_output=True,
    )
    if pip_result.returncode != 0:
        if pip_result.stdout:
            print(pip_result.stdout, flush=True)
        if pip_result.stderr:
            print(pip_result.stderr, file=sys.stderr, flush=True)
        raise RuntimeError("Runtime dependency installation failed")

    os.environ["WEMM_RUNTIME_SOURCE_COMMIT"] = RUNTIME_COMMIT
    sys.path.insert(0, str(RUNTIME_ROOT))
    importlib.invalidate_caches()

    worker_module = importlib.import_module("wemm_runtime.worker")
    worker_file = Path(worker_module.__file__).resolve()
    if RUNTIME_ROOT.resolve() not in worker_file.parents:
        raise RuntimeError(
            f"Runtime import authority mismatch: {worker_file} is not under {RUNTIME_ROOT}"
        )

    notebook_module = sys.modules.get("wemm_notebook")
    if notebook_module is None or not getattr(notebook_module, "__file__", None):
        raise RuntimeError("Presentation package authority is unavailable")
    notebook_file = Path(notebook_module.__file__).resolve()
    if RUNTIME_ROOT.resolve() in notebook_file.parents:
        raise RuntimeError(
            "Presentation package resolved from frozen runtime; source authority is mixed"
        )

    print("Phụ thuộc runtime / Runtime dependencies : PASS")
    print("Nguồn đã pin / Pinned source             : PASS")
    print("Commit runtime / Runtime commit           : " + RUNTIME_COMMIT)
    print("Nguồn module thực thi / Import authority  : " + str(worker_file))
    print("Nguồn presentation / Presentation source  : " + str(notebook_file))
    print("PUBLIC_DEMO_SOURCE_BOOTSTRAP=PASS")
    print("RUNTIME_SOURCE_COMMIT=" + RUNTIME_COMMIT)
    print("RUNTIME_IMPORT_PATH=" + str(worker_file))
    print("NOTEBOOK_PRESENTATION_IMPORT_PATH=" + str(notebook_file))
    return RUNTIME_ROOT
