from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from wemm_runtime.config import RuntimeConfig
from wemm_runtime.worker import SubprocessEmbeddingWorker, WorkerTimeoutError

MODEL = Path("/tmp/wemm-model")

CHILD_SCRIPT = """
import json
import os
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
    sys.stdout.flush()


sys.stdout.write("smoke-child: initializing, please wait\\n")
sys.stdout.flush()
emit({
    "ok": True,
    "type": "ready",
    "pid": os.getpid(),
    "input_device": "cpu",
    "gpu_ids": [0],
    "module_counts": {"0": 1},
    "offload_targets": [],
    "allowed_dimensions": [4096, 1024],
    "model_identity": {"name": "smoke-child", "revision": "smoke"},
    "dtype": "float16",
})
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if request.get("op") == "shutdown":
        emit({"ok": True, "type": "shutdown"})
        sys.exit(0)
    sys.stdout.write("smoke-child: computing embedding\\n")
    sys.stdout.flush()
    emit({
        "ok": True,
        "type": "embedding",
        "dimension": int(request.get("dimension", 4096)),
        "vector": [0.25, 0.5, 0.75],
    })
sys.exit(0)
"""

SILENT_CHILD_SCRIPT = """
import sys
import time

sys.stdout.write("smoke-child: silently waiting\\n")
sys.stdout.flush()
while True:
    time.sleep(3600)
"""

REAL_POPEN = subprocess.Popen


def _build_bootstrap(child_script: str) -> str:
    return f"exec({child_script!r})"


@pytest.fixture
def install_test_child(monkeypatch):
    """Route real subprocess.Popen to a tiny test child over actual OS pipes."""

    def _install(child_script: str) -> str:
        bootstrap = _build_bootstrap(child_script)

        def _popen(*args, **kwargs):
            argv = list(args[0]) if args else list(kwargs["args"])
            for idx, part in enumerate(argv):
                if part == "-c":
                    argv[idx + 1] = bootstrap
                    break
            if args:
                return REAL_POPEN(argv, *args[1:], **kwargs)
            kwargs["args"] = argv
            return REAL_POPEN(**kwargs)

        monkeypatch.setattr("wemm_runtime.worker.subprocess.Popen", _popen)
        return bootstrap

    return _install


def _make_worker(*, fast: bool) -> SubprocessEmbeddingWorker:
    return SubprocessEmbeddingWorker(
        RuntimeConfig(model_path=MODEL, local_files_only=True, offline=True),
        startup_timeout_s=0.1 if fast else 15.0,
        request_timeout_s=0.1 if fast else 15.0,
        shutdown_timeout_s=0.1 if fast else 5.0,
    )


def test_worker_protocol_over_real_subprocess_pipes(install_test_child):
    install_test_child(CHILD_SCRIPT)
    child_pids: list[int] = []
    worker = _make_worker(fast=False)
    try:
        info = worker.start(on_spawn=child_pids.append)
        assert child_pids and info.pid == child_pids[0]
        assert info.model_identity["name"] == "smoke-child"
        assert info.allowed_dimensions == (4096, 1024)

        vector = worker.embed_text("xin chao", dimension=1024)
        assert vector == [0.25, 0.5, 0.75]

        log = "".join(worker._log_tail)
        assert "initializing, please wait" in log
        assert "computing embedding" in log

        reader = worker._reader_thread
        rc = worker.shutdown(timeout_s=5.0)
        assert rc == 0
        assert worker.process is None
        assert worker._reader_thread is None
        if reader is not None:
            assert not reader.is_alive()
    finally:
        if worker.process is not None:
            worker.terminate()


def test_real_subprocess_reads_shutdown_ack_before_exit(install_test_child):
    install_test_child(CHILD_SCRIPT)
    worker = _make_worker(fast=False)
    try:
        worker.start()
        assert worker.embed_text("hello") == [0.25, 0.5, 0.75]
        assert worker.shutdown(timeout_s=5.0) == 0
    finally:
        if worker.process is not None:
            worker.terminate()


def test_real_silent_child_timeout_is_bounded(install_test_child):
    install_test_child(SILENT_CHILD_SCRIPT)
    worker = _make_worker(fast=True)
    child_pids: list[int] = []
    started = time.monotonic()
    with pytest.raises(WorkerTimeoutError):
        worker.start(on_spawn=child_pids.append)
    elapsed = time.monotonic() - started
    assert child_pids
    assert worker.process is None
    assert worker.info is None
    assert elapsed < 30.0