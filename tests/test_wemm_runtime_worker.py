from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from wemm_runtime.config import RuntimeConfig
from wemm_runtime.worker import (
    SubprocessEmbeddingWorker,
    WorkerTimeoutError,
)

MODEL = Path("/tmp/wemm-model")

READY = {
    "ok": True,
    "type": "ready",
    "pid": 999,
    "input_device": "cpu",
    "gpu_ids": [0, 1],
    "module_counts": {"0": 1, "1": 1},
    "offload_targets": ["cpu"],
    "allowed_dimensions": [4096, 1024],
    "model_identity": {"name": "tencent-wemm", "revision": "rc2"},
    "dtype": "float16",
}

EMBED = {"ok": True, "type": "embedding", "dimension": 4096, "vector": [0.1, 0.2, 0.3]}


class ScriptedStdout:
    """Runs a script of lines/blocks/EOF instead of a real subprocess pipe.

    EOF denotes child exit; the fake wires ``on_eof`` so ``wait()`` observes a
    clean process exit just like a real subprocess whose stdout pipe closed.
    """

    def __init__(self):
        self._script: deque[tuple[str, object]] = deque()
        self.stop = threading.Event()
        self.on_eof = None
        self._stdin: RecordingStdin | None = None
        self._seen_writes = 0

    def attach_stdin(self, stdin: RecordingStdin) -> None:
        self._stdin = stdin

    def line(self, text: str) -> None:
        self._script.append(("line", text))

    def block(self) -> None:
        self._script.append(("block", None))

    def wait_for_write(self) -> None:
        self._script.append(("wait_for_write", None))

    def eof(self) -> None:
        self._script.append(("eof", None))

    def readline(self) -> str:
        while True:
            if not self._script:
                self.stop.wait()
                if self.on_eof is not None:
                    self.on_eof()
                return ""
            kind, payload = self._script.popleft()
            if kind == "line":
                text = str(payload)
                return text if text.endswith("\n") else text + "\n"
            if kind == "block":
                self.stop.wait()
                continue
            if kind == "wait_for_write":
                while not self.stop.is_set():
                    if self._stdin is not None and len(self._stdin.writes) > self._seen_writes:
                        break
                    time.sleep(0.002)
                if self._stdin is not None:
                    self._seen_writes = len(self._stdin.writes)
                continue
            if kind == "eof":
                if self.on_eof is not None:
                    self.on_eof()
                return ""


class RecordingStdin:
    def __init__(self):
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        pass


class FakeProc:
    def __init__(self, stdout: ScriptedStdout):
        self.stdout = stdout
        self.stdin = RecordingStdin()
        self.pid = 12345
        self.returncode: int | None = None
        self.stop = threading.Event()
        self.terminated = threading.Event()
        stdout.stop = self.stop
        stdout.attach_stdin(self.stdin)
        stdout.on_eof = self._child_eof

    def _child_eof(self) -> None:
        if self.returncode is None:
            self.returncode = 0
        self.terminated.set()

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated.set()
        self.stop.set()
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.terminated.set()
        self.stop.set()
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        self.terminated.wait(timeout=timeout)
        return self.returncode


@pytest.fixture
def install_popen(monkeypatch):
    def _install(proc: FakeProc):
        class _FakePopen:
            def __init__(self, *args, **kwargs):
                self.stdout = proc.stdout
                self.stdin = proc.stdin
                self.pid = proc.pid

            @property
            def returncode(self):
                return proc.returncode

            def poll(self):
                return proc.poll()

            def wait(self, timeout=None):
                return proc.wait(timeout)

            def terminate(self):
                proc.terminate()

            def kill(self):
                proc.kill()

        monkeypatch.setattr("wemm_runtime.worker.subprocess.Popen", _FakePopen)
        return proc

    return _install


def make_worker(fast=False):
    return SubprocessEmbeddingWorker(
        RuntimeConfig(model_path=MODEL, local_files_only=True, offline=True),
        startup_timeout_s=0.1 if fast else 5.0,
        request_timeout_s=0.1 if fast else 5.0,
        shutdown_timeout_s=0.1 if fast else 5.0,
    )


def test_ready_response_succeeds(install_popen):
    proc = FakeProc(ScriptedStdout())
    out = proc.stdout
    out.line(json.dumps(READY))
    out.wait_for_write()
    out.line(json.dumps({"ok": True, "type": "shutdown"}))
    out.eof()
    install_popen(proc)
    worker = make_worker(fast=True)
    info = worker.start()
    assert info.pid == 999
    assert info.model_identity["name"] == "tencent-wemm"
    rc = worker.shutdown(timeout_s=1.0)
    assert rc == 0
    assert worker.process is None


def test_startup_timeout_terminates_child(install_popen):
    proc = FakeProc(ScriptedStdout())
    proc.stdout.line("initializing, please wait\n")
    install_popen(proc)
    worker = make_worker(fast=True)
    with pytest.raises(WorkerTimeoutError):
        worker.start()
    assert proc.terminated.is_set()
    assert worker.process is None


def test_request_timeout_is_bounded(install_popen):
    proc = FakeProc(ScriptedStdout())
    proc.stdout.line(json.dumps(READY))
    install_popen(proc)
    worker = make_worker(fast=True)
    worker.start()
    with pytest.raises(WorkerTimeoutError):
        worker.embed_text("hello")
    assert proc.terminated.is_set()
    assert worker.process is None


def test_shutdown_silent_child_is_terminated(install_popen):
    proc = FakeProc(ScriptedStdout())
    proc.stdout.line(json.dumps(READY))
    install_popen(proc)
    worker = make_worker(fast=True)
    worker.start()
    rc = worker.shutdown()
    assert rc == -15
    assert proc.terminated.is_set()
    assert worker.process is None
    shutdown_ops = [json.loads(w) for w in proc.stdin.writes if "shutdown" in w]
    assert shutdown_ops == [{"op": "shutdown"}]


def test_non_json_logs_do_not_deadlock(install_popen):
    proc = FakeProc(ScriptedStdout())
    out = proc.stdout
    out.line("loading weights...\n")
    out.line(json.dumps(READY))
    out.wait_for_write()
    out.line("progress 50%\n")
    out.line("progress 100%\n")
    out.line(json.dumps(EMBED))
    out.wait_for_write()
    out.line(json.dumps({"ok": True, "type": "shutdown"}))
    out.eof()
    install_popen(proc)
    worker = make_worker(fast=True)
    worker.start()
    vector = worker.embed_text("xin chao")
    assert vector == [0.1, 0.2, 0.3]
    log = "".join(worker._log_tail)
    assert "loading weights..." in log
    assert "progress 100%" in log
    assert worker.shutdown(timeout_s=1.0) == 0


def test_eof_before_response_reports_returncode_and_log_tail(install_popen):
    proc = FakeProc(ScriptedStdout())
    out = proc.stdout
    proc.returncode = 2
    out.line("crashed after banner\n")
    out.eof()
    install_popen(proc)
    worker = make_worker(fast=True)
    with pytest.raises(RuntimeError) as excinfo:
        worker.start()
    text = str(excinfo.value)
    assert "returncode=2" in text
    assert "crashed after banner" in text


def test_worker_failed_message_raises_runtime_error(install_popen):
    proc = FakeProc(ScriptedStdout())
    proc.stdout.line(
        json.dumps(
            {
                "ok": False,
                "type": "error",
                "exception_type": "ValueError",
                "message": "boom",
                "traceback": "tb-detail",
            }
        )
    )
    install_popen(proc)
    worker = make_worker(fast=True)
    with pytest.raises(RuntimeError) as excinfo:
        worker.start()
    assert "ValueError" in str(excinfo.value)
    assert "boom" in str(excinfo.value)