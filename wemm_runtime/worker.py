from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RuntimeConfig


class WorkerTimeoutError(RuntimeError):
    """The worker did not produce a protocol response within the allowed bound."""


@dataclass(frozen=True)
class WorkerInfo:
    pid: int
    input_device: str
    gpu_ids: tuple[int, ...]
    module_counts: dict[int, int]
    offload_targets: tuple[str, ...]
    allowed_dimensions: tuple[int, ...]
    model_identity: dict[str, Any]
    dtype: str


class SubprocessEmbeddingWorker:
    """GPU-isolated JSON-line worker.

    The controller itself never imports torch. This is intentionally friendly to
    notebooks: model CUDA state lives only in the child process, so a clean
    shutdown returns VRAM to the system without relying on notebook GC.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        startup_timeout_s: float = 600.0,
        request_timeout_s: float = 120.0,
        shutdown_timeout_s: float = 15.0,
    ):
        self.config = config
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.process: subprocess.Popen[str] | None = None
        self.info: WorkerInfo | None = None
        self._config_path: Path | None = None
        self._log_tail: deque[str] = deque(maxlen=200)
        self._reader_thread: threading.Thread | None = None
        self._reader_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    def _reader_main(self) -> None:
        """Continuously drain merged stdout/stderr onto an internal queue.

        Queue items are ``("line", text)`` for each output line, ``("eof",
        returncode)`` when the child closes its streams, or ``("error", exc)``
        on an unexpected read failure.
        """
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline()
                if not line:
                    self._reader_queue.put(("eof", self.process.poll()))
                    return
                self._reader_queue.put(("line", line))
        except Exception as exc:
            self._reader_queue.put(("error", exc))

    def _stop_reader(self) -> None:
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _read_message(self, timeout_s: float) -> dict[str, Any]:
        if self.process is None:
            raise RuntimeError("Worker is not started")
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.terminate()
                raise WorkerTimeoutError(
                    f"WeMM worker did not respond within {timeout_s:g}s; "
                    f"process terminated"
                )
            try:
                item = self._reader_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            kind, payload = item
            if kind == "error":
                raise RuntimeError(f"WeMM worker read error: {payload}")
            if kind == "eof":
                log_tail = "".join(self._log_tail)[-8000:]
                raise RuntimeError(
                    f"WeMM worker exited before producing a response; "
                    f"returncode={payload} log_tail={log_tail}"
                )
            line = str(payload)
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._log_tail.append(line)
                continue
            if isinstance(message, dict) and "ok" in message:
                break
        if not message.get("ok"):
            raise RuntimeError(
                f"WeMM worker failed: {message.get('exception_type')}: "
                f"{message.get('message')}\n{message.get('traceback', '')}"
            )
        return message

    def start(self, on_spawn: Callable[[int], None] | None = None) -> WorkerInfo:
        if self.process is not None:
            raise RuntimeError("Worker already started")

        temp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="wemm-runtime-",
            delete=False,
        )
        payload = {
            **asdict(self.config),
            "model_path": str(self.config.normalized_model_path()),
            "gpu_ids": list(self.config.gpu_ids),
            "allowed_dimensions": list(self.config.allowed_dimensions),
        }
        json.dump(payload, temp)
        temp.write("\n")
        temp.close()
        self._config_path = Path(temp.name)

        repo_root = Path(__file__).resolve().parents[1]
        child_env = dict(os.environ)
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            str(repo_root)
            if not existing_pythonpath
            else str(repo_root) + os.pathsep + existing_pythonpath
        )

        bootstrap = (
            "import sys; "
            f"sys.path.insert(0, {str(repo_root)!r}); "
            "from wemm_runtime.worker_process import main; "
            "raise SystemExit(main())"
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                bootstrap,
                "--config",
                str(self._config_path),
            ],
            cwd=str(repo_root),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if on_spawn is not None:
            on_spawn(int(self.process.pid))
        self._reader_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._reader_main, name="wemm-worker-reader", daemon=True
        )
        self._reader_thread.start()
        try:
            message = self._read_message(self.startup_timeout_s)
            if message.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker startup response: {message}")
        except BaseException:
            self.terminate()
            raise
        self.info = WorkerInfo(
            pid=int(message["pid"]),
            input_device=str(message["input_device"]),
            gpu_ids=tuple(int(x) for x in message["gpu_ids"]),
            module_counts={
                int(k): int(v) for k, v in message["module_counts"].items()
            },
            offload_targets=tuple(str(x) for x in message["offload_targets"]),
            allowed_dimensions=tuple(
                int(x) for x in message["allowed_dimensions"]
            ),
            model_identity=dict(message["model_identity"]),
            dtype=str(message["dtype"]),
        )
        return self.info

    def _request(self, payload: dict[str, Any]) -> list[float]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Worker is not started")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"WeMM worker is not alive; returncode={self.process.returncode}"
            )
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        message = self._read_message(self.request_timeout_s)
        if message.get("type") != "embedding":
            raise RuntimeError(f"Unexpected worker response: {message}")
        vector = message.get("vector")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"Invalid worker vector response: {message}")
        return [float(x) for x in vector]

    def embed_text(self, text: str, dimension: int = 4096) -> list[float]:
        return self._request(
            {"op": "embed_text", "text": str(text), "dimension": int(dimension)}
        )

    def embed_image(self, image_path: str | Path, dimension: int = 4096) -> list[float]:
        return self._request(
            {
                "op": "embed_image",
                "image_path": str(Path(image_path)),
                "dimension": int(dimension),
            }
        )

    def shutdown(self, timeout_s: float = 30.0) -> int:
        process = self.process
        if process is None:
            return 0
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    process.stdin.flush()
                except OSError:
                    pass
                try:
                    self._read_message(self.shutdown_timeout_s)
                except Exception:
                    pass
            process.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            self._stop_reader()
            if self._config_path is not None:
                self._config_path.unlink(missing_ok=True)
            self._config_path = None
            self.process = None
        return int(process.returncode or 0)

    def terminate(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._stop_reader()
        if self._config_path is not None:
            self._config_path.unlink(missing_ok=True)
        self._config_path = None
        self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


EmbeddingWorker = SubprocessEmbeddingWorker
