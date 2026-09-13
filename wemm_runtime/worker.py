from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RuntimeConfig


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

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.info: WorkerInfo | None = None
        self._config_path: Path | None = None
        self._log_tail: deque[str] = deque(maxlen=200)

    def _read_message(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Worker is not started")
        while True:
            line = self.process.stdout.readline()
            if not line:
                log_tail = "".join(self._log_tail)[-8000:]
                raise RuntimeError(
                    f"WeMM worker exited before producing a response; "
                    f"returncode={self.process.poll()} log_tail={log_tail}"
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Third-party/custom model code may emit informational/progress text.
                # stderr is intentionally merged into stdout so the controller drains
                # both streams continuously and cannot deadlock on a full stderr pipe.
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
        try:
            message = self._read_message()
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
        message = self._read_message()
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
                process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                process.stdin.flush()
                try:
                    self._read_message()
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
