from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from ..embedding import check_embedding
from ..evidence import atomic_write_json, nvidia_smi_snapshot, process_snapshot
from ..offline import enforce_offline
from .logging import emit_event
from .state import ServicePhase, ServiceState


class ServiceNotReadyError(RuntimeError):
    error_code = "NOT_READY"


class QueueFullError(RuntimeError):
    error_code = "QUEUE_FULL"


class RequestTimeoutError(RuntimeError):
    error_code = "REQUEST_TIMEOUT"


class InferenceFailedError(RuntimeError):
    error_code = "INFERENCE_FAILED"


class InferenceOomError(RuntimeError):
    error_code = "CUDA_OOM"


@dataclass
class InferenceJob:
    job_id: str
    workload: Literal["text", "image", "image_text"]
    payload: list[Any]
    dimension: int
    item_count: int
    enqueued_at: float
    future: asyncio.Future | None
    cancelled: bool = False


@dataclass
class InferenceResult:
    embeddings: list[list[float]]
    queue_ms: float
    inference_ms: float


class InferenceScheduler:
    """Single-flight inference coordinator.

    One executor thread owns model load and every model call, so at most one
    GPU inference is ever active. HTTP handlers only enqueue jobs against an
    item-bounded budget; they never touch the model directly.
    """

    def __init__(
        self,
        runtime_loader: Callable[[], Any],
        *,
        queue_max_items: int = 16,
        request_timeout_s: float = 30.0,
        evidence_dir: Path | None = None,
    ) -> None:
        self._runtime_loader = runtime_loader
        self._queue_max_items = queue_max_items
        self._request_timeout_s = request_timeout_s
        self._evidence_dir = Path(evidence_dir) if evidence_dir else None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wemm-inference")
        self._queue: asyncio.Queue | None = None
        self._budget_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._stopping = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._load_task: asyncio.Task | None = None
        self._state = ServiceState()
        self._runtime: Any = None
        self.queued_items = 0
        self.max_observed_queue_items = 0
        self.inflight_inference = 0
        self.max_observed_inflight_inference = 0
        self.requests_total = 0
        self.requests_rejected = 0
        self.inference_jobs_total = 0
        self.oom_count = 0

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def phase(self) -> ServicePhase:
        return self._state.phase

    def note_request(self) -> None:
        self.requests_total += 1

    def note_rejected(self) -> None:
        self.requests_rejected += 1

    def metrics(self) -> dict[str, int]:
        return {
            "requests_total": self.requests_total,
            "requests_rejected": self.requests_rejected,
            "inference_jobs_total": self.inference_jobs_total,
            "oom_count": self.oom_count,
            "queued_items": self.queued_items,
            "max_observed_queue_items": self.max_observed_queue_items,
            "inflight_inference": self.inflight_inference,
            "max_observed_inflight_inference": self.max_observed_inflight_inference,
        }

    async def initialize(self) -> None:
        await self.start()

    async def start(self) -> None:
        enforce_offline()
        async with self._lifecycle_lock:
            if self._stopping.is_set():
                self._state.phase = ServicePhase.STOPPED
                return
            if self._state.phase == ServicePhase.READY:
                return
            self._queue = asyncio.Queue()
            self._state.phase = ServicePhase.LOADING_MODEL
        emit_event("model_load_start", queue_max_items=self._queue_max_items)
        load_started = time.perf_counter()
        self._load_task = asyncio.get_running_loop().create_task(self._load_runtime())
        try:
            runtime = await self._load_task
        except Exception as exc:
            async with self._lifecycle_lock:
                if self._stopping.is_set():
                    self._state.phase = ServicePhase.STOPPED
                else:
                    self._state.phase = ServicePhase.FAILED
                    self._state.error = f"{type(exc).__name__}: {exc}"
            emit_event("model_load_failed", error_type=type(exc).__name__)
            if not self._stopping.is_set():
                raise
            return
        async with self._lifecycle_lock:
            if self._stopping.is_set() or self._state.phase != ServicePhase.LOADING_MODEL:
                self._state.phase = ServicePhase.STOPPED
                return
            self._runtime = runtime
            self._state.model_load_seconds = time.perf_counter() - load_started
            emit_event("model_load_complete", seconds=self._state.model_load_seconds)
            self._worker_task = asyncio.create_task(self._worker())
            self._state.phase = ServicePhase.READY

    async def _load_runtime(self) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._runtime_loader)

    async def submit(
        self,
        workload: str,
        payload: list[Any],
        dimension: int,
        *,
        timeout_s: float | None = None,
    ) -> InferenceResult:
        if self._state.phase != ServicePhase.READY:
            self.requests_rejected += 1
            raise ServiceNotReadyError(f"service is {self._state.phase.value}")
        item_count = len(payload)
        if item_count < 1:
            self.requests_rejected += 1
            raise QueueFullError("empty payloads cannot be scheduled")
        loop = asyncio.get_running_loop()
        job = InferenceJob(
            job_id=uuid.uuid4().hex,
            workload=workload,
            payload=list(payload),
            dimension=dimension,
            item_count=item_count,
            enqueued_at=time.monotonic(),
            future=loop.create_future(),
        )
        async with self._budget_lock:
            if self._state.phase != ServicePhase.READY:
                self.requests_rejected += 1
                raise ServiceNotReadyError(f"service is {self._state.phase.value}")
            budget = self.queued_items + item_count
            if budget > self._queue_max_items:
                self.requests_rejected += 1
                raise QueueFullError(
                    f"queue item budget exhausted: {self.queued_items}+{item_count}>{self._queue_max_items}"
                )
            self.queued_items = budget
            self.max_observed_queue_items = max(self.max_observed_queue_items, budget)
        self.inference_jobs_total += 1
        emit_event("queue_enter", job_id=job.job_id, workload=workload, item_count=item_count)
        await self._queue.put(job)
        deadline = self._request_timeout_s if timeout_s is None else timeout_s
        try:
            return await asyncio.wait_for(job.future, timeout=deadline)
        except asyncio.TimeoutError:
            job.cancelled = True
            self.requests_rejected += 1
            raise RequestTimeoutError(f"inference did not start within {deadline}s") from None

    async def _worker(self) -> None:
        while True:
            if self._stopping.is_set() and self._queue.empty():
                break
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            async with self._budget_lock:
                self.queued_items = max(0, self.queued_items - job.item_count)
            if job.cancelled or job.future is None or job.future.done():
                continue
            emit_event(
                "inference_start",
                job_id=job.job_id,
                workload=job.workload,
                item_count=job.item_count,
                dimension=job.dimension,
            )
            inference_started = time.perf_counter()
            try:
                rows = await self._run_inference(job)
            except Exception as exc:
                if job.future is not None and not job.future.done():
                    job.future.set_exception(exc)
                continue
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            queue_ms = (inference_started - job.enqueued_at) * 1000.0
            emit_event(
                "inference_complete",
                job_id=job.job_id,
                workload=job.workload,
                item_count=job.item_count,
                dimension=job.dimension,
                inference_ms=round(inference_ms, 3),
                queue_ms=round(queue_ms, 3),
            )
            if job.future is None or job.future.done():
                continue
            job.future.set_result(
                InferenceResult(
                    embeddings=rows,
                    queue_ms=queue_ms,
                    inference_ms=inference_ms,
                )
            )

    async def _run_inference(self, job: InferenceJob) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._invoke_sync, job)
        except Exception as exc:
            if self._is_oom(exc):
                self.oom_count += 1
                raise InferenceOomError("CUDA out of memory during inference") from exc
            raise InferenceFailedError(f"inference failed: {type(exc).__name__}") from exc

    def _invoke_sync(self, job: InferenceJob) -> list[list[float]]:
        if self._runtime is None:
            raise RuntimeError("runtime is not loaded")
        self.inflight_inference += 1
        self.max_observed_inflight_inference = max(
            self.max_observed_inflight_inference,
            self.inflight_inference,
        )
        try:
            runtime_dimension = None if job.dimension == 4096 else job.dimension
            if job.workload == "text":
                embedding = self._runtime.embed_texts(job.payload, dimension=runtime_dimension)
            elif job.workload == "image":
                embedding = self._runtime.embed_images(job.payload, dimension=runtime_dimension)
            elif job.workload == "image_text":
                embedding = self._runtime.embed_image_texts(job.payload, dimension=runtime_dimension)
            else:
                raise RuntimeError(f"unsupported workload: {job.workload}")
            check_embedding(embedding, job.dimension)
            rows = embedding.detach().float().cpu().tolist()
            return [list(row) for row in rows]
        finally:
            self.inflight_inference -= 1

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        if type(exc).__name__ == "OutOfMemoryError":
            return True
        text = str(exc).lower()
        return "out of memory" in text or "oom" in text

    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            if self._state.phase == ServicePhase.STOPPED:
                return
            self._state.phase = ServicePhase.DRAINING
            emit_event("shutdown_start")
            self._stopping.set()
            worker_task = self._worker_task
            load_task = self._load_task
        if worker_task is not None:
            try:
                await worker_task
            except Exception:
                pass
        if load_task is not None:
            try:
                await load_task
            except Exception:
                pass
        self._executor.shutdown(wait=False)
        async with self._lifecycle_lock:
            self._state.phase = ServicePhase.STOPPED
        self._write_service_summary()
        emit_event("shutdown_complete")

    def _runtime_brief(self) -> dict[str, Any] | None:
        runtime = self._runtime
        if runtime is None:
            return None
        brief: dict[str, Any] = {
            "model_dir": str(getattr(runtime, "model_dir", "") or ""),
            "input_device": str(getattr(runtime, "input_device", "") or ""),
        }
        identity = getattr(runtime, "identity", None)
        if identity is not None and hasattr(identity, "to_dict"):
            brief["identity"] = identity.to_dict()
        elif identity is not None:
            brief["identity"] = {"matryoshka_dimensions": list(getattr(identity, "matryoshka_dimensions", ()))}
        report = getattr(runtime, "device_map_report", None)
        if report is not None and hasattr(report, "to_dict"):
            brief["device_map"] = report.to_dict()
        return brief

    def _write_service_summary(self) -> None:
        if not self._evidence_dir:
            return
        payload = {
            "schema_version": 1,
            "final_state": self._state.phase.value,
            "model": self._runtime_brief(),
            "model_load_seconds": self._state.model_load_seconds,
            "counters": self.metrics(),
            "process": process_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
        }
        atomic_write_json(Path(self._evidence_dir) / "service-summary.json", payload)