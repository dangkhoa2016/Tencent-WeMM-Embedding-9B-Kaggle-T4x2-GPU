import asyncio
import json
import time

import pytest
import torch

from wemm_kaggle.api.scheduler import (
    InferenceFailedError,
    InferenceOomError,
    InferenceResult,
    InferenceScheduler,
    QueueFullError,
    RequestTimeoutError,
    ServiceNotReadyError,
)
from wemm_kaggle.api.state import ServicePhase


class FakeRuntime:
    def __init__(self, fail=None, delay=0.0):
        self.load_count = 0
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.fail = fail
        self.delay = delay

    def load(self):
        self.load_count += 1
        return self

    def _invoke(self, workload, n, dimension):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append((workload, n, dimension))
        if self.delay:
            time.sleep(self.delay)
        if self.fail == "oom":
            self.active -= 1
            raise RuntimeError("CUDA out of memory.")
        if self.fail == "error":
            self.active -= 1
            raise RuntimeError("model exploded")
        self.active -= 1
        return torch.rand(n, dimension or 4096)

    def embed_texts(self, texts, dimension=None):
        return self._invoke("text", len(texts), dimension)

    def embed_images(self, images, dimension=None):
        return self._invoke("image", len(images), dimension)

    def embed_image_texts(self, items, dimension=None):
        return self._invoke("image_text", len(items), dimension)


def scheduler_for(fake, **kwargs):
    return InferenceScheduler(lambda: fake.load(), **kwargs)


def test_runtime_is_loaded_once():
    fake = FakeRuntime()
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        assert sched.state.phase == ServicePhase.READY
        await sched.shutdown()

    asyncio.run(main())
    assert fake.load_count == 1
    assert sched.state.phase == ServicePhase.STOPPED


def test_four_concurrent_submits_never_exceed_one_active_inference():
    fake = FakeRuntime(delay=0.05)
    sched = scheduler_for(fake)

    async def main():
        await sched.start()

        async def one(i):
            return await sched.submit("text", [f"t{i}"], 4096)

        results = await asyncio.gather(*(one(i) for i in range(4)))
        assert len(results) == 4
        await sched.shutdown()

    asyncio.run(main())
    assert fake.max_active == 1
    assert sched.max_observed_inflight_inference == 1
    assert len(fake.calls) == 4


def test_b1_b2_b4_pass_through_unchanged():
    fake = FakeRuntime()
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        await sched.submit("text", [f"t{i}" for i in range(1)], 4096)
        await sched.submit("image", [object() for _ in range(2)], 4096)
        await sched.submit("image_text", [("im", "tx") for _ in range(4)], 4096)
        await sched.shutdown()

    asyncio.run(main())
    assert [call[1] for call in fake.calls] == [1, 2, 4]
    assert [call[0] for call in fake.calls] == ["text", "image", "image_text"]


def test_4096_maps_to_runtime_none_and_1024_to_1024():
    fake = FakeRuntime()
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        await sched.submit("text", ["a"], 4096)
        await sched.submit("text", ["b"], 1024)
        await sched.shutdown()

    asyncio.run(main())
    assert fake.calls[0] == ("text", 1, None)
    assert fake.calls[1] == ("text", 1, 1024)


def test_result_shape_and_timing():
    fake = FakeRuntime()
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        result = await sched.submit("text", ["hello"], 1024)
        assert isinstance(result, InferenceResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert all(isinstance(v, float) for v in result.embeddings[0])
        assert result.queue_ms >= 0
        assert result.inference_ms >= 0
        await sched.shutdown()

    asyncio.run(main())


def test_queue_budget_refuses_overflow():
    fake = FakeRuntime()
    sched = scheduler_for(fake, queue_max_items=2)

    async def main():
        await sched.start()
        async with sched._budget_lock:
            sched.queued_items = sched._queue_max_items
        with pytest.raises(QueueFullError):
            await sched.submit("text", ["a"], 4096)
        await sched.shutdown()

    asyncio.run(main())


def test_concurrent_large_submits_trigger_backpressure():
    fake = FakeRuntime(delay=0.2)
    sched = scheduler_for(fake, queue_max_items=4)

    async def main():
        await sched.start()
        tasks = [asyncio.create_task(sched.submit("text", ["t" for _ in range(4)], 4096)) for _ in range(4)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rejected = [r for r in results if isinstance(r, QueueFullError)]
        accepted = [r for r in results if isinstance(r, InferenceResult)]
        assert rejected, "at least one submit must exceed the queue budget"
        assert accepted, "at least one submit must succeed"
        await sched.shutdown()

    asyncio.run(main())
    assert sched.max_observed_inflight_inference == 1


def test_cancelled_queued_job_is_skipped_and_timeout_before_start_raises():
    fake = FakeRuntime(delay=0.3)
    sched = scheduler_for(fake, request_timeout_s=30.0)

    async def main():
        await sched.start()
        first = asyncio.create_task(sched.submit("text", ["first"], 4096))
        await asyncio.sleep(0.05)
        with pytest.raises(RequestTimeoutError):
            await sched.submit("text", ["second"], 4096, timeout_s=0.1)
        result = await first
        assert len(result.embeddings) == 1
        await asyncio.sleep(0.05)
        await sched.shutdown()

    asyncio.run(main())
    assert fake.calls == [("text", 1, None)]


def test_model_failure_reaches_caller():
    fake = FakeRuntime(fail="error")
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        with pytest.raises(InferenceFailedError):
            await sched.submit("text", ["boom"], 4096)
        await sched.shutdown()

    asyncio.run(main())


def test_oom_like_failure_increments_oom_count():
    fake = FakeRuntime(fail="oom")
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        with pytest.raises(InferenceOomError):
            await sched.submit("text", ["oom"], 4096)
        await sched.shutdown()

    asyncio.run(main())
    assert sched.oom_count == 1
    assert sched.max_observed_inflight_inference == 1


def test_submit_before_ready_raises():
    fake = FakeRuntime()
    sched = scheduler_for(fake)

    async def main():
        with pytest.raises(ServiceNotReadyError):
            await sched.submit("text", ["a"], 4096)
        await sched.shutdown()

    asyncio.run(main())


def test_shutdown_drains_active_work_and_reaches_stopped():
    fake = FakeRuntime(delay=0.15)
    sched = scheduler_for(fake)

    async def main():
        await sched.start()
        fut = asyncio.create_task(sched.submit("text", ["drain"], 4096))
        await asyncio.sleep(0.03)
        await sched.shutdown()
        result = await fut
        assert len(result.embeddings) == 1
        assert sched.state.phase == ServicePhase.STOPPED

    asyncio.run(main())


def test_service_summary_written_on_shutdown(tmp_path):
    fake = FakeRuntime()
    sched = scheduler_for(fake, evidence_dir=tmp_path)

    async def main():
        await sched.start()
        await sched.submit("text", ["a"], 4096)
        await sched.shutdown()

    asyncio.run(main())
    payload = json.loads((tmp_path / "service-summary.json").read_text(encoding="utf-8"))
    assert payload["final_state"] == "STOPPED"
    assert payload["counters"]["max_observed_inflight_inference"] == 1
    assert payload["counters"]["oom_count"] == 0
    assert payload["model_load_seconds"] is not None
    assert "process" in payload and "nvidia_smi" in payload