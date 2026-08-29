import io
import threading
import time

import torch
from fastapi.testclient import TestClient

from wemm_kaggle.api.app import create_app
from wemm_kaggle.api.config import ApiSettings
from wemm_kaggle.api.scheduler import (
    InferenceOomError,
    QueueFullError,
    RequestTimeoutError,
)

TOKEN = "t" * 48


def settings(**kw):
    base = {"api_token": TOKEN}
    base.update(kw)
    return ApiSettings(**base)


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def png_bytes(size=(64, 64)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def embed_texts(self, payload, dimension=None):
        return torch.rand(len(payload), dimension or 4096)

    def embed_images(self, payload, dimension=None):
        return torch.rand(len(payload), dimension or 4096)

    def embed_image_texts(self, payload, dimension=None):
        return torch.rand(len(payload), dimension or 4096)


def blocking_loader(release):
    def loader():
        if not release.wait(10):
            raise RuntimeError("blocking loader was never released")
        return FakeModel()

    return loader


def fast_loader():
    return lambda: FakeModel()


def wait_ready(client, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/readyz")
        if resp.status_code == 200:
            return resp
        time.sleep(0.01)
    raise AssertionError("service never became READY")


def test_healthz_returns_200_while_loader_is_blocked():
    release = threading.Event()
    app = create_app(settings(), runtime_loader=blocking_loader(release))
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["state"] in ("STARTING", "LOADING_MODEL")
        release.set()
        wait_ready(client)


def test_readyz_503_while_blocked_then_200_after_release():
    release = threading.Event()
    app = create_app(settings(), runtime_loader=blocking_loader(release))
    with TestClient(app) as client:
        nready = client.get("/readyz")
        assert nready.status_code == 503
        assert nready.json()["status"] == "not_ready"
        release.set()
        ready = wait_ready(client)
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "state": "READY"}


def test_embedding_before_ready_returns_503():
    release = threading.Event()
    app = create_app(settings(), runtime_loader=blocking_loader(release))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/embeddings/text",
            headers=auth(),
            json={"inputs": ["hello"], "dimension": 4096},
        )
        assert resp.status_code == 503
        assert resp.json()["error"] == "NOT_READY"
        release.set()
        wait_ready(client)


def test_text_endpoint_maps_b2_to_one_scheduler_job():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        resp = client.post(
            "/v1/embeddings/text",
            headers=auth(),
            json={"inputs": ["alpha", "beta"], "dimension": 1024},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["data"]) == 2
        assert all(len(item["embedding"]) == 1024 for item in body["data"])
        assert body["workload"] == "text"
        assert body["dimension"] == 1024
        assert body["model"] == "Tencent/WeMM-Embedding-9B"
        assert body["request_id"]
        assert set(body["timing_ms"]) == {"queue", "inference", "total"}
        assert resp.headers["X-Request-ID"] == body["request_id"]
        assert app.state.scheduler.inference_jobs_total == 1


def test_image_endpoint_accepts_valid_repeated_images():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        files = [
            ("images", ("a.png", png_bytes(), "image/png")),
            ("images", ("b.png", png_bytes(), "image/png")),
        ]
        resp = client.post(
            "/v1/embeddings/image",
            headers=auth(),
            files=files,
            data={"dimension": "1024"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 2
        assert resp.json()["workload"] == "image"


def test_image_text_endpoint_enforces_equal_image_text_count():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        files = [("images", (f"b{i}.png", png_bytes(), "image/png")) for i in range(2)]
        mismatched = client.post(
            "/v1/embeddings/image-text",
            headers=auth(),
            files=files,
            data={"texts": ["a", "b", "c"]},
        )
        assert mismatched.status_code == 422
        matched = client.post(
            "/v1/embeddings/image-text",
            headers=auth(),
            files=files,
            data={"texts": ["a", "b"]},
        )
        assert matched.status_code == 200
        assert matched.json()["count"] == 2
        assert matched.json()["workload"] == "image_text"


def test_missing_auth_returns_401():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        resp = client.post("/v1/embeddings/text", json={"inputs": ["a"]})
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_queue_full_maps_to_503_with_retry_after():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        scheduler = app.state.scheduler

        async def queue_full(workload, payload, dimension, *, timeout_s=None):
            raise QueueFullError("queue item budget exhausted")

        scheduler.submit = queue_full
        resp = client.post(
            "/v1/embeddings/text",
            headers=auth(),
            json={"inputs": ["a"]},
        )
        assert resp.status_code == 503
        assert resp.json()["error"] == "QUEUE_FULL"
        assert resp.headers.get("Retry-After") == "1"


def test_timeout_maps_to_504():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        scheduler = app.state.scheduler

        async def timeout(workload, payload, dimension, *, timeout_s=None):
            raise RequestTimeoutError("inference did not start")

        scheduler.submit = timeout
        resp = client.post(
            "/v1/embeddings/text",
            headers=auth(),
            json={"inputs": ["a"]},
        )
        assert resp.status_code == 504
        assert resp.json()["error"] == "REQUEST_TIMEOUT"


def test_oom_maps_to_503():
    app = create_app(settings(), runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        scheduler = app.state.scheduler

        async def oom(workload, payload, dimension, *, timeout_s=None):
            raise InferenceOomError("CUDA out of memory during inference")

        scheduler.submit = oom
        resp = client.post(
            "/v1/embeddings/text",
            headers=auth(),
            json={"inputs": ["a"]},
        )
        assert resp.status_code == 503
        assert resp.json()["error"] == "CUDA_OOM"


def test_shutdown_invokes_scheduler_shutdown():
    app = create_app(settings(), runtime_loader=fast_loader())
    scheduler = app.state.scheduler
    calls = []
    original = scheduler.shutdown

    async def spy():
        calls.append(1)
        await original()

    scheduler.shutdown = spy
    with TestClient(app) as client:
        wait_ready(client)
    assert calls == [1]

def test_configured_request_caps_are_enforced():
    app=create_app(settings(max_batch=1,max_text_chars=3),runtime_loader=fast_loader())
    with TestClient(app) as client:
        wait_ready(client)
        assert client.post("/v1/embeddings/text",headers=auth(),json={"inputs":["a","b"]}).status_code==422
        assert client.post("/v1/embeddings/text",headers=auth(),json={"inputs":["four"]}).status_code==422
        files=[("images",("a.png",png_bytes(),"image/png")),("images",("b.png",png_bytes(),"image/png"))]
        assert client.post("/v1/embeddings/image",headers=auth(),files=files).status_code==422
        one=[("images",("a.png",png_bytes(),"image/png"))]
        assert client.post("/v1/embeddings/image-text",headers=auth(),files=one,data={"texts":["four"]}).status_code==422
