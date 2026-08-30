from __future__ import annotations

import math

import httpx
import pytest

from wemm_kaggle.search import embedding_client as ec
from wemm_kaggle.search.embedding_client import EmbeddingClient, EmbeddingClientConfig
from wemm_kaggle.search.mrl import FULL_DIMENSION


def _unit(length: int) -> list[float]:
    return [1.0 / math.sqrt(length)] * length


_ORIGINAL_HTTPX_CLIENT = httpx.Client


def _make_client(handler):
    transport = httpx.MockTransport(handler)
    return _ORIGINAL_HTTPX_CLIENT(transport=transport)


def _patch(monkeypatch, handler):
    monkeypatch.setattr(ec.httpx, "Client", lambda **kw: _make_client(handler))
    return EmbeddingClient(EmbeddingClientConfig(api_token="tok"))


def _ok_handler(request: httpx.Request) -> httpx.Response:
    body = {"data": [{"index": 0, "embedding": _unit(FULL_DIMENSION)}]}
    return httpx.Response(200, json=body)


def test_embed_ok(monkeypatch):
    client = _patch(monkeypatch, _ok_handler)
    vecs = client.embed_texts_4096(["hi"])
    assert len(vecs) == 1
    assert len(vecs[0]) == FULL_DIMENSION


def test_embed_batch_size_constraint(monkeypatch):
    client = _patch(monkeypatch, _ok_handler)
    with pytest.raises(ValueError):
        client.embed_texts_4096([])
    with pytest.raises(ValueError):
        client.embed_texts_4096(["a", "b", "c", "d", "e"])


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _unit(FULL_DIMENSION)}]})

    client = _patch(monkeypatch, handler)
    client._config = EmbeddingClientConfig(api_token="tok", max_retries=5, initial_backoff_s=0.0)
    vecs = client.embed_texts_4096(["hi"])
    assert len(vecs) == 1
    assert calls["n"] >= 3


def test_non_retryable_400(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad"})

    client = _patch(monkeypatch, handler)
    with pytest.raises(ValueError):
        client.embed_texts_4096(["hi"])


def test_wrong_vector_length(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]})

    client = _patch(monkeypatch, handler)
    with pytest.raises(ValueError):
        client.embed_texts_4096(["hi"])


def test_nan_vector_rejected(monkeypatch):
    v = _unit(FULL_DIMENSION)
    v[0] = float("nan")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": v}]})

    client = _patch(monkeypatch, handler)
    with pytest.raises(ValueError):
        client.embed_texts_4096(["hi"])


def test_missing_dimension_field(monkeypatch):
    # response omits 'dimension' request field but returns valid-length vector
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _unit(FULL_DIMENSION)}]})

    client = _patch(monkeypatch, handler)
    vecs = client.embed_texts_4096(["hi"])
    assert len(vecs) == 1


def test_ready_probes_readyz_not_healthz(monkeypatch):
    # /healthz returns 200 while /readyz is 503 (model still loading): must be False.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(503, json={"status": "not_ready", "state": "LOADING_MODEL"})
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"status": "ok"})

    client = _patch(monkeypatch, handler)
    assert client.ready() is False


def test_ready_true_on_readyz_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/readyz"
        return httpx.Response(200, json={"status": "ready", "state": "READY"})

    client = _patch(monkeypatch, handler)
    assert client.ready() is True


def test_ready_false_on_readyz_non_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    client = _patch(monkeypatch, handler)
    assert client.ready() is False


def test_ready_false_on_connection_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _patch(monkeypatch, handler)
    assert client.ready() is False


def test_ready_false_on_malformed_body(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)  # HTTP 200 but no JSON ready contract

    client = _patch(monkeypatch, handler)
    assert client.ready() is False


def test_headers_forward_token(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _unit(FULL_DIMENSION)}]})

    client = _patch(monkeypatch, handler)
    client.embed_texts_4096(["hi"])
    assert seen["auth"] == "Bearer tok"
