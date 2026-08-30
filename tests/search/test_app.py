from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from wemm_kaggle.search.app import build_app
from wemm_kaggle.search.mrl import FULL_DIMENSION
from wemm_kaggle.search.qdrant_store import COLLECTION_4096, COLLECTION_1024, QdrantStore
from wemm_kaggle.search.schemas import SearchRequest, SearchResponse, SearchHit, SearchTimingMs
from wemm_kaggle.search.service import SearchDependencyError, SearchService


def _unit(n: int) -> list[float]:
    return [1.0 / math.sqrt(n)] * n


class FakeEmbedding:
    def embed_texts_4096(self, texts):
        return [_unit(FULL_DIMENSION) for _ in texts]

    def ready(self):
        return True


class FakePoint:
    def __init__(self, qid, score):
        self.score = score
        self.payload = {"qid": qid, "text_en": "x", "text_vi": "y"}


class FakeResp:
    def __init__(self):
        self.points = [FakePoint("Q42", 0.9)]


class FakeQdrantStore(QdrantStore):
    def __init__(self, ready=True):
        super().__init__(client=_FakeQueryClient())
        self._ready = ready

    def verify_schema(self, dim, allowed_range):
        if not self._ready:
            raise Exception("down")
        return {"dimension": dim}


class _FakeQueryClient:
    def __init__(self):
        self.last = {}

    def query_points(self, **kw):
        self.last.update(kw)
        return FakeResp()


def _app(ready=True):
    svc = SearchService(embedding=FakeEmbedding(), qdrant=FakeQdrantStore(ready=ready))
    return build_app(svc)


def test_healthz():
    client = TestClient(_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ok():
    client = TestClient(_app(ready=True))
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_readyz_down_503():
    client = TestClient(_app(ready=False))
    r = client.get("/readyz")
    assert r.status_code == 503


def test_search_endpoint():
    client = TestClient(_app())
    r = client.post(
        "/v1/search",
        json={"query": "hi", "query_language": "en", "target_language": "vi", "dimension": 4096, "limit": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["vector_name"] == "vi"
    assert body["collection"] == COLLECTION_4096
    assert body["hits"][0]["qid"] == "Q42"


def test_search_dimension_1024():
    client = TestClient(_app())
    r = client.post(
        "/v1/search",
        json={"query": "hi", "target_language": "vi", "dimension": 1024, "limit": 10},
    )
    assert r.status_code == 200
    assert r.json()["collection"] == COLLECTION_1024


def test_search_invalid_query():
    client = TestClient(_app())
    r = client.post("/v1/search", json={"query": "  ", "dimension": 4096})
    assert r.status_code == 422
