from __future__ import annotations

import math

import pytest

from wemm_kaggle.search.embedding_client import EmbeddingClient
from wemm_kaggle.search.mrl import FULL_DIMENSION, DERIVED_DIMENSION
from wemm_kaggle.search.qdrant_store import (
    COLLECTION_4096,
    COLLECTION_1024,
    QdrantStore,
)
from wemm_kaggle.search.schemas import SearchRequest
from wemm_kaggle.search.service import SearchDependencyError, SearchService


def _unit(n: int) -> list[float]:
    return [1.0 / math.sqrt(n)] * n


class FakeEmbedding:
    def __init__(self):
        self.calls = 0

    def embed_texts_4096(self, texts):
        self.calls += 1
        return [_unit(FULL_DIMENSION) for _ in texts]

    def ready(self):
        return True


class FakePoint:
    def __init__(self, qid, score, text_en="", text_vi=""):
        self.score = score
        self.payload = {"qid": qid, "text_en": text_en, "text_vi": text_vi}


class FakeResp:
    def __init__(self, points):
        self.points = points


class FakeQueryClient:
    def __init__(self):
        self.last = {}

    def query_points(self, **kw):
        self.last.update(kw)
        return FakeResp([FakePoint("Q42", 0.95, "a", "b"), FakePoint("Q7", 0.80)])


class FakeQdrantStore(QdrantStore):
    def __init__(self, collection_4096=COLLECTION_4096, collection_1024=COLLECTION_1024):
        super().__init__(client=FakeQueryClient(), collection_4096=collection_4096, collection_1024=collection_1024)
        self.ready_ok = True

    def verify_schema(self, dim, allowed_range):
        if not self.ready_ok:
            raise Exception("not ready")
        return {"dimension": dim}

    @property
    def query_client(self):
        return self._client


def make_service(qdrant_store=None):
    emb = FakeEmbedding()
    qd = qdrant_store if qdrant_store is not None else FakeQdrantStore()
    svc = SearchService(embedding=emb, qdrant=qd)
    svc._emb = emb
    return svc


def test_search_en_to_vi_routes_vi_vector_4096():
    svc = make_service()
    req = SearchRequest(query="hello", query_language="en", target_language="vi", dimension=4096, limit=10)
    resp = svc.search(req, request_id="r1")
    assert resp.collection == COLLECTION_4096
    assert resp.vector_name == "vi"
    assert resp.hits[0].qid == "Q42"
    assert resp.hits[0].rank == 1
    assert resp.hits[1].rank == 2
    assert resp.timing_ms.total > 0.0


def test_search_vi_to_en_routes_en_vector():
    svc = make_service()
    req = SearchRequest(query="xin chao", query_language="vi", target_language="en", dimension=4096)
    resp = svc.search(req, request_id="r2")
    assert resp.vector_name == "en"


def test_search_en_to_vi_target_routes_vi():
    svc = make_service()
    req = SearchRequest(query="hi", query_language="en", target_language="vi", dimension=4096)
    resp = svc.search(req, request_id="r3")
    assert resp.vector_name == "vi"


def test_search_dimension_1024_collection():
    svc = make_service()
    req = SearchRequest(query="hi", dimension=DERIVED_DIMENSION, target_language="vi")
    resp = svc.search(req, request_id="r4")
    assert resp.collection == COLLECTION_1024


def test_search_injects_1024_transform():
    svc = make_service()
    req = SearchRequest(query="hi", dimension=DERIVED_DIMENSION, target_language="vi")
    resp = svc.search(req, request_id="r5")
    assert resp.timing_ms.transform_1024 > 0.0


def test_search_embedding_failure_raises():
    class BrokenEmbedding:
        def embed_texts_4096(self, texts):
            raise RuntimeError("down")

    svc = SearchService(embedding=BrokenEmbedding(), qdrant=FakeQdrantStore())
    req = SearchRequest(query="hi", dimension=4096, target_language="vi")
    with pytest.raises(SearchDependencyError):
        svc.search(req, request_id="r6")


def test_ready_when_all_ok():
    svc = make_service()
    r = svc.ready()
    assert r["ready"] is True


def test_ready_false_when_qdrant_down():
    qd = FakeQdrantStore()
    qd.ready_ok = False
    svc = SearchService(embedding=FakeEmbedding(), qdrant=qd)
    r = svc.ready()
    assert r["ready"] is False


def test_ready_checks_configured_collection_names():
    qd = FakeQdrantStore(collection_4096="custom_4096", collection_1024="custom_1024")
    svc = SearchService(embedding=FakeEmbedding(), qdrant=qd)
    detail = svc.ready()["detail"]
    assert "custom_4096" in detail
    assert "custom_1024" in detail
    assert COLLECTION_4096 not in detail
    assert COLLECTION_1024 not in detail


def test_search_queries_configured_collection():
    qd = FakeQdrantStore(collection_4096="custom_4096", collection_1024="custom_1024")
    svc = make_service(qdrant_store=qd)
    req = SearchRequest(query="hello", dimension=4096, target_language="vi", limit=10)
    resp = svc.search(req, request_id="c1")
    assert resp.collection == "custom_4096"
    assert qd.query_client.last["collection_name"] == "custom_4096"
    assert qd.query_client.last["using"] == "vi"


def test_search_response_collection_matches_actual_query_target():
    svc = make_service()
    req = SearchRequest(query="xin chao", query_language="vi", target_language="en", dimension=DERIVED_DIMENSION, limit=10)
    resp = svc.search(req, request_id="c2")
    assert resp.collection == svc.qdrant.collection_for(DERIVED_DIMENSION)
    assert resp.collection == COLLECTION_1024
    assert svc.qdrant.query_client.last["collection_name"] == COLLECTION_1024


def test_4096_1024_collection_mapping_is_independent():
    qd = FakeQdrantStore(collection_4096="only_4096_custom", collection_1024="only_1024_custom")
    assert qd.collection_for(FULL_DIMENSION) == "only_4096_custom"
    assert qd.collection_for(DERIVED_DIMENSION) == "only_1024_custom"


def test_from_env_propagates_collection_names(monkeypatch):
    import wemm_kaggle.search.qdrant_store as qs_module

    class FakeRemoteClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(qs_module, "QdrantClient", FakeRemoteClient)
    from wemm_kaggle.search.qdrant_store import COLLECTION_4096 as DEFAULT_4096
    from wemm_kaggle.search.qdrant_store import COLLECTION_1024 as DEFAULT_1024

    svc = SearchService.from_env(
        api_token="tok",
        collection_4096="env_4096",
        collection_1024="env_1024",
    )
    assert svc.qdrant.collection_for(FULL_DIMENSION) == "env_4096"
    assert svc.qdrant.collection_for(DERIVED_DIMENSION) == "env_1024"

    svc_default = SearchService.from_env(api_token="tok")
    assert svc_default.qdrant.collection_for(FULL_DIMENSION) == DEFAULT_4096
    assert svc_default.qdrant.collection_for(DERIVED_DIMENSION) == DEFAULT_1024
