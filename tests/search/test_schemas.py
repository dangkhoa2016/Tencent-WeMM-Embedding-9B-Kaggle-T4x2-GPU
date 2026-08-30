from __future__ import annotations

import pytest
from pydantic import ValidationError

from wemm_kaggle.search.schemas import SearchRequest, SearchResponse, SearchHit


def test_valid_request():
    r = SearchRequest(query="hello", query_language="en", target_language="vi", dimension=4096, limit=10)
    assert r.limit == 10


def test_request_blank_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


def test_request_invalid_language():
    with pytest.raises(ValidationError):
        SearchRequest(query="x", query_language="fr")


def test_request_invalid_dimension():
    with pytest.raises(ValidationError):
        SearchRequest(query="x", dimension=2048)


def test_request_limit_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="x", limit=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="x", limit=101)


def test_response_roundtrip():
    resp = SearchResponse(
        request_id="rid",
        query_language="en",
        target_language="vi",
        dimension=4096,
        collection="c",
        vector_name="vi",
        limit=10,
        hits=[SearchHit(rank=1, qid="Q42", score=0.9, text_en="a", text_vi="b")],
        timing_ms={"embedding": 1.0, "transform_1024": 0.0, "qdrant": 2.0, "total": 3.0},
    )
    assert resp.hits[0].qid == "Q42"
    assert resp.hits[0].score == 0.9
    assert resp.vector_name == "vi"
