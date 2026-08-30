from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Language = Literal["en", "vi"]
AllowedDimension = Literal[4096, 1024]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    query_language: Language = "en"
    target_language: Language = "vi"
    dimension: AllowedDimension = 4096
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be non-empty after trimming whitespace")
        return value


class SearchHit(BaseModel):
    rank: int
    qid: str
    score: float
    text_en: str
    text_vi: str


class SearchTimingMs(BaseModel):
    embedding: float = 0.0
    transform_1024: float = 0.0
    qdrant: float = 0.0
    total: float = 0.0


class SearchResponse(BaseModel):
    request_id: str
    query_language: str
    target_language: str
    dimension: int
    collection: str
    vector_name: str
    limit: int
    hits: list[SearchHit]
    timing_ms: SearchTimingMs
