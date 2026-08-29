from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

Dimension = Literal[4096, 1024]
DEFAULT_MAX_TEXT_CHARS = 8192


class TextEmbeddingRequest(BaseModel):
    inputs: list[str] = Field(min_length=1, max_length=4)
    dimension: Dimension = 4096
    max_text_chars: ClassVar[int] = DEFAULT_MAX_TEXT_CHARS

    @field_validator("inputs")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        limit = cls.max_text_chars
        for value in values:
            if not value or len(value) > limit:
                raise ValueError(f"each text must contain 1..{limit} characters")
        return values


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class TimingMs(BaseModel):
    queue: float
    inference: float
    total: float


class EmbeddingResponse(BaseModel):
    request_id: str
    model: str
    workload: str
    dimension: int
    count: int
    data: list[EmbeddingItem]
    timing_ms: TimingMs