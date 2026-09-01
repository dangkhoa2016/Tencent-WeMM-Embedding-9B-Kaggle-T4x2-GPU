from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import QdrantConfig


@dataclass(frozen=True)
class QdrantSearchResult:
    rank: int
    point_id: str
    qid: str
    score: float
    payload: dict[str, Any]


class QdrantRetriever:
    """Small provider-neutral Qdrant client used by demos and services."""

    def __init__(self, config: QdrantConfig, client=None):
        self.config = config
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=config.url,
                timeout=float(config.timeout_s),
            )
        self.client = client

    def query(
        self,
        *,
        dimension: int,
        vector: Sequence[float],
        using: str,
        limit: int = 5,
    ) -> list[QdrantSearchResult]:
        response = self.client.query_points(
            collection_name=self.config.collection(int(dimension)),
            query=list(vector),
            using=str(using),
            limit=int(limit),
            with_payload=True,
        )
        rows: list[QdrantSearchResult] = []
        for rank, point in enumerate(response.points, 1):
            payload = dict(point.payload or {})
            rows.append(
                QdrantSearchResult(
                    rank=rank,
                    point_id=str(point.id),
                    qid=str(payload.get("qid", "")),
                    score=float(point.score),
                    payload=payload,
                )
            )
        if not rows:
            raise RuntimeError(
                f"Qdrant returned no hits for dimension={dimension}, using={using!r}"
            )
        return rows

    @staticmethod
    def expected_rank(
        rows: Sequence[QdrantSearchResult],
        expected_qid: str,
    ) -> int | None:
        for row in rows:
            if row.qid == expected_qid:
                return row.rank
        return None
