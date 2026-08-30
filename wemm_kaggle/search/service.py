from __future__ import annotations

import time

from qdrant_client import QdrantClient

from .config import EMBEDDING_URL
from .embedding_client import EmbeddingClient, EmbeddingClientConfig
from .mrl import DERIVED_DIMENSION, FULL_DIMENSION, truncate_and_normalize
from .qdrant_store import COLLECTION_1024, COLLECTION_4096, MAX_POINTS, QdrantStore
from .schemas import SearchRequest, SearchResponse, SearchHit, SearchTimingMs


class SearchDependencyError(Exception):
    pass


class SearchService:
    def __init__(
        self,
        embedding: EmbeddingClient,
        qdrant: QdrantStore,
        embedding_url: str = EMBEDDING_URL,
    ) -> None:
        self.embedding = embedding
        self.qdrant = qdrant
        self.embedding_url = embedding_url

    @classmethod
    def from_env(
        cls,
        *,
        api_token: str,
        embedding_url: str = EMBEDDING_URL,
        qdrant_host: str = "127.0.0.1",
        qdrant_port: int = 6333,
        collection_4096: str = COLLECTION_4096,
        collection_1024: str = COLLECTION_1024,
    ) -> "SearchService":
        embedding = EmbeddingClient(
            EmbeddingClientConfig(base_url=embedding_url, api_token=api_token)
        )
        qdrant = QdrantStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_4096=collection_4096,
            collection_1024=collection_1024,
        )
        return cls(embedding=embedding, qdrant=qdrant, embedding_url=embedding_url)

    def ready(self) -> dict:
        detail: dict[str, str] = {}
        try:
            if self.embedding.ready():
                detail["embedding_api"] = "ready"
            else:
                detail["embedding_api"] = "unavailable"
        except Exception:
            detail["embedding_api"] = "unavailable"

        for dimension in (FULL_DIMENSION, DERIVED_DIMENSION):
            collection = self.qdrant.collection_for(dimension)
            try:
                self.qdrant.verify_schema(dimension, (0, MAX_POINTS))
                detail[collection] = "ready"
            except Exception:
                detail[collection] = "unavailable"

        ready = bool(detail) and all(v == "ready" for v in detail.values())
        return {"ready": ready, "detail": detail}

    def search(self, request: SearchRequest, request_id: str = "local") -> SearchResponse:
        started = time.monotonic()
        query_text = request.query

        emb_start = time.monotonic()
        try:
            full_vec = self.embedding.embed_texts_4096([query_text])[0]
        except Exception as exc:
            raise SearchDependencyError("embedding temporarily unavailable") from exc
        emb_ms = (time.monotonic() - emb_start) * 1000.0

        transform_ms = 0.0
        query_vector = full_vec
        if request.dimension == DERIVED_DIMENSION:
            tstart = time.monotonic()
            query_vector = truncate_and_normalize(full_vec, DERIVED_DIMENSION)
            transform_ms = (time.monotonic() - tstart) * 1000.0

        vector_name = "vi" if request.target_language == "vi" else "en"
        collection = self.qdrant.collection_for(request.dimension)

        try:
            qstart = time.monotonic()
            resp = self.qdrant.query(
                dimension=request.dimension,
                query=query_vector,
                using=vector_name,
                limit=int(request.limit),
                with_payload=True,
            )
            qdrant_ms = (time.monotonic() - qstart) * 1000.0
        except Exception as exc:
            raise SearchDependencyError("Qdrant search failed") from exc

        hits = []
        for rank, scored in enumerate(resp.points, start=1):
            payload = dict(scored.payload or {})
            hits.append(
                SearchHit(
                    rank=rank,
                    qid=str(payload.get("qid", "")),
                    score=float(scored.score),
                    text_en=str(payload.get("text_en", "")),
                    text_vi=str(payload.get("text_vi", "")),
                )
            )

        total_ms = (time.monotonic() - started) * 1000.0
        return SearchResponse(
            request_id=request_id,
            query_language=request.query_language,
            target_language=request.target_language,
            dimension=request.dimension,
            collection=collection,
            vector_name=vector_name,
            limit=request.limit,
            hits=hits,
            timing_ms=SearchTimingMs(
                embedding=emb_ms,
                transform_1024=transform_ms,
                qdrant=qdrant_ms,
                total=total_ms,
            ),
        )
