from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

from .config import (
    EMBEDDING_INITIAL_BACKOFF_S,
    EMBEDDING_MAX_BACKOFF_S,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_NORM_TOLERANCE,
    EMBEDDING_TIMEOUT_S,
    EMBEDDING_URL,
)
from .mrl import FULL_DIMENSION

MAX_BATCH = 4
NON_RETRYABLE = frozenset({400, 401, 403, 404, 413, 415, 422})


@dataclass(frozen=True)
class EmbeddingClientConfig:
    base_url: str = EMBEDDING_URL
    api_token: str = ""
    timeout_s: float = EMBEDDING_TIMEOUT_S
    max_retries: int = EMBEDDING_MAX_RETRIES
    initial_backoff_s: float = EMBEDDING_INITIAL_BACKOFF_S
    max_backoff_s: float = EMBEDDING_MAX_BACKOFF_S
    norm_tolerance: float = EMBEDDING_NORM_TOLERANCE


class EmbeddingClient:
    def __init__(self, config: EmbeddingClientConfig | None = None) -> None:
        self._config = config or EmbeddingClientConfig()
        self._client = httpx.Client(timeout=self._config.timeout_s)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_token}",
            "Content-Type": "application/json",
        }

    def ready(self) -> bool:
        """Probe /readyz so a LOADING_MODEL dependency is never reported ready.

        Readiness requires HTTP 200 and the exact ``{"status": "ready",
        "state": "READY"}`` contract. Anything else fails closed.
        """
        try:
            resp = self._client.get(f"{self._config.base_url}/readyz")
            if resp.status_code != 200:
                return False
            try:
                body = resp.json()
            except ValueError:
                return False
            return body.get("status") == "ready" and body.get("state") == "READY"
        except httpx.HTTPError:
            return False

    def _norm(self, values: list[float]) -> float:
        return math.sqrt(sum(float(x) ** 2 for x in values))

    def _validate(self, request: list[str], vectors: list[list[float]]) -> None:
        if len(vectors) != len(request):
            raise ValueError(
                f"embedding API returned {len(vectors)} vectors for {len(request)} texts"
            )
        for vector in vectors:
            if len(vector) != FULL_DIMENSION:
                raise ValueError(
                    f"embedding API returned vector length {len(vector)}; expected {FULL_DIMENSION}"
                )
            if not all(math.isfinite(float(x)) for x in vector):
                raise ValueError("embedding API returned a non-finite (NaN/Inf) vector")
            norm = self._norm(vector)
            if abs(norm - 1.0) > self._config.norm_tolerance:
                raise ValueError(
                    f"embedding vector L2 norm {norm:.6f} outside tolerance "
                    f"{self._config.norm_tolerance}"
                )

    def embed_texts_4096(self, texts: list[str]) -> list[list[float]]:
        if len(texts) < 1 or len(texts) > MAX_BATCH:
            raise ValueError(f"batch length must be 1..{MAX_BATCH}; got {len(texts)}")
        attempt = 0
        delay = self._config.initial_backoff_s
        while True:
            attempt += 1
            try:
                resp = self._client.post(
                    f"{self._config.base_url}/v1/embeddings/text",
                    headers=self._headers(),
                    json={"inputs": texts, "dimension": FULL_DIMENSION},
                )
                if resp.status_code == 503 and attempt < self._config.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, self._config.max_backoff_s)
                    continue
                if resp.status_code in NON_RETRYABLE:
                    raise ValueError(
                        f"embedding API rejected request with HTTP {resp.status_code}"
                    )
                if resp.status_code != 200:
                    raise RuntimeError(f"embedding API returned HTTP {resp.status_code}")
                body = resp.json()
                vectors = [
                    list(item["embedding"]) for item in sorted(body["data"], key=lambda x: x["index"])
                ]
                self._validate(texts, vectors)
                return vectors
            except httpx.HTTPError as exc:
                if attempt >= self._config.max_retries:
                    raise RuntimeError(f"embedding API unreachable after {attempt} attempts") from exc
                time.sleep(delay)
                delay = min(delay * 2, self._config.max_backoff_s)
