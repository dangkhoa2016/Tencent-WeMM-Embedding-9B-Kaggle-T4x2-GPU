from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

EMBEDDING_URL = "http://127.0.0.1:8090"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
SEARCH_HOST = "127.0.0.1"
SEARCH_PORT = 8091

EMBEDDING_TIMEOUT_S = 60.0
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_INITIAL_BACKOFF_S = 0.5
EMBEDDING_MAX_BACKOFF_S = 8.0
EMBEDDING_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SearchSettings:
    search_host: str = SEARCH_HOST
    search_port: int = SEARCH_PORT
    embedding_url: str = EMBEDDING_URL
    embedding_token: str = ""
    qdrant_host: str = QDRANT_HOST
    qdrant_port: int = QDRANT_PORT
    qdrant_timeout_s: float = 60.0
    embedding_timeout_s: float = EMBEDDING_TIMEOUT_S
    embedding_max_retries: int = EMBEDDING_MAX_RETRIES
    embedding_initial_backoff_s: float = EMBEDDING_INITIAL_BACKOFF_S
    embedding_max_backoff_s: float = EMBEDDING_MAX_BACKOFF_S
    embedding_norm_tolerance: float = EMBEDDING_NORM_TOLERANCE
    collections_4096: str = "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1"
    collections_1024: str = "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1"
    deferred: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        host = self.search_host.lower().strip()
        if host not in LOOPBACK_HOSTS:
            raise ValueError(
                "WEMM_SEARCH_HOST must be a loopback address in "
                f"{sorted(LOOPBACK_HOSTS)}; got {self.search_host!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> "SearchSettings":
        def as_int(name: str, default: int) -> int:
            try:
                return int(mapping.get(name, "") or default)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer; got {mapping.get(name)!r}") from exc

        def as_float(name: str, default: float) -> float:
            try:
                return float(mapping.get(name, "") or default)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number; got {mapping.get(name)!r}") from exc

        return cls(
            search_host=mapping.get("WEMM_SEARCH_HOST", SEARCH_HOST),
            search_port=as_int("WEMM_SEARCH_PORT", SEARCH_PORT),
            embedding_url=mapping.get("WEMM_EMBEDDING_URL", EMBEDDING_URL),
            embedding_token=mapping.get("WEMM_API_TOKEN", ""),
            qdrant_host=mapping.get("WEMM_QDRANT_HOST", QDRANT_HOST),
            qdrant_port=as_int("WEMM_QDRANT_PORT", QDRANT_PORT),
            qdrant_timeout_s=as_float("WEMM_QDRANT_TIMEOUT_S", 60.0),
            embedding_timeout_s=as_float("WEMM_EMBEDDING_TIMEOUT_S", EMBEDDING_TIMEOUT_S),
            embedding_max_retries=as_int("WEMM_EMBEDDING_MAX_RETRIES", EMBEDDING_MAX_RETRIES),
            embedding_initial_backoff_s=as_float(
                "WEMM_EMBEDDING_INITIAL_BACKOFF_S", EMBEDDING_INITIAL_BACKOFF_S
            ),
            embedding_max_backoff_s=as_float(
                "WEMM_EMBEDDING_MAX_BACKOFF_S", EMBEDDING_MAX_BACKOFF_S
            ),
            embedding_norm_tolerance=as_float(
                "WEMM_EMBEDDING_NORM_TOLERANCE", EMBEDDING_NORM_TOLERANCE
            ),
            collections_4096=mapping.get(
                "WEMM_COLLECTION_4096", "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1"
            ),
            collections_1024=mapping.get(
                "WEMM_COLLECTION_1024", "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1"
            ),
        )


settings = SearchSettings.from_mapping(dict(os.environ))
