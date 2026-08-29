from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MIN_TOKEN_LENGTH = 32
PROTOCOL_MAX_BATCH = 4
DEFAULT_MAX_TEXT_CHARS = 8192
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8090
    api_token: str = ""
    max_batch: int = PROTOCOL_MAX_BATCH
    queue_max_items: int = 16
    request_timeout_s: float = 30.0
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_image_edge: int = 4096
    max_image_pixels: int = 16_777_216
    per_gpu_mib: int = 14200
    evidence_dir: Path | None = None

    def __post_init__(self) -> None:
        if len(self.api_token) < MIN_TOKEN_LENGTH:
            raise ValueError(
                f"WEMM_API_TOKEN must be at least {MIN_TOKEN_LENGTH} characters; got length {len(self.api_token)}"
            )
        if not 1 <= self.max_batch <= PROTOCOL_MAX_BATCH:
            raise ValueError(f"WEMM_MAX_BATCH must be in 1..{PROTOCOL_MAX_BATCH}; got {self.max_batch}")
        if not 1 <= self.max_text_chars <= DEFAULT_MAX_TEXT_CHARS:
            raise ValueError(f"WEMM_MAX_TEXT_CHARS must be in 1..{DEFAULT_MAX_TEXT_CHARS}; got {self.max_text_chars}")
        if self.queue_max_items < 1 or self.request_timeout_s <= 0:
            raise ValueError("queue limit and request timeout must be positive")
        if min(self.max_image_bytes, self.max_image_edge, self.max_image_pixels) < 1:
            raise ValueError("image safety limits must be positive")
        if self.per_gpu_mib < 1024:
            raise ValueError("WEMM_PER_GPU_MIB is implausibly small")
        host = self.host.lower().strip()
        if host not in LOOPBACK_HOSTS:
            raise ValueError(
                "WEMM_API_HOST must be a loopback address in "
                f"{sorted(LOOPBACK_HOSTS)}; got {self.host!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> "ApiSettings":
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

        evidence = mapping.get("WEMM_API_EVIDENCE_DIR")
        return cls(
            host=mapping.get("WEMM_API_HOST", "127.0.0.1"),
            port=as_int("WEMM_API_PORT", 8090),
            api_token=mapping.get("WEMM_API_TOKEN", ""),
            max_batch=as_int("WEMM_MAX_BATCH", 4),
            queue_max_items=as_int("WEMM_QUEUE_MAX_ITEMS", 16),
            request_timeout_s=as_float("WEMM_REQUEST_TIMEOUT_S", 30.0),
            max_text_chars=as_int("WEMM_MAX_TEXT_CHARS", 8192),
            max_image_bytes=as_int("WEMM_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES),
            max_image_edge=as_int("WEMM_MAX_IMAGE_EDGE", 4096),
            max_image_pixels=as_int("WEMM_MAX_IMAGE_PIXELS", 16_777_216),
            per_gpu_mib=as_int("WEMM_PER_GPU_MIB", 14200),
            evidence_dir=Path(evidence) if evidence else None,
        )

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls.from_mapping(os.environ)