from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ServicePhase(str, Enum):
    STARTING = "STARTING"
    LOADING_MODEL = "LOADING_MODEL"
    READY = "READY"
    DRAINING = "DRAINING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class ServiceState:
    phase: ServicePhase = ServicePhase.STARTING
    error: str | None = None
    model_load_seconds: float | None = None