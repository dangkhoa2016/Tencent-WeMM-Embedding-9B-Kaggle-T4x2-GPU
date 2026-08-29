"""REST API configuration for the Tencent WeMM-Embedding-9B T4x2 service."""

from .auth import build_auth_dependency
from .config import ApiSettings
from .logging import emit_event
from .state import ServicePhase, ServiceState

__all__ = [
    "ApiSettings",
    "ServicePhase",
    "ServiceState",
    "build_auth_dependency",
    "emit_event",
]