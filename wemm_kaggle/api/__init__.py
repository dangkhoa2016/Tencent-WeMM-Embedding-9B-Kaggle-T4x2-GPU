"""REST API configuration for the Tencent WeMM-Embedding-9B T4x2 service."""

from .auth import build_auth_dependency
from .config import ApiSettings

__all__ = [
    "ApiSettings",
    "build_auth_dependency",
]
