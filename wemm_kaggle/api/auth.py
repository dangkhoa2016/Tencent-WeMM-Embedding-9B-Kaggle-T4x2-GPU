from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _reject(error_code: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=error_code,
        headers={"WWW-Authenticate": "Bearer"},
    )


def build_auth_dependency(token: str):
    """Return a FastAPI dependency enforcing constant-time bearer-token auth."""

    def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
        if credentials is None:
            raise _reject("AUTH_REQUIRED")
        provided = credentials.credentials.encode("utf-8")
        expected = token.encode("utf-8")
        if not secrets.compare_digest(provided, expected):
            raise _reject("AUTH_INVALID")
        return None

    return require_auth