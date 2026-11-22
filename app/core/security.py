from __future__ import annotations

from fastapi import HTTPException, status


def unauthorized(detail: str = "A valid Firebase ID token is required.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})
