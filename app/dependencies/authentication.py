from __future__ import annotations

from typing import Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.firebase import FirebaseUnavailable, firebase
from app.core.security import unauthorized

bearer = HTTPBearer(auto_error=False)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized()
    try:
        clients = firebase.require()
        claims = clients.auth.verify_id_token(credentials.credentials, check_revoked=True)
        if not claims.get("uid"):
            raise unauthorized()
        return claims
    except FirebaseUnavailable as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise unauthorized("The Firebase ID token is invalid or expired.") from exc
