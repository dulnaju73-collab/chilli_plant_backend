from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies.authentication import current_user
from app.schemas.user import UserProfile, UserProfileUpdate
from app.services.firebase_service import FirebaseService
from app.core.firebase import FirebaseUnavailable

router = APIRouter(prefix="/users", tags=["users"])
service = FirebaseService()


@router.get("/me", response_model=UserProfile)
def get_me(claims: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    try:
        return service.get_profile(claims)
    except FirebaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Firebase profile service is temporarily unavailable.") from exc


@router.put("/me", response_model=UserProfile)
def update_me(payload: UserProfileUpdate, claims: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    try:
        return service.update_profile(claims["uid"], payload.username)
    except FirebaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Firebase profile service is temporarily unavailable.") from exc
