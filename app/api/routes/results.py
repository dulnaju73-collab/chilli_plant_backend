from datetime import date, datetime, time, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.core.firebase import FirebaseUnavailable
from app.dependencies.authentication import current_user
from app.schemas.results import ResultsPage, ScanResult
from app.services.firebase_service import FirebaseService

router = APIRouter(prefix="/results", tags=["results"])
service = FirebaseService()


@router.get("", response_model=ResultsPage)
def list_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=80),
    disease: str | None = Query(None, max_length=64),
    scan_date: date | None = Query(None, alias="date"),
    year: int | None = Query(None, ge=2000, le=2200),
    timezone_offset_minutes: int = Query(0, ge=-840, le=840),
    claims: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    if scan_date and year:
        raise HTTPException(status_code=422, detail="Choose either date or year filtering.")
    try:
        date_start = datetime.combine(scan_date, time.min, tzinfo=timezone.utc) if scan_date else None
        disease_key = None
        if disease:
            compact = disease.strip().replace(" ", "_").lower()
            disease_key = next((label for label in ("Bacterial_spot", "Cercospora_Leaf_Spot", "Curl_Virus", "Healthy_Leaf", "Nutrition_Deficiency", "White_Spot") if label.lower() == compact), None)
            if disease_key is None:
                raise HTTPException(status_code=422, detail="Unknown supported disease class.")
        return service.list_results(claims["uid"], page, page_size, search, disease_key, date_start, year, timezone_offset_minutes)
    except FirebaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Firebase scan history is temporarily unavailable.") from exc


@router.get("/{result_id}", response_model=ScanResult)
def get_result(result_id: str, claims: dict[str, Any] = Depends(current_user)) -> ScanResult:
    try:
        result = service.get_result(claims["uid"], result_id)
    except FirebaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Firebase scan history is temporarily unavailable.") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Scan result not found.")
    return result


@router.delete("/{result_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_result(result_id: str, claims: dict[str, Any] = Depends(current_user)) -> Response:
    try:
        deleted = service.delete_result(claims["uid"], result_id)
    except FirebaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Firebase scan history is temporarily unavailable.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan result not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
