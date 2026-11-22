from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.firebase import FirebaseUnavailable, firebase
from app.schemas.results import ScanResult


def _normalise_result(uid: str, snapshot: Any) -> ScanResult:
    data = snapshot.to_dict() or {}
    created_at = data.get("created_at")
    if created_at is not None and not isinstance(created_at, datetime):
        created_at = created_at.to_datetime() if hasattr(created_at, "to_datetime") else None
    return ScanResult(
        id=snapshot.id,
        disease=data.get("disease", "Unknown"),
        confidence=float(data.get("confidence", 0)),
        is_healthy=bool(data.get("is_healthy", False)),
        validation_status=data.get("validation_status", "validated"),
        created_at=created_at,
        image_url=data.get("image_url"),
        image_path=data.get("image_path"),
        prediction_model_version=data.get("prediction_model_version", "unknown"),
    )


class FirebaseService:
    def __init__(self, clients: Any = None) -> None:
        self.clients = clients or firebase

    def _db(self) -> Any:
        return self.clients.require().firestore

    def _profile_ref(self, uid: str) -> Any:
        return self._db().collection("users").document(uid)

    def get_profile(self, claims: dict[str, Any]) -> dict[str, Any]:
        uid = claims["uid"]
        ref = self._profile_ref(uid)
        snapshot = ref.get()
        profile = snapshot.to_dict() if snapshot.exists else {}
        if not profile:
            created_at = None
            try:
                auth_user = self.clients.auth.get_user(uid)
                created_ms = auth_user.user_metadata.creation_timestamp
                if created_ms:
                    created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
            except Exception:
                created_at = None
            profile = {
                "username": claims.get("name") or (claims.get("email", "").split("@")[0]) or "User",
                "email": claims.get("email"),
                "created_at": created_at,
            }
            ref.set({**profile, "updated_at": self.clients.firestore_sdk.SERVER_TIMESTAMP}, merge=True)
        created_at = profile.get("created_at")
        if created_at is not None and not isinstance(created_at, datetime):
            created_at = created_at.to_datetime() if hasattr(created_at, "to_datetime") else None
        return {"uid": uid, "username": profile.get("username", "User"), "email": claims.get("email") or profile.get("email"), "created_at": created_at}

    def update_profile(self, uid: str, username: str) -> dict[str, Any]:
        ref = self._profile_ref(uid)
        ref.set({"username": username.strip(), "updated_at": self.clients.firestore_sdk.SERVER_TIMESTAMP}, merge=True)
        profile = ref.get().to_dict() or {}
        created_at = profile.get("created_at")
        if created_at is not None and not isinstance(created_at, datetime):
            created_at = created_at.to_datetime() if hasattr(created_at, "to_datetime") else None
        return {"uid": uid, "username": profile.get("username", username.strip()), "email": profile.get("email"), "created_at": created_at}

    def save_scan(self, uid: str, disease: str, confidence: float, image_data: bytes, image_format: str, model_version: str, validation_status: str = "validated") -> dict[str, Any]:
        clients = self.clients.require()
        scan_id = uuid.uuid4().hex
        image_path = None
        image_url = None
        if clients.storage_bucket is not None:
            extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image_format]
            image_path = f"users/{uid}/scans/{scan_id}.{extension}"
            blob = clients.storage_bucket.blob(image_path)
            content_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[image_format]
            blob.upload_from_string(image_data, content_type=content_type)
        ref = self._profile_ref(uid).collection("scans").document(scan_id)
        try:
            ref.set({
                "disease": disease,
                "confidence": round(confidence * 100, 2),
                "is_healthy": disease == "Healthy_Leaf",
                "image_url": image_url,
                "image_path": image_path,
                "validation_status": validation_status,
                "created_at": clients.firestore_sdk.SERVER_TIMESTAMP,
                "prediction_model_version": model_version,
            })
        except Exception:
            if image_path and clients.storage_bucket is not None:
                clients.storage_bucket.blob(image_path).delete()
            raise
        return {"id": scan_id, "disease": disease, "confidence": round(confidence * 100, 2), "is_healthy": disease == "Healthy_Leaf", "image_url": image_url, "image_path": image_path, "validation_status": validation_status, "created_at": datetime.now(timezone.utc)}

    def _query(self, uid: str, disease: str | None = None, diseases: list[str] | None = None, start: datetime | None = None, end: datetime | None = None) -> Any:
        query = self._profile_ref(uid).collection("scans")
        if disease:
            query = query.where("disease", "==", disease)
        elif diseases:
            query = query.where("disease", "in", diseases)
        if start:
            query = query.where("created_at", ">=", start)
        if end:
            query = query.where("created_at", "<", end)
        return query.order_by("created_at", direction=self.clients.firestore_sdk.Query.DESCENDING)

    def list_results(self, uid: str, page: int, page_size: int, search: str | None, disease: str | None, date: datetime | None, year: int | None, timezone_offset_minutes: int = 0) -> dict[str, Any]:
        local_offset = timedelta(minutes=timezone_offset_minutes)
        start, end = None, None
        if date:
            start = date.replace(hour=0, minute=0, second=0, microsecond=0) - local_offset
            end = start + timedelta(days=1)
        elif year:
            start = datetime(year, 1, 1, tzinfo=timezone.utc) - local_offset
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - local_offset
        search = search.strip().lower() if search else None
        search_start, search_end = _date_search_bounds(search) if search else (None, None)
        if search_start:
            search_start -= local_offset
            search_end -= local_offset
            if start is None:
                start, end = search_start, search_end
            else:
                start = max(start, search_start)
                end = min(end, search_end) if end else search_end
            if end is not None and start >= end:
                return {"items": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}
        searchable_diseases = None
        if search and not search_start:
            searchable_diseases = [label for label in _CLASS_LABELS if search in _display_label(label).lower() or search in label.lower()]
            if disease and disease not in searchable_diseases:
                return {"items": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}
            if not searchable_diseases:
                return {"items": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}
        base = self._query(uid, disease=disease, diseases=searchable_diseases, start=start, end=end)
        count = base.count(alias="total").get()
        total = int(count[0][0].value)
        selected = list(base.offset((page - 1) * page_size).limit(page_size).stream())
        items = [_normalise_result(uid, doc).model_dump() for doc in selected]
        return {"items": items, "total": total, "page": page, "page_size": page_size, "has_more": page * page_size < total}

    def get_result(self, uid: str, result_id: str) -> ScanResult | None:
        snap = self._profile_ref(uid).collection("scans").document(result_id).get()
        return _normalise_result(uid, snap) if snap.exists else None

    def delete_result(self, uid: str, result_id: str) -> bool:
        clients = self.clients.require()
        ref = self._profile_ref(uid).collection("scans").document(result_id)
        snap = ref.get()
        if not snap.exists:
            return False
        data = snap.to_dict() or {}
        image_path = data.get("image_path")
        if image_path and clients.storage_bucket is not None:
            clients.storage_bucket.blob(image_path).delete()
        ref.delete()
        return True


_CLASS_LABELS = ("Bacterial_spot", "Cercospora_Leaf_Spot", "Curl_Virus", "Healthy_Leaf", "Nutrition_Deficiency", "White_Spot")


def _display_label(label: str) -> str:
    return label.replace("_", " ")


def _date_search_bounds(search: str | None) -> tuple[datetime | None, datetime | None]:
    if not search:
        return None, None
    try:
        if re.fullmatch(r"\d{4}", search):
            year = int(search)
            return datetime(year, 1, 1, tzinfo=timezone.utc), datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        formats = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y")
        for fmt in formats:
            try:
                parsed = datetime.strptime(search.title() if "%b" in fmt or "%B" in fmt else search, fmt).replace(tzinfo=timezone.utc)
                return parsed, parsed + timedelta(days=1)
            except ValueError:
                continue
        for fmt in ("%b %Y", "%B %Y"):
            try:
                parsed = datetime.strptime(search.title(), fmt).replace(tzinfo=timezone.utc)
                return parsed, (parsed.replace(year=parsed.year + 1, month=1) if parsed.month == 12 else parsed.replace(month=parsed.month + 1))
            except ValueError:
                continue
    except (ValueError, OverflowError):
        return None, None
    return None, None
