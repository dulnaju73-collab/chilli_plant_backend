from datetime import date
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from PIL import Image

import app.main as main_module
from app.core.firebase import firebase
from app.dependencies.authentication import current_user
from app.services.prediction_service import PredictionService
from app.services.image_validation_service import ImageValidationService
from app.dependencies import authentication


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(firebase, "initialize", lambda: None)
    monkeypatch.setattr(main_module, "DiseaseClassifier", lambda *_, **__: object())
    monkeypatch.setattr(main_module, "ChilliImageValidator", lambda *_, **__: object())
    main_module.app.dependency_overrides.clear()
    with TestClient(main_module.app) as test_client:
        yield test_client
    main_module.app.dependency_overrides.clear()


def test_protected_endpoints_reject_missing_firebase_token(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401


def test_versioned_health_endpoint_reports_readiness(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["disease_model_ready"] is True
    assert "validator_ready" in response.json()


def test_auth_dependency_uses_verified_firebase_claims(monkeypatch):
    class Auth:
        def verify_id_token(self, token, check_revoked):
            assert token == "firebase-id-token"
            assert check_revoked is True
            return {"uid": "verified-uid", "email": "user@example.com"}

    monkeypatch.setattr(firebase, "require", lambda: SimpleNamespace(auth=Auth()))
    claims = authentication.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials="firebase-id-token"))
    assert claims["uid"] == "verified-uid"


def test_auth_dependency_rejects_invalid_firebase_token(monkeypatch):
    class Auth:
        def verify_id_token(self, *_args, **_kwargs):
            raise ValueError("invalid")

    monkeypatch.setattr(firebase, "require", lambda: SimpleNamespace(auth=Auth()))
    with pytest.raises(HTTPException) as error:
        authentication.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token"))
    assert error.value.status_code == 401


def test_profile_and_history_use_verified_uid(client, monkeypatch):
    from app.api.routes import results, users

    captured = {}

    class FakeUsers:
        def get_profile(self, claims):
            return {"uid": claims["uid"], "username": "grower", "email": "grower@example.com"}

    class FakeResults:
        def list_results(self, uid, page, page_size, search, disease, scan_date, year, timezone_offset_minutes=0):
            captured.update(uid=uid, search=search, disease=disease, date=scan_date, year=year, page=page)
            return {"items": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}

    monkeypatch.setattr(users, "service", FakeUsers())
    monkeypatch.setattr(results, "service", FakeResults())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user-1", "email": "grower@example.com"}

    profile = client.get("/api/v1/users/me")
    history = client.get("/api/v1/results", params={"search": "Cercospora", "date": "2026-10-07"})
    assert profile.status_code == 200
    assert profile.json()["uid"] == "verified-user-1"
    assert history.status_code == 200
    assert captured["uid"] == "verified-user-1"
    assert captured["search"] == "Cercospora"
    assert captured["date"].date() == date(2026, 10, 7)


def test_profile_update_uses_authenticated_uid(client, monkeypatch):
    from app.api.routes import users

    captured = {}

    class FakeUsers:
        def update_profile(self, uid, username):
            captured.update(uid=uid, username=username)
            return {"uid": uid, "username": username, "email": "user@example.com"}

    monkeypatch.setattr(users, "service", FakeUsers())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user"}
    response = client.put("/api/v1/users/me", json={"username": "pepper-grower"})
    assert response.status_code == 200
    assert response.json()["username"] == "pepper-grower"
    assert captured == {"uid": "verified-user", "username": "pepper-grower"}


def test_results_filters_and_pagination_are_forwarded(client, monkeypatch):
    from app.api.routes import results

    captured = {}

    class FakeResults:
        def list_results(self, uid, page, page_size, search, disease, scan_date, year, timezone_offset_minutes=0):
            captured.update(uid=uid, page=page, page_size=page_size, disease=disease, year=year)
            return {"items": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}

    monkeypatch.setattr(results, "service", FakeResults())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "user-two"}
    response = client.get("/api/v1/results", params={"page": 2, "page_size": 15, "disease": "Cercospora Leaf Spot", "year": 2026})
    assert response.status_code == 200
    assert captured == {"uid": "user-two", "page": 2, "page_size": 15, "disease": "Cercospora_Leaf_Spot", "year": 2026}


def test_results_reject_unknown_disease_filter(client, monkeypatch):
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "user-two"}
    response = client.get("/api/v1/results", params={"disease": "Tomato Blight"})
    assert response.status_code == 422


def test_unauthorized_user_cannot_delete_another_user_result(client, monkeypatch):
    from app.api.routes import results

    captured = {}

    class FakeResults:
        def delete_result(self, uid, result_id):
            captured.update(uid=uid, result_id=result_id)
            return False

    monkeypatch.setattr(results, "service", FakeResults())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "current-user"}
    response = client.delete("/api/v1/results/other-users-scan")
    assert response.status_code == 404
    assert captured == {"uid": "current-user", "result_id": "other-users-scan"}


def test_result_detail_and_delete_are_scoped_to_authenticated_uid(client, monkeypatch):
    from app.api.routes import results
    from app.schemas.results import ScanResult

    captured = []

    class FakeResults:
        def get_result(self, uid, result_id):
            captured.append(("get", uid, result_id))
            return ScanResult(
                id=result_id,
                disease="Healthy_Leaf",
                confidence=92.5,
                is_healthy=True,
                validation_status="validated",
                prediction_model_version="model-v1",
            )

        def delete_result(self, uid, result_id):
            captured.append(("delete", uid, result_id))
            return True

    monkeypatch.setattr(results, "service", FakeResults())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "owner-uid"}
    detail = client.get("/api/v1/results/scan-7")
    deleted = client.delete("/api/v1/results/scan-7")
    assert detail.status_code == 200
    assert detail.json()["id"] == "scan-7"
    assert deleted.status_code == 204
    assert captured == [("get", "owner-uid", "scan-7"), ("delete", "owner-uid", "scan-7")]


def test_history_returns_service_unavailable_when_firebase_fails(client, monkeypatch):
    from app.api.routes import results
    from app.core.firebase import FirebaseUnavailable

    class BrokenResults:
        def list_results(self, *_args, **_kwargs):
            raise FirebaseUnavailable("Firebase is unavailable.")

    monkeypatch.setattr(results, "service", BrokenResults())
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user"}
    response = client.get("/api/v1/results")
    assert response.status_code == 503


def test_non_chilli_validation_stops_disease_inference():
    calls = {"classifier": 0, "storage": 0}

    class Validator:
        def validate(self, _):
            return False, 0.99

    class Classifier:
        def predict(self, _):
            calls["classifier"] += 1
            raise AssertionError("Disease inference must not run for invalid images")

    class Storage:
        def save_scan(self, **_):
            calls["storage"] += 1

    service = PredictionService(Classifier(), SimpleNamespace(validator=Validator(), validate=Validator().validate), Storage())
    with pytest.raises(Exception) as error:
        service.analyze("uid", b"image", object(), "PNG")
    assert getattr(error.value, "status_code", None) == 422
    assert calls == {"classifier": 0, "storage": 0}


def _png_upload():
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(image, format="PNG")
    return {"file": ("leaf.png", image.getvalue(), "image/png")}


def test_analyze_returns_model_confidence_and_persists_scan(client):
    from app.api.routes import predictions

    class Validator:
        def validate(self, _):
            return True, 0.98

    class Classifier:
        def predict(self, _):
            return "Healthy_Leaf", 0.8562

    class Storage:
        def save_scan(self, uid, disease, confidence, image_data, image_format, model_version, validation_status="validated"):
            assert uid == "verified-user"
            assert disease == "Healthy_Leaf"
            assert confidence == pytest.approx(0.8562)
            assert image_format == "PNG"
            assert validation_status == "validated"
            return {"id": "server-created-id", "disease": disease, "confidence": 85.62, "is_healthy": True, "created_at": "2026-11-22T00:00:00Z", "validation_status": validation_status}

    validator = Validator()
    predictions.configure_prediction_service(PredictionService(Classifier(), ImageValidationService(validator), Storage()))
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user"}
    response = client.post("/api/v1/predictions/analyze", files=_png_upload())
    assert response.status_code == 200
    assert response.json()["prediction"]["id"] == "server-created-id"
    assert response.json()["prediction"]["confidence"] == 85.62
    assert response.json()["prediction"]["validation_status"] == "validated"


def test_analyze_rejects_non_chilli_image_with_422_before_disease_inference(client):
    from app.api.routes import predictions

    called = {"classifier": 0, "storage": 0}

    class Validator:
        def validate(self, _):
            return False, 0.12

    class Classifier:
        def predict(self, _):
            called["classifier"] += 1
            return "Healthy_Leaf", 0.99

    class Storage:
        def save_scan(self, *_args, **_kwargs):
            called["storage"] += 1
            return {}

    predictions.configure_prediction_service(PredictionService(Classifier(), ImageValidationService(Validator()), Storage()))
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user"}
    response = client.post("/api/v1/predictions/analyze", files=_png_upload())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_PLANT_IMAGE"
    assert called == {"classifier": 0, "storage": 0}


def test_missing_validator_is_reported_and_disease_model_is_not_run(client):
    main_module.app.dependency_overrides[current_user] = lambda: {"uid": "verified-user"}
    response = client.post("/api/v1/predictions/validate", files=_png_upload())
    assert response.status_code == 503
    assert "IMAGE_VALIDATION_MODEL_PATH" in response.json()["detail"]
