from __future__ import annotations

import logging
import json
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class FirebaseUnavailable(RuntimeError):
    pass


class FirebaseClients:
    def __init__(self) -> None:
        self.app: Any = None
        self.auth: Any = None
        self.firestore: Any = None
        self.firestore_sdk: Any = None
        self.storage_bucket: Any = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.auth is not None and self.firestore is not None

    def initialize(self) -> None:
        if self.ready:
            return
        try:
            import firebase_admin
            from firebase_admin import auth, credentials, firestore, storage

            options = {"projectId": settings.firebase_project_id} if settings.firebase_project_id else None
            if settings.firebase_service_account_json:
                try:
                    service_account = json.loads(settings.firebase_service_account_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON must contain valid service account JSON.") from exc
                credential = credentials.Certificate(service_account)
            elif settings.firebase_service_account_path:
                credential = credentials.Certificate(settings.firebase_service_account_path)
            else:
                credential = credentials.ApplicationDefault()

            self.app = firebase_admin.initialize_app(credential, options=options)
            self.auth = auth
            self.firestore_sdk = firestore
            self.firestore = firestore.client(self.app)
            if settings.firebase_storage_bucket:
                self.storage_bucket = storage.bucket(settings.firebase_storage_bucket, app=self.app)
            self.error = None
        except Exception as exc:  # Startup must still expose health/config guidance.
            self.error = str(exc)
            logger.exception("Firebase Admin initialization failed")

    def require(self) -> "FirebaseClients":
        if not self.ready:
            raise FirebaseUnavailable("Firebase Admin is not configured. Set FIREBASE_PROJECT_ID and provide service account credentials or Application Default Credentials.")
        return self


firebase = FirebaseClients()
