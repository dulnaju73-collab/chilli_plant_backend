from types import SimpleNamespace

from app.services.firebase_service import FirebaseService


class FakeBlob:
    def __init__(self, path, state):
        self.path = path
        self.state = state

    def upload_from_string(self, data, content_type):
        self.state["uploaded"] = (self.path, data, content_type)

    def delete(self):
        self.state["deleted_blob"] = self.path


class FakeBucket:
    name = "fake-bucket"

    def __init__(self, state):
        self.state = state

    def blob(self, path):
        return FakeBlob(path, self.state)


class FakeDocument:
    def __init__(self, uid, scan_id, state):
        self.uid = uid
        self.scan_id = scan_id
        self.state = state

    def collection(self, name):
        assert name == "scans"
        return self

    def document(self, scan_id):
        return FakeDocument(self.uid, scan_id, self.state)

    def set(self, data, merge=False):
        self.state["stored"] = (self.uid, self.scan_id, data)

    def get(self):
        data = self.state.get("existing", {}).get((self.uid, self.scan_id))
        return SimpleNamespace(exists=data is not None, to_dict=lambda: data or {})

    def delete(self):
        self.state["deleted_record"] = (self.uid, self.scan_id)


class FakeFirestore:
    SERVER_TIMESTAMP = object()

    def __init__(self, state):
        self.state = state

    def collection(self, name):
        assert name == "users"
        return self

    def document(self, uid):
        return FakeDocument(uid, "", self.state)


class FakeClients:
    def __init__(self, bucket=None):
        self.state = {"existing": {}}
        self.firestore = FakeFirestore(self.state)
        self.firestore_sdk = SimpleNamespace(SERVER_TIMESTAMP=FakeFirestore.SERVER_TIMESTAMP, Query=SimpleNamespace(DESCENDING="DESCENDING"))
        self.storage_bucket = bucket
        if bucket:
            bucket.state = self.state

    def require(self):
        return self


def test_scan_result_is_saved_under_verified_user_and_uses_server_timestamp():
    clients = FakeClients(FakeBucket({}))
    service = FirebaseService(clients)
    saved = service.save_scan("verified-user", "Healthy_Leaf", 0.8123, b"png", "PNG", "model-v1")
    uid, scan_id, data = clients.state["stored"]
    assert uid == "verified-user"
    assert scan_id == saved["id"]
    assert data["created_at"] is clients.firestore_sdk.SERVER_TIMESTAMP
    assert data["confidence"] == 81.23
    assert data["image_path"] == f"users/verified-user/scans/{scan_id}.png"
    assert clients.state["uploaded"][0] == data["image_path"]


def test_deletion_removes_only_own_result_and_stored_image():
    clients = FakeClients(FakeBucket({}))
    clients.state["existing"][("verified-user", "scan-1")] = {"image_path": "users/verified-user/scans/scan-1.png"}
    service = FirebaseService(clients)
    assert service.delete_result("verified-user", "scan-1") is True
    assert clients.state["deleted_blob"] == "users/verified-user/scans/scan-1.png"
    assert clients.state["deleted_record"] == ("verified-user", "scan-1")
    assert service.delete_result("someone-else", "scan-1") is False
