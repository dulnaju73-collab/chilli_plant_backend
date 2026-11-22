# Chilli Plant Health API

FastAPI backend for the Flutter chilli leaf health application. The backend owns identity verification, image validation, disease inference, scan persistence, and user-scoped history. Interactive OpenAPI documentation is available at `/docs`.

## Architecture

```text
app/
  api/routes/       Versioned users, predictions, and results endpoints
  core/             Environment configuration, Firebase Admin, token security
  dependencies/     Verified Firebase ID token dependency
  models/           Disease TFLite classifier and separate chilli validator
  schemas/          Pydantic request and response contracts
  services/         Prediction workflow, image validation, Firebase persistence
  utils/            Safe image decoding and domain exceptions
```

The bundled six-class disease model is `models/chilli_leaf_disease_final.tflite` (1.70 MB); its matching class order is in `models/class_names.json`. Inspection of the TFLite flatbuffer confirms a float32 `[1, 128, 128, 3]` input and six outputs. The inference implementation currently converts to RGB, resizes bilinearly to 128 × 128, and divides by 255. The checked-in training script does not document this exact artifact: it builds a 224 × 224 MobileNetV2 while the bundled model has a different shape and operator graph, and it applies rescaling in both its data generator and model. The original training artifact/data are absent, so input normalization cannot be independently verified from the repository. Do not treat normalization as confirmed until the model owner supplies the exact export/training pipeline or a validated reference inference.

Chilli identity validation is a separate model stage. The repository currently does **not** contain a chilli-versus-other-plant validator model. Configure a separately trained TFLite validator using `IMAGE_VALIDATION_MODEL_PATH`, set its chilli-positive output index, and validate its input normalization against its training code. Until then `/predictions/validate` and `/predictions/analyze` return HTTP 503 with setup guidance; disease inference is not run. The disease model is not used as a substitute validator.

## Local setup

Use Python 3.10–3.12 (TensorFlow 2.18 does not support every newer Python release):

```sh
cd services/chilli_backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`. Configure Firebase Admin credentials via a service account file stored outside the repository, the `FIREBASE_SERVICE_ACCOUNT_JSON` secret, or Application Default Credentials in a managed environment. Never put a service account private key in Flutter or commit it. Enable Firebase Authentication email/password sign-in, Firestore, and Firebase Storage if retaining uploaded images. The example includes this app's configured project ID and Storage bucket; leave `FIREBASE_STORAGE_BUCKET` empty to disable image retention.

For a local service account path, use an absolute path in `FIREBASE_SERVICE_ACCOUNT_PATH`. For production, prefer the platform's workload identity / Application Default Credentials. Firebase Admin verifies ID tokens minted by Firebase Authentication, and the UID is read only from verified token claims.

### Validator model contract

Train and export an independent TFLite model that distinguishes chilli plant/leaf images from unrelated images. Training data must include labelled chilli leaves and plants across healthy/diseased conditions and varied camera, lighting, growth-stage, and background conditions, plus negative examples such as other crops, people, animals, buildings, vehicles, and unrelated objects. Keep held-out validation images separated by plant/field source to reduce near-duplicate leakage, and tune the acceptance threshold against false acceptance of non-chilli images. Store the model outside source control and set `IMAGE_VALIDATION_MODEL_PATH`. The model must accept `[1, height, width, 3]` RGB inputs and produce either a one-output sigmoid score or class scores. Set `IMAGE_VALIDATION_POSITIVE_CLASS_INDEX` for the chilli-positive class and choose a threshold based on held-out validation data. Ensure it uses RGB values divided by 255; if it uses a different normalization, update `ChilliImageValidator` to match the recorded training pipeline. Do not enable the endpoint until non-chilli, other-crop, and chilli images have been evaluated against that model.

### Start the API

From `services/chilli_backend`:

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`GET /api/v1/health` (also available at `/health`) reports model, validator, and Firebase readiness. In local development, use `http://127.0.0.1:8000/docs`. To test from an Android emulator, Flutter's default API address is `http://10.0.2.2:8000`. For a physical phone, set `API_BASE_URL` to the reachable development host URL. Production deployments must use HTTPS and explicit trusted `ALLOWED_ORIGINS` values.

## API

All `/api/v1` endpoints other than `/predictions/classes` require `Authorization: Bearer <Firebase ID token>`. User IDs in request bodies or query parameters are never used for access decisions.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/users/me` | Get profile for verified user |
| PUT | `/api/v1/users/me` | Update permitted username |
| GET | `/api/v1/predictions/classes` | Get model class names/version |
| POST | `/api/v1/predictions/validate` | Validate image readability and chilli identity |
| POST | `/api/v1/predictions/analyze` | Validate, classify, and persist scan |
| GET | `/api/v1/results` | User's scan history with pagination and filters |
| GET | `/api/v1/results/{result_id}` | Get one user's result |
| DELETE | `/api/v1/results/{result_id}` | Delete own result and stored image |

Prediction image uploads use `multipart/form-data` with form field `file`. Result listing supports `page`, `page_size` (maximum 100), `search`, `disease`, `date=YYYY-MM-DD`, or `year`; `timezone_offset_minutes` aligns date filtering with the device's local calendar. Disease values can be labels such as `Cercospora Leaf Spot`. Firestore composite indexes are in `firestore.indexes.json`; deploy them with Firebase CLI when required.

Responses use confidence percentages (0–100). Successful scan documents are stored in `users/{verified_uid}/scans/{scan_id}` and include the canonical model label, confidence, validation status, model version, optional image path, and Firestore server timestamp. Profile documents are under `users/{verified_uid}`. If Storage is configured, image bytes are stored below `users/{uid}/scans/`; otherwise no image is retained. Scan history queries are always scoped to the verified UID.

## Flutter integration

Flutter gets a Firebase ID token from the signed-in Firebase Auth user and attaches it through `lib/core/network/api_client.dart`. Configure a deployed service with:

```sh
flutter run --dart-define=API_BASE_URL=https://your-api.example.com
```

Username login currently maps usernames to the legacy `@chilliapp.local` email alias to keep the existing sign-in screen. Do not expose Admin credentials in the Flutter app. Authentication, profile, prediction, results, and dashboard data are wired through `lib/services/`.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

Tests use fake Firebase/model services and do not require production credentials. A separate deployment smoke test should verify Firebase Auth token verification, Firestore server timestamps, deployed indexes, optional Storage write/delete, and validator quality on held-out images.

## Deployment

### Docker and Render

The included Dockerfile packages the actual TFLite classifier and labels. Build and run from this directory:

```sh
docker build -t chilli-plant-api .
docker run --rm -p 10000:10000 --env-file .env chilli-plant-api
```

For Render, connect `dulnaju73-collab/chilli_plant_backend` as a **Docker** web service. The repository root is the service root and the Dockerfile is `./Dockerfile`; `render.yaml` records the Blueprint configuration and `/api/v1/health` health-check path. Set `FIREBASE_SERVICE_ACCOUNT_JSON` as a Render secret. The current Blueprint uses `1c-2g` because TensorFlow's actual runtime memory has not yet been measured and the 512 MB starter tier is not a safe assumption; measure container peak RSS before choosing a smaller plan. The `1c-2g` plan is a paid 1 CPU / 2 GB option. The deploy is not scan-ready without a separately trained chilli validator model; `IMAGE_VALIDATION_MODEL_PATH` must point to that file and the validator must pass real held-out chilli and non-chilli images. This repository does not contain that model.

This environment does not include Docker/Podman, Firebase Admin credentials, a Render account integration, or GitHub authorization for the requested account. Therefore a container image build, live Firebase calls, and a live Render deployment cannot be claimed from this checkout. Keep one worker initially; each worker loads its own model. The in-memory rate limiter is per process; use a shared store or gateway if scaling to multiple workers. Configure Firestore indexes and CORS before rollout. Do not serve prediction traffic until `/api/v1/health` reports disease model, validator, and Firebase readiness as true and the actual model preprocessing has been verified.
