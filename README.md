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

Chilli identity validation is a separate optional model stage. The repository currently does **not** contain a chilli-versus-other-plant validator model. By default, `IMAGE_VALIDATION_REQUIRED=true` requires a separately trained TFLite validator configured with `IMAGE_VALIDATION_MODEL_PATH`; set its chilli-positive output index and verify its preprocessing against its training code. Production temporarily sets `IMAGE_VALIDATION_REQUIRED=false`, allowing disease classification without the identity gate. Scans made in this mode are stored with `validation_status=validation_disabled`; the six-class disease model may still assign a disease label to a non-chilli image.

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

The app uses Firebase email/password authentication and attaches the signed-in user's Firebase ID token to backend requests. Do not expose Firebase Admin credentials in the Flutter app. Authentication, profile, prediction, results, and dashboard data are wired through `lib/services/`.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

Tests use fake Firebase/model services and do not require production credentials. A separate deployment smoke test should verify Firebase Auth token verification, Firestore server timestamps, deployed indexes, optional Storage write/delete, and validator quality on held-out images.

## Deployment

### AWS EC2 with GitHub Actions

`.github/workflows/deploy-ec2.yml` builds the Docker image, publishes it to GitHub Container Registry, then deploys the API and Caddy HTTPS proxy to the configured Amazon Linux 2023 instance when code is pushed to `main` (or when run manually). The API URL is `https://16.178.139.98`.

Before the first workflow run:

1. In the GitHub repository, open **Settings → Secrets and variables → Actions** and add these repository secrets:
   - `EC2_SSH_PRIVATE_KEY`: the contents of the EC2 PEM key. Keep the PEM out of this repository and chat.
   - `FIREBASE_SERVICE_ACCOUNT_JSON`: the service-account JSON for Firebase project `flutterchilli`.
2. In the EC2 security group, allow inbound TCP 80 and 443 from clients. Keep SSH TCP 22 restricted to your own IP.
3. Use an Elastic IP for the instance so the configured IP and certificate name stay stable.

For a Flutter Web client, set the optional repository variable `ALLOWED_ORIGINS` to a comma-separated list of exact browser origins, including ports, such as `https://app.example.com,http://localhost:8080`.

Caddy uses a short-lived Let's Encrypt IP certificate and renews it automatically. The health check also requires Firebase Admin and the bundled disease model to load. The production deploy currently leaves `FIREBASE_STORAGE_BUCKET` unset because the configured bucket was not found. Predictions are saved to Firestore without retaining the uploaded image until a valid bucket is configured. This instance is a `t3.micro` with about 1 GB RAM; the TensorFlow runtime has not been measured at that size, so use at least a 2 GB instance if the model container is killed or becomes unstable.

The repository contains the six-class disease model, but no separate chilli-versus-other-plant validator model. Set `IMAGE_VALIDATION_REQUIRED=false` to temporarily skip the gate, or configure `IMAGE_VALIDATION_MODEL_PATH` with a dedicated validator and set the requirement back to true.

### Docker and Render

The included Dockerfile packages the actual TFLite classifier and labels. Build and run from this directory:

```sh
docker build -t chilli-plant-api .
docker run --rm -p 10000:10000 --env-file .env chilli-plant-api
```

For Render, connect `dulnaju73-collab/chilli_plant_backend` as a **Docker** web service. The repository root is the service root and the Dockerfile is `./Dockerfile`; `render.yaml` records the Blueprint configuration and `/api/v1/health` health-check path. Set `FIREBASE_SERVICE_ACCOUNT_JSON` as a Render secret. The current Blueprint uses `1c-2g` because TensorFlow's actual runtime memory has not yet been measured and the 512 MB starter tier is not a safe assumption; measure container peak RSS before choosing a smaller plan. The `1c-2g` plan is a paid 1 CPU / 2 GB option. The production deploy temporarily sets `IMAGE_VALIDATION_REQUIRED=false` until a separate validator model is added. Set the flag to true after `IMAGE_VALIDATION_MODEL_PATH` points to a dedicated file and it passes held-out chilli and non-chilli images.

Keep one worker initially; each worker loads its own model. The in-memory rate limiter is per process; use a shared store or gateway if scaling to multiple workers. Configure Firestore indexes and CORS before rollout. When validation is enabled, require `/api/v1/health` to report disease model, validator, and Firebase readiness as true and verify the validator preprocessing.
