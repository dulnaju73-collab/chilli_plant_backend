#!/usr/bin/env bash
set -Eeuo pipefail

: "${IMAGE_REF:?IMAGE_REF is required}"

APP_DIR=/opt/chilli-plant-api
NETWORK=chilli-backend
PUBLIC_HOST=16.178.139.98
ALLOWED_ORIGINS="$(printf '%s' "${ALLOWED_ORIGINS_B64:-}" | base64 -d)"
IMAGE_VALIDATION_MODEL_PATH="$(printf '%s' "${IMAGE_VALIDATION_MODEL_PATH_B64:-}" | base64 -d)"

cleanup_registry_login() {
    docker logout ghcr.io >/dev/null 2>&1 || true
}
trap cleanup_registry_login EXIT

docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
docker pull "$IMAGE_REF"

if [[ -n "$IMAGE_VALIDATION_MODEL_PATH" ]]; then
    case "$IMAGE_VALIDATION_MODEL_PATH" in
        /app/models/*.tflite) ;;
        *) echo "IMAGE_VALIDATION_MODEL_PATH must point to a .tflite file under /app/models." >&2; exit 1 ;;
    esac
    docker run --rm --entrypoint python "$IMAGE_REF" \
        -c 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if p.is_file() else f"Validator model is missing from the image: {p}")' \
        "$IMAGE_VALIDATION_MODEL_PATH"
fi

docker rm -f chilli-api >/dev/null 2>&1 || true
api_env=(
    --env PORT=8000
    --env FIREBASE_PROJECT_ID=flutterchilli
    --env FIREBASE_STORAGE_BUCKET=flutterchilli.firebasestorage.app
    --env FIREBASE_SERVICE_ACCOUNT_PATH=/run/secrets/firebase-service-account.json
    --env MODEL_PATH=/app/models/chilli_leaf_disease_final.tflite
    --env LABELS_PATH=/app/models/class_names.json
    --env IMAGE_VALIDATION_REQUIRED=false
    --env "ALLOWED_ORIGINS=$ALLOWED_ORIGINS"
)
if [[ -n "$IMAGE_VALIDATION_MODEL_PATH" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_MODEL_PATH=$IMAGE_VALIDATION_MODEL_PATH")
fi
docker run -d \
    --name chilli-api \
    --network "$NETWORK" \
    --restart unless-stopped \
    "${api_env[@]}" \
    --volume "$APP_DIR/secrets/firebase-service-account.json:/run/secrets/firebase-service-account.json:ro" \
    "$IMAGE_REF"

docker rm -f chilli-caddy >/dev/null 2>&1 || true
docker run -d \
    --name chilli-caddy \
    --network "$NETWORK" \
    --restart unless-stopped \
    --publish 80:80 \
    --publish 443:443 \
    --volume "$APP_DIR/Caddyfile:/etc/caddy/Caddyfile:ro" \
    --volume chilli-caddy-data:/data \
    --volume chilli-caddy-config:/config \
    caddy:2.11.9-alpine

for attempt in $(seq 1 30); do
    if docker exec chilli-api python -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)' \
        >/dev/null 2>&1; then
        echo "Chilli API container is responding."
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        docker logs --tail 80 chilli-api
        exit 1
    fi
    sleep 2
done

echo "Deployment started. Public endpoint: https://${PUBLIC_HOST}"
docker ps --filter name=chilli-api --filter name=chilli-caddy --format 'table {{.Names}}\t{{.Status}}'
