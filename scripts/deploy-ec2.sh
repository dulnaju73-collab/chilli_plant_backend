#!/usr/bin/env bash
set -Eeuo pipefail

: "${IMAGE_REF:?IMAGE_REF is required}"

APP_DIR=/opt/chilli-plant-api
NETWORK=chilli-backend
PUBLIC_HOST=16.179.128.119
decode_b64() {
    printf '%s' "${1:-}" | base64 -d
}
trim_ws() {
    local value="${1:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}
ALLOWED_ORIGINS="$(trim_ws "$(decode_b64 "${ALLOWED_ORIGINS_B64:-}")")"
IMAGE_VALIDATION_MODEL_PATH="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_MODEL_PATH_B64:-}")")"
IMAGE_VALIDATION_MODEL_PATH="$(trim_ws "${IMAGE_VALIDATION_MODEL_PATH#IMAGE_VALIDATION_MODEL_PATH=}")"
if [[ ${#IMAGE_VALIDATION_MODEL_PATH} -ge 2 && ( ( "$IMAGE_VALIDATION_MODEL_PATH" == \"*\" && "$IMAGE_VALIDATION_MODEL_PATH" == *\" ) || ( "$IMAGE_VALIDATION_MODEL_PATH" == \'*\' && "$IMAGE_VALIDATION_MODEL_PATH" == *\' ) ) ]]; then
    IMAGE_VALIDATION_MODEL_PATH="$(trim_ws "${IMAGE_VALIDATION_MODEL_PATH:1:${#IMAGE_VALIDATION_MODEL_PATH}-2}")"
fi
if [[ "$IMAGE_VALIDATION_MODEL_PATH" == models/*.tflite ]]; then
    IMAGE_VALIDATION_MODEL_PATH="/app/$IMAGE_VALIDATION_MODEL_PATH"
elif [[ "$IMAGE_VALIDATION_MODEL_PATH" == ./models/*.tflite ]]; then
    IMAGE_VALIDATION_MODEL_PATH="/app/${IMAGE_VALIDATION_MODEL_PATH#./}"
fi
IMAGE_VALIDATION_REQUIRED="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_REQUIRED_B64:-}")")"
IMAGE_VALIDATION_POSITIVE_CLASS_INDEX="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_POSITIVE_CLASS_INDEX_B64:-}")")"
IMAGE_VALIDATION_THRESHOLD="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_THRESHOLD_B64:-}")")"
IMAGE_VALIDATION_INVERT_SCORE="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_INVERT_SCORE_B64:-}")")"
IMAGE_VALIDATION_INPUT_SCALE="$(trim_ws "$(decode_b64 "${IMAGE_VALIDATION_INPUT_SCALE_B64:-}")")"

cleanup_registry_login() {
    docker logout ghcr.io >/dev/null 2>&1 || true
}
trap cleanup_registry_login EXIT

docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
docker pull "$IMAGE_REF"

validator_model_exists=false
if [[ -n "$IMAGE_VALIDATION_MODEL_PATH" ]]; then
    case "$IMAGE_VALIDATION_MODEL_PATH" in
        *".."*) echo "IMAGE_VALIDATION_MODEL_PATH must not contain '..'." >&2; exit 1 ;;
        /app/models/*.tflite) ;;
        *) echo "IMAGE_VALIDATION_MODEL_PATH must be a container path under /app/models/*.tflite (e.g. /app/models/leaf_detection.tflite or models/leaf_detection.tflite), not a URL or other path." >&2; exit 1 ;;
    esac
    if docker run --rm --entrypoint python "$IMAGE_REF" \
        -c 'import pathlib, sys; sys.exit(0 if pathlib.Path(sys.argv[1]).is_file() else 1)' \
        "$IMAGE_VALIDATION_MODEL_PATH"; then
        validator_model_exists=true
    fi
fi

validation_required_normalized="$(printf '%s' "$IMAGE_VALIDATION_REQUIRED" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
case "$validation_required_normalized" in
    1|true|yes|on)
        if [[ "$validator_model_exists" != "true" ]]; then
            echo "IMAGE_VALIDATION_REQUIRED is enabled, but validator model is missing from the image: ${IMAGE_VALIDATION_MODEL_PATH:-<unset>}" >&2
            exit 1
        fi
        effective_validation_required=true
        ;;
    0|false|no|off)
        effective_validation_required=false
        ;;
    "")
        if [[ "$validator_model_exists" == "true" ]]; then
            effective_validation_required=true
        else
            if [[ -n "$IMAGE_VALIDATION_MODEL_PATH" ]]; then
                echo "Warning: Validator model is not present in the image yet (${IMAGE_VALIDATION_MODEL_PATH}); deploying with IMAGE_VALIDATION_REQUIRED=false until the model file is added." >&2
            fi
            effective_validation_required=false
        fi
        ;;
    *)
        echo "Unsupported IMAGE_VALIDATION_REQUIRED value: $IMAGE_VALIDATION_REQUIRED" >&2
        exit 1
        ;;
esac

docker rm -f chilli-api >/dev/null 2>&1 || true
api_env=(
    --env PORT=8000
    --env FIREBASE_PROJECT_ID=flutterchilli
    --env FIREBASE_SERVICE_ACCOUNT_PATH=/run/secrets/firebase-service-account.json
    --env MODEL_PATH=/app/models/chilli_leaf_disease_final.tflite
    --env LABELS_PATH=/app/models/class_names.json
    --env "IMAGE_VALIDATION_REQUIRED=$effective_validation_required"
    --env "ALLOWED_ORIGINS=$ALLOWED_ORIGINS"
)
if [[ "$validator_model_exists" == "true" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_MODEL_PATH=$IMAGE_VALIDATION_MODEL_PATH")
fi
if [[ -n "$IMAGE_VALIDATION_POSITIVE_CLASS_INDEX" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_POSITIVE_CLASS_INDEX=$IMAGE_VALIDATION_POSITIVE_CLASS_INDEX")
fi
if [[ -n "$IMAGE_VALIDATION_THRESHOLD" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_THRESHOLD=$IMAGE_VALIDATION_THRESHOLD")
fi
if [[ -n "$IMAGE_VALIDATION_INVERT_SCORE" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_INVERT_SCORE=$IMAGE_VALIDATION_INVERT_SCORE")
fi
if [[ -n "$IMAGE_VALIDATION_INPUT_SCALE" ]]; then
    api_env+=(--env "IMAGE_VALIDATION_INPUT_SCALE=$IMAGE_VALIDATION_INPUT_SCALE")
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
