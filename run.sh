#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mehup/embedding-service:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-embedding-service}"
NETWORK_NAME="${DOCKER_NETWORK:-mehup-ai}"
HOST_PORT="${HOST_PORT:-8002}"
MODEL_DIR="${MODEL_DIR:-/data/models/embedding}"
MODEL_NAME="${EMBEDDING_MODEL_NAME:-local-embedding}"
BACKEND="${EMBEDDING_BACKEND:-auto}"
DEVICE="${EMBEDDING_DEVICE:-cpu}"

docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1 || docker network create "${NETWORK_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --network "${NETWORK_NAME}" \
  -p "${HOST_PORT}:8002" \
  -v "${MODEL_DIR}:/models/default:ro" \
  -e EMBEDDING_MODEL_PATH=/models/default \
  -e EMBEDDING_MODEL_NAME="${MODEL_NAME}" \
  -e EMBEDDING_BACKEND="${BACKEND}" \
  -e EMBEDDING_DEVICE="${DEVICE}" \
  "${IMAGE_NAME}"

echo "Started ${CONTAINER_NAME}: http://127.0.0.1:${HOST_PORT}/health"
