#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-mehup/embedding-service:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-embedding-service}"
BUILD_IMAGE="${BUILD_IMAGE:-true}"
RUN_CONTAINER="${RUN_CONTAINER:-true}"
DOCKER_NETWORK="${DOCKER_NETWORK:-}"
HOST_PORT="${HOST_PORT:-8002}"
MODEL_DIR="${MODEL_DIR:-./models/piccolo-large-zh}"
MODEL_NAME="${EMBEDDING_MODEL_NAME:-local-embedding}"
BACKEND="${EMBEDDING_BACKEND:-auto}"
DEVICE="${EMBEDDING_DEVICE:-cpu}"

if [[ "${BUILD_IMAGE}" != "false" ]]; then
  docker build -t "${IMAGE_NAME}" .
  echo "Built ${IMAGE_NAME}"
fi

if [[ "${RUN_CONTAINER}" == "false" ]]; then
  exit 0
fi

DOCKER_ARGS=()
if [[ -n "${DOCKER_NETWORK}" ]]; then
  docker network inspect "${DOCKER_NETWORK}" >/dev/null 2>&1 || docker network create "${DOCKER_NETWORK}"
  DOCKER_ARGS+=(--network "${DOCKER_NETWORK}")
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR not found: ${MODEL_DIR}" >&2
  echo "Put the embedding model under ./models/piccolo-large-zh or pass MODEL_DIR=/path/to/model." >&2
  exit 1
fi
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  "${DOCKER_ARGS[@]}" \
  -p "${HOST_PORT}:8002" \
  -v "${MODEL_DIR}:/models/default:ro" \
  -e EMBEDDING_MODEL_PATH=/models/default \
  -e EMBEDDING_MODEL_NAME="${MODEL_NAME}" \
  -e EMBEDDING_BACKEND="${BACKEND}" \
  -e EMBEDDING_DEVICE="${DEVICE}" \
  "${IMAGE_NAME}"

echo "Started ${CONTAINER_NAME}: http://127.0.0.1:${HOST_PORT}/health"
