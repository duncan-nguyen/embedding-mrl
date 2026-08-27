#!/usr/bin/env bash
#
# Build the self-contained training image and push it to Docker Hub.
#
#   DOCKERHUB_USER=myname ./docker/build_and_push.sh
#   DOCKERHUB_USER=myname TAG=v1 MODELS="BAAI/bge-m3" ./docker/build_and_push.sh
#   PUSH=0 ./docker/build_and_push.sh          # build locally, do not push
#
set -euo pipefail

cd "$(dirname "$0")/.."

: "${DOCKERHUB_USER:?set DOCKERHUB_USER to your Docker Hub account, e.g. DOCKERHUB_USER=myname $0}"

IMAGE="${IMAGE:-${DOCKERHUB_USER}/embedding-mrl}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
PUSH="${PUSH:-1}"
PLATFORM="${PLATFORM:-linux/amd64}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}"
MODELS="${MODELS:-google-bert/bert-base-uncased huawei-noah/TinyBERT_General_6L_768D BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B}"
VERIFY_MODELS="${VERIFY_MODELS:-1}"

echo "image     : ${IMAGE}:${TAG}"
echo "base      : ${BASE_IMAGE}"
echo "platform  : ${PLATFORM}"
echo "models    : ${MODELS}"
echo "push      : ${PUSH}"
echo

args=(
    build
    --platform "${PLATFORM}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "MODELS=${MODELS}"
    --build-arg "VERIFY_MODELS=${VERIFY_MODELS}"
    --build-arg "BUILD_REF=${TAG}"
    -t "${IMAGE}:${TAG}"
    -t "${IMAGE}:latest"
)

# Only needed for gated repos.
if [ -n "${HF_TOKEN:-}" ]; then
    args+=(--secret "id=hf_token,env=HF_TOKEN")
fi

if [ "${PUSH}" = "1" ]; then
    if ! docker system info 2>/dev/null | grep -q 'Username:'; then
        echo "Not logged in to Docker Hub - running 'docker login' first."
        docker login
    fi
    args+=(--push)
else
    args+=(--load)
fi

# buildx honours --platform and --push; fall back to the classic builder if absent.
if docker buildx version >/dev/null 2>&1; then
    docker buildx "${args[@]}" .
else
    echo "docker buildx not found; using the classic builder (no cross-platform build)"
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "MODELS=${MODELS}" \
        --build-arg "VERIFY_MODELS=${VERIFY_MODELS}" \
        --build-arg "BUILD_REF=${TAG}" \
        -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .
    if [ "${PUSH}" = "1" ]; then
        docker push "${IMAGE}:${TAG}"
        docker push "${IMAGE}:latest"
    fi
fi

echo
docker image inspect "${IMAGE}:${TAG}" --format 'built {{.RepoTags}}  size {{div .Size 1048576}} MB' 2>/dev/null || true
echo "Run it with:"
echo "  docker run --rm --gpus all -v \"\$PWD/outputs:/workspace/outputs\" ${IMAGE}:${TAG} --config configs/mipic/bgem3.yaml"
