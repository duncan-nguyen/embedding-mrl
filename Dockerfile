# syntax=docker/dockerfile:1.7
#
# Self-contained training image: code + data + model weights, runnable with no
# network access on the internal GPU server.
#
#   docker build -t <user>/embedding-mrl:latest .
#   docker run --gpus all -v "$PWD/outputs:/workspace/outputs" \
#       <user>/embedding-mrl:latest --config configs/mipic/bgem3.yaml
#
# Pick a base whose CUDA matches the server's driver (see docker/README.md).
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# --------------------------------------------------------------------------- #
# Stage 1 - download the checkpoints (needs network; the runtime stage does not)
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS models

ARG MODELS="google-bert/bert-base-uncased huawei-noah/TinyBERT_General_6L_768D BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B"
ENV HF_HOME=/opt/hf \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "huggingface_hub>=0.23"

COPY docker/download_models.py /tmp/download_models.py

# A token is only needed for gated repos; the default four are public.
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python /tmp/download_models.py ${MODELS}

# --------------------------------------------------------------------------- #
# Stage 2 - runtime
# --------------------------------------------------------------------------- #
FROM ${BASE_IMAGE} AS runtime

ARG MODELS="google-bert/bert-base-uncased huawei-noah/TinyBERT_General_6L_768D BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B"
ARG VERIFY_MODELS=1
ARG BUILD_REF=unknown

LABEL org.opencontainers.image.title="embedding-mrl" \
      org.opencontainers.image.description="Matryoshka embedding training (MRL / ESE / MIPIC) with data and models baked in" \
      org.opencontainers.image.source="https://github.com/duncan-nguyen/embedding-mrl" \
      org.opencontainers.image.revision="${BUILD_REF}" \
      ai.embedding-mrl.models="${MODELS}"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Everything except torch - the base image already carries the CUDA build and a
# PyPI wheel would silently downgrade it to the CPU one.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt

# Package metadata first so the layer caches independently of the source.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docker/entrypoint.sh docker/verify_models.py /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# ~17 MB of CSVs: the SimCSE training corpus and every evaluation set.
COPY data/ ./data/

COPY --from=models /opt/hf /opt/hf

# Fail the build here rather than on the training server hours later.
RUN if [ "${VERIFY_MODELS}" = "1" ]; then python /usr/local/bin/verify_models.py ${MODELS}; fi

# Config validation + the offline unit suite, as a build-time self-check.
RUN python -m pytest tests/ -q -p no:cacheprovider && \
    for cfg in configs/*/*.yaml; do embedding-mrl --config "$cfg" --print-config > /dev/null; done && \
    echo "all 16 configs resolve"

# uid 1000 keeps files written into a mounted outputs/ owned by a normal user.
RUN if ! getent group 1000 >/dev/null; then groupadd --gid 1000 mrl; fi && \
    if ! getent passwd 1000 >/dev/null; then useradd --uid 1000 --gid 1000 --create-home mrl; fi && \
    mkdir -p /workspace/outputs && \
    find /workspace -name '__pycache__' -prune -exec rm -rf {} + && \
    chown -R 1000:1000 /workspace
USER 1000:1000

VOLUME ["/workspace/outputs"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
