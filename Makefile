# Thin wrappers around the Docker workflow. Override anything on the command
# line, e.g. `make build MODELS="BAAI/bge-m3"`.

DOCKERHUB_USER ?= $(USER)
IMAGE          ?= $(DOCKERHUB_USER)/embedding-mrl
TAG            ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
CONFIG         ?= configs/mipic/bgem3.yaml
GPUS           ?= all

.PHONY: help test build build-fast push run shell run-all clean

help:
	@echo "test       run the offline test suite on the host"
	@echo "build      build the image locally (models + data baked in)"
	@echo "build-fast build without the model-loading verification step"
	@echo "push       build and push to Docker Hub as $(IMAGE):$(TAG)"
	@echo "run        run one experiment in the container (CONFIG=$(CONFIG))"
	@echo "run-all    run all 12 experiments in the container"
	@echo "shell      open a shell inside the image"
	@echo ""
	@echo "IMAGE=$(IMAGE)  TAG=$(TAG)"

test:
	pytest -q

build:
	DOCKERHUB_USER=$(DOCKERHUB_USER) IMAGE=$(IMAGE) TAG=$(TAG) PUSH=0 ./docker/build_and_push.sh

build-fast:
	DOCKERHUB_USER=$(DOCKERHUB_USER) IMAGE=$(IMAGE) TAG=$(TAG) PUSH=0 VERIFY_MODELS=0 ./docker/build_and_push.sh

push:
	DOCKERHUB_USER=$(DOCKERHUB_USER) IMAGE=$(IMAGE) TAG=$(TAG) PUSH=1 ./docker/build_and_push.sh

run:
	docker run --rm --gpus $(GPUS) --shm-size=8g \
		-v "$(PWD)/outputs:/workspace/outputs" \
		$(IMAGE):$(TAG) --config $(CONFIG)

run-all:
	docker run --rm --gpus $(GPUS) --shm-size=8g \
		-v "$(PWD)/outputs:/workspace/outputs" \
		$(IMAGE):$(TAG) ./scripts/run_all.sh

shell:
	docker run --rm -it --gpus $(GPUS) --shm-size=8g \
		-v "$(PWD)/outputs:/workspace/outputs" \
		$(IMAGE):$(TAG) bash

clean:
	rm -rf outputs .pytest_cache **/__pycache__
