# Docker: a self-contained training image

The image carries **code + data + model weights**, so the internal server needs
no access to Docker Hub's neighbours, Hugging Face, or PyPI at run time.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked in.

## 1. Pick a base image that matches the server's driver

This is the one setting you must get right — check with `nvidia-smi` on the
server and pick a CUDA version its driver supports.

| driver on the server | `BASE_IMAGE` |
| --- | --- |
| ≥ 525 | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` *(default)* |
| ≥ 550 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| 470–525 | `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` |

```bash
ssh server nvidia-smi --query-gpu=driver_version --format=csv
```

## 2. Build

```bash
# all four backbones (default)
DOCKERHUB_USER=yourname ./docker/build_and_push.sh

# or just what you need — each model you drop is 0.3–2.3 GB off the image
DOCKERHUB_USER=yourname MODELS="BAAI/bge-m3" ./docker/build_and_push.sh

# different CUDA
DOCKERHUB_USER=yourname \
  BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  ./docker/build_and_push.sh
```

`PUSH=0` builds without pushing. `make build` / `make push` wrap the same script.

The build **fails fast** if anything is wrong: every baked checkpoint must load
offline (`docker/verify_models.py`), the test suite must pass, and all twelve
configs must resolve. Pass `VERIFY_MODELS=0` to skip the model check while
iterating.

### Size

| layer | approx. |
| --- | --- |
| CUDA PyTorch base | ~7 GB |
| Python deps | ~0.6 GB |
| Data (CSV) | 17 MB |
| bert-base-uncased | 0.44 GB |
| TinyBERT-6L | 0.25 GB |
| BGE-M3 | 2.3 GB |
| Qwen3-Embedding-0.6B | 1.2 GB |
| **total, all four** | **~12 GB** |

`docker/download_models.py` already skips ONNX exports, TF/Flax mirrors and the
duplicate `.bin` copy of a safetensors checkpoint — that alone saves several GB
on BGE-M3. To go smaller, bake fewer models via `MODELS`.

> Docker Hub's free tier allows images this size, but the first push is slow.
> If the internal server can reach a private registry, point `IMAGE` at that
> instead: `IMAGE=registry.internal/team/embedding-mrl`.

## 3. Push

`build_and_push.sh` runs `docker login` when you are not already authenticated.
For CI, use an access token rather than your password:

```bash
echo "$DOCKERHUB_TOKEN" | docker login -u yourname --password-stdin
DOCKERHUB_USER=yourname ./docker/build_and_push.sh
```

Two tags are pushed: the git short SHA and `latest`.

## 4. Run on the server

```bash
docker pull yourname/embedding-mrl:latest

docker run --rm --gpus all --shm-size=8g \
    -v "$PWD/outputs:/workspace/outputs" \
    yourname/embedding-mrl:latest \
    --config configs/mipic/bgem3.yaml
```

Anything starting with `-` goes to the training CLI, so overrides work as usual:

```bash
docker run --rm --gpus all --shm-size=8g -v "$PWD/outputs:/workspace/outputs" \
    yourname/embedding-mrl:latest \
    --config configs/mrl/bert.yaml --set train.epochs=3 --set train.batch_size=32
```

Anything else runs as a command:

```bash
docker run --rm yourname/embedding-mrl:latest pytest -q
docker run --rm -it yourname/embedding-mrl:latest bash
docker run --rm --gpus all -v "$PWD/outputs:/workspace/outputs" \
    yourname/embedding-mrl:latest ./scripts/run_all.sh mipic
```

`docker compose run --rm train --config configs/ese/bgem3.yaml` does the same
via `docker-compose.yml`.

### Long runs

```bash
docker run -d --name mipic-bgem3 --restart=unless-stopped --gpus all --shm-size=8g \
    -v "$PWD/outputs:/workspace/outputs" \
    yourname/embedding-mrl:latest --config configs/mipic/bgem3.yaml

docker logs -f mipic-bgem3
```

Results stream into the mounted `outputs/` as `results_epoch{N}.json`, so
progress is visible without attaching.

## Notes

- `--shm-size=8g` (or `--ipc=host`) is required: the DataLoader uses worker
  processes and Docker's default 64 MB of `/dev/shm` will crash them.
- The container runs as uid 1000, so files in the mounted `outputs/` are owned
  by a normal user rather than root. If your host user is not uid 1000, add
  `--user "$(id -u):$(id -g)"`.
- Selecting GPUs: `--gpus '"device=0,1"'`.
- The image never writes to the model cache, so it is safe to run several
  containers from it at once.
