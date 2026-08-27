#!/usr/bin/env python3
"""Bake HF checkpoints into the image at build time.

Downloads only what ``AutoModel`` / ``AutoTokenizer`` actually need: config,
tokenizer files and one set of weights. ONNX exports, TF/Flax mirrors and the
duplicate ``.bin`` copy of a safetensors checkpoint are skipped, which cuts
several GB off the image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Formats and extras we never load from Python.
IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "onnx/**",
    "openvino/**",
    "coreml/**",
    "*.mlmodel",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "imgs/**",
    "assets/**",
    # BGE-M3 ships extra ColBERT/SPLADE heads that AutoModel does not use.
    "colbert_linear.pt",
    "sparse_linear.pt",
    "1_Pooling/**",
    "2_Dense/**",
]


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_duplicate_weights(snapshot: Path) -> int:
    """Drop ``pytorch_model*.bin`` when an equivalent safetensors file exists."""
    if not any(snapshot.rglob("*.safetensors")):
        return 0

    freed = 0
    for stale in list(snapshot.rglob("pytorch_model*.bin")) + list(
        snapshot.rglob("pytorch_model.bin.index.json")
    ):
        target = stale.resolve()  # snapshots/ entries are symlinks into blobs/
        size = target.stat().st_size if target.exists() else 0
        stale.unlink(missing_ok=True)
        if target.exists() and target != stale:
            target.unlink(missing_ok=True)
        freed += size
    return freed


def fetch(repo_id: str, token: str | None) -> None:
    print(f"==> {repo_id}", flush=True)
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            ignore_patterns=IGNORE_PATTERNS,
            token=token or None,
            max_workers=4,
        )
    )
    freed = prune_duplicate_weights(path)
    note = f" (pruned {human(freed)} of duplicate .bin weights)" if freed else ""
    print(f"    {human(directory_size(path))}{note}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="HF repo ids to bake into the image")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Using an authenticated HF session.", flush=True)

    for repo_id in args.models:
        try:
            fetch(repo_id, token)
        except Exception as exc:  # fail the build loudly, not silently
            print(f"FAILED to download {repo_id}: {exc}", file=sys.stderr)
            return 1

    cache = Path(os.environ.get("HF_HOME", "/opt/hf"))
    print(f"\nTotal cache: {human(directory_size(cache))} at {cache}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
