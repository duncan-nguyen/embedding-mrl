#!/usr/bin/env python3
"""Build-time check: every baked model must load fully offline.

Runs in the final stage so a broken or incomplete bake fails the build instead
of failing hours later on the training server.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModel, AutoTokenizer


def check(repo_id: str) -> bool:
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            repo_id, output_hidden_states=True, trust_remote_code=True
        )
        model.eval()

        batch = tokenizer(["a offline smoke test sentence"], return_tensors="pt")
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True)

        hidden = out.last_hidden_state
        print(
            f"  OK  {repo_id:42s} hidden={hidden.shape[-1]:5d} "
            f"layers={len(out.hidden_states) - 1:3d} tokenizer={type(tokenizer).__name__}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"  FAIL {repo_id}: {exc}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    models = sys.argv[1:]
    if not models:
        print("no models to verify", file=sys.stderr)
        return 1

    print("Verifying baked checkpoints load offline:", flush=True)
    ok = [check(repo_id) for repo_id in models]
    if not all(ok):
        return 1
    print(f"All {len(models)} checkpoints load offline.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
