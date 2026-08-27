"""Small shared helpers: seeding, device selection, AMP shims, logging."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("embedding_mrl")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Hub/HTTP chatter drowns out the training log at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Hub/HTTP chatter drowns out the training log at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch, matching the notebooks' behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    LOGGER.warning("No GPU available -> running on CPU (this will be very slow).")
    return torch.device("cpu")


def resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name!r}")
    return dtype


def autocast(enabled: bool, device: torch.device, dtype: torch.dtype = torch.float16):
    """`torch.amp.autocast` with a fallback for older torch releases."""
    device_type = device.type
    enabled = enabled and device_type == "cuda"
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.0
        return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def make_grad_scaler(enabled: bool, device: torch.device):
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_memory_info() -> dict[str, str]:
    """Per-GPU ``allocated/reserved`` in MB, for the tqdm postfix."""
    info: dict[str, str] = {}
    for dev_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(dev_id) / 1024**2
        reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
        info[f"gpu{dev_id}"] = f"{allocated:.0f}/{reserved:.0f}MB"
    return info


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
