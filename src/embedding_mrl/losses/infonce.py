"""Contrastive objectives shared by all three methods."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def info_nce(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-batch InfoNCE where the i-th query matches the i-th key.

    Args:
        query: ``[B, D]``
        key: ``[B, D]``
    Returns:
        ``(loss, logits)`` with ``logits`` of shape ``[B, B]``.
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels), logits


def matryoshka_info_nce(
    a: torch.Tensor,
    b: torch.Tensor,
    nested_dims: Sequence[int],
    temperature: float = 0.07,
    weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """InfoNCE summed over every nested prefix of the embedding.

    This is the MRL baseline objective and the ``L_MRL`` term inside MIPIC.

    Args:
        a: ``[B, D]`` first view.
        b: ``[B, D]`` second view.
        nested_dims: prefix lengths to apply the loss at; dims above ``D`` are skipped.
        weights: optional per-dimension weights (defaults to uniform 1.0).
    Returns:
        ``(total_loss, {"dim_{d}": logits})``
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"expected 2D [batch, dim] tensors, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must match, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    full_dim = a.size(1)
    dims = [d for d in nested_dims if d <= full_dim]
    if not dims:
        raise ValueError(
            f"no nested dim fits in full_dim={full_dim} (asked for {list(nested_dims)})"
        )
    if weights is not None and len(weights) != len(dims):
        raise ValueError("weights must have the same length as the usable nested_dims")

    total_loss = a.new_zeros(())
    all_logits: dict[str, torch.Tensor] = {}

    for idx, dim in enumerate(dims):
        loss, logits = info_nce(a[:, :dim], b[:, :dim], temperature=temperature)
        total_loss = total_loss + (loss if weights is None else weights[idx] * loss)
        all_logits[f"dim_{dim}"] = logits

    return total_loss, all_logits


def log_dimension_weights(num_dims: int) -> list[float]:
    """``1 / (1 + log(i + 1))`` - the progressive weighting used by ESE and CKA self-distillation."""
    import math

    return [1.0 / (1 + math.log(i + 1)) for i in range(num_dims)]
