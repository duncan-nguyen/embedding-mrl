"""Sequence -> vector pooling.

The original notebooks were inconsistent here: MRL and MIPIC read the CLS token
(``last_hidden_state[:, 0, :]``) while ESE mean-pooled over the attention mask.
Both are available and selected via ``model.pooling`` in the config so each
method reproduces its notebook exactly.
"""

from __future__ import annotations

import torch


def mean_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Attention-masked average over the sequence axis.

    Args:
        hidden_state: ``[B, L, D]``
        attention_mask: ``[B, L]``
    Returns:
        ``[B, D]``
    """
    # A float32 mask promotes the reduction out of fp16 under autocast; summing
    # hundreds of tokens in fp16 would lose precision.
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def cls_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """First-token ("[CLS]") representation. ``attention_mask`` is unused."""
    del attention_mask
    return hidden_state[:, 0, :]


def last_token_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Last non-padding token - the correct choice for causal encoders (Qwen3)."""
    lengths = attention_mask.sum(dim=1).long() - 1
    lengths = lengths.clamp(min=0)
    index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_state.size(-1))
    return hidden_state.gather(1, index).squeeze(1)


_POOLERS = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "last": last_token_pooling,
}


def get_pooler(name: str):
    try:
        return _POOLERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown pooling {name!r}; expected one of {sorted(_POOLERS)}"
        ) from exc


def pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor, mode: str
) -> torch.Tensor:
    return get_pooler(mode)(hidden_state, attention_mask)
