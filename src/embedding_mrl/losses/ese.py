"""ESE / EPRESSO baseline: Matryoshka InfoNCE over dimensions *and* layers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..pooling import mean_pooling
from .infonce import log_dimension_weights


def epresso_simcse(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    matryoshka_dims: Sequence[int],
    temperature: float = 0.07,
    matryoshka_weights: Sequence[float] | None = None,
    use_layer_weight: bool = True,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Weighted InfoNCE across nested dimensions for one pair of pooled views.

    Args:
        emb1: ``[B, D]`` first view.
        emb2: ``[B, D]`` second view (the in-batch positive).
        matryoshka_dims: nested prefix lengths.
        matryoshka_weights: per-dimension weights; defaults to ``1/(1+log(i+1))``.
        use_layer_weight: when ``False`` every dimension is weighted 1.0.
    Returns:
        ``(total_loss, loss_dict, acc_dict)`` - the dicts hold plain floats for logging.
    """
    if emb1.dim() != 2 or emb2.dim() != 2:
        raise ValueError(
            f"expected 2D tensors, got {tuple(emb1.shape)} and {tuple(emb2.shape)}"
        )
    if emb1.shape != emb2.shape:
        raise ValueError(
            f"emb1/emb2 mismatch: {tuple(emb1.shape)} vs {tuple(emb2.shape)}"
        )

    batch_size, full_dim = emb1.shape
    device = emb1.device

    valid_dims = [d for d in matryoshka_dims if d <= full_dim]
    if not valid_dims:
        raise ValueError(f"no matryoshka dim fits full_dim={full_dim}")
    if matryoshka_weights is None:
        matryoshka_weights = log_dimension_weights(len(valid_dims))

    total_loss = emb1.new_zeros(())
    loss_dict: dict[str, float] = {}
    acc_dict: dict[str, float] = {}
    labels = torch.arange(batch_size, device=device)

    for idx, dim in enumerate(valid_dims):
        weight = matryoshka_weights[idx] if use_layer_weight else 1.0

        query = F.normalize(emb1[:, :dim], p=2, dim=-1)
        key = F.normalize(emb2[:, :dim], p=2, dim=-1)
        logits = torch.matmul(query, key.T) / temperature

        loss = F.cross_entropy(logits, labels)
        weighted = weight * loss
        total_loss = total_loss + weighted

        loss_dict[f"loss_dim_{dim}"] = loss.item()
        loss_dict[f"weighted_loss_dim_{dim}"] = weighted.item()
        with torch.no_grad():
            acc_dict[f"acc_dim_{dim}"] = (
                (logits.argmax(dim=-1) == labels).float().mean().item()
            )

    return total_loss, loss_dict, acc_dict


def epresso_simcse_from_hidden_states(
    hidden_states1: Sequence[torch.Tensor],
    hidden_states2: Sequence[torch.Tensor],
    attention_mask1: torch.Tensor,
    attention_mask2: torch.Tensor,
    matryoshka_dims: Sequence[int],
    temperature: float = 0.07,
    n_layers_per_step: int = 1,
    use_intermediate_layers: bool = True,
    use_layer_weight: bool = True,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Full 2D (dimension x layer) EPRESSO loss from pre-computed hidden states.

    Taking hidden states as an argument - rather than re-running the encoder as
    the notebook did - keeps the objective identical while removing a duplicate
    forward pass per step.

    Args:
        hidden_states1/2: the encoder's ``hidden_states`` tuples for both views.
        n_layers_per_step: how many intermediate layers to sample each step
            (``<= 0`` uses all of them).
    Returns:
        ``(total_loss, final_layer_loss_dict, final_layer_acc_dict)``
    """
    num_layers = len(hidden_states1)
    if num_layers != len(hidden_states2):
        raise ValueError("both views must expose the same number of hidden states")

    final_emb1 = mean_pooling(hidden_states1[-1], attention_mask1)
    final_emb2 = mean_pooling(hidden_states2[-1], attention_mask2)

    total_loss, final_loss_dict, final_acc_dict = epresso_simcse(
        final_emb1,
        final_emb2,
        matryoshka_dims=matryoshka_dims,
        temperature=temperature,
        use_layer_weight=use_layer_weight,
    )

    if use_intermediate_layers and num_layers > 2:
        # Skip the embedding layer (0) and the final layer (-1).
        layer_indices = list(range(1, num_layers - 1))
        if 0 < n_layers_per_step < len(layer_indices):
            layer_indices = random.sample(layer_indices, n_layers_per_step)

        for layer_idx in layer_indices:
            layer_emb1 = mean_pooling(hidden_states1[layer_idx], attention_mask1)
            layer_emb2 = mean_pooling(hidden_states2[layer_idx], attention_mask2)
            layer_loss, _, _ = epresso_simcse(
                layer_emb1,
                layer_emb2,
                matryoshka_dims=matryoshka_dims,
                temperature=temperature,
                use_layer_weight=use_layer_weight,
            )
            # Deeper layers count for more than shallow ones.
            layer_weight = 1.0 / (1 + math.log(num_layers - layer_idx))
            total_loss = total_loss + layer_weight * layer_loss

    return total_loss, final_loss_dict, final_acc_dict


def epresso_simcse_with_layers(
    model,
    input_ids1: torch.Tensor,
    attention_mask1: torch.Tensor,
    input_ids2: torch.Tensor,
    attention_mask2: torch.Tensor,
    matryoshka_dims: Sequence[int],
    temperature: float = 0.07,
    n_layers_per_step: int = 1,
    use_intermediate_layers: bool = True,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Convenience wrapper that runs the encoder itself (notebook-compatible signature)."""
    outputs1 = model(
        input_ids=input_ids1,
        attention_mask=attention_mask1,
        output_hidden_states=True,
        return_dict=True,
    )
    outputs2 = model(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        output_hidden_states=True,
        return_dict=True,
    )
    return epresso_simcse_from_hidden_states(
        outputs1.hidden_states,
        outputs2.hidden_states,
        attention_mask1,
        attention_mask2,
        matryoshka_dims=matryoshka_dims,
        temperature=temperature,
        n_layers_per_step=n_layers_per_step,
        use_intermediate_layers=use_intermediate_layers,
    )
