"""Geometric Successive Refinement loss primitives.

The teacher supplies corpus-level spectral coordinates.  The student supplies
ordinary coordinate bands.  Both are compared through additive squared-distance
increments over unordered pairs, so the optimisation batch size never limits
the number of meaningful spectral shells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


Shell = tuple[int, int]


@dataclass
class GSRShellLossOutput:
    """Tensor-valued loss output retained for diagnostics and autograd probes."""

    total_loss: torch.Tensor
    shell_losses: Dict[str, torch.Tensor]
    shell_weights: Dict[str, float]
    majorized_shell_losses: Dict[str, torch.Tensor]
    student_distances: Dict[str, torch.Tensor]
    teacher_distances: Dict[str, torch.Tensor]


def shell_key(shell: Shell) -> str:
    start, end = shell
    return f"dim_{start}_{end}"


def full_normalize(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalise once at full width while keeping zero rows finite."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"expected [batch, dim] embeddings, got {tuple(embeddings.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    norms = embeddings.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    return embeddings.float() / norms


def condensed_squared_distances(points: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances for all unordered pairs in torch.pdist order."""
    if points.ndim != 2:
        raise ValueError(f"expected [batch, dim] points, got {tuple(points.shape)}")
    if points.size(0) < 2:
        raise ValueError("GSR needs at least two examples to form a pair")
    if points.size(1) < 1:
        raise ValueError("a GSR shell must contain at least one coordinate")
    return torch.pdist(points.float(), p=2).square()


def build_shell_slices(dims: Sequence[int], full_dim: int | None = None) -> list[Shell]:
    """Convert increasing prefix endpoints into non-empty coordinate bands."""
    endpoints = [int(dim) for dim in dims]
    if not endpoints:
        raise ValueError("GSR needs at least one geometry dimension")
    if any(dim <= 0 for dim in endpoints):
        raise ValueError(f"geometry dimensions must be positive, got {endpoints}")
    if endpoints != sorted(set(endpoints)):
        raise ValueError(
            f"geometry dimensions must be strictly increasing, got {endpoints}"
        )
    if full_dim is not None and endpoints[-1] != full_dim:
        raise ValueError(
            f"largest geometry dimension must equal full_dim={full_dim}, "
            f"got {endpoints[-1]}"
        )
    boundaries = [0, *endpoints]
    return list(zip(boundaries[:-1], boundaries[1:]))


def prefix_risk_majorizer_weights(num_shells: int) -> list[float]:
    """Return the diagonal majorizer weights for cumulative prefix risk.

    If ``e_j`` is the residual-distance error of shell ``j``, then the error at
    prefix ``k`` is ``sum_{j <= k} e_j``.  Applying Cauchy--Schwarz to every
    prefix and summing over all supported geometry prefixes gives

    ``sum_k (sum_{j <= k} e_j)^2 <= sum_j beta_j e_j^2``,

    where ``beta_j = sum_{k=j}^K k``.  These weights therefore follow from the
    prefix-risk bound rather than from empirical loss or gradient scaling.
    """
    if not isinstance(num_shells, int) or isinstance(num_shells, bool):
        raise TypeError(f"num_shells must be an integer, got {num_shells!r}")
    if num_shells < 1:
        raise ValueError(f"num_shells must be positive, got {num_shells}")
    total = num_shells * (num_shells + 1) // 2
    return [float(total - (j - 1) * j // 2) for j in range(1, num_shells + 1)]


def merge_tied_shells(
    dims: Sequence[int],
    eigenvalues: torch.Tensor,
    tolerance: float,
    eps: float = 1e-12,
) -> tuple[list[Shell], list[int]]:
    """Merge geometry bands when an eigenvalue tie crosses their boundary.

    Task prefixes remain unchanged.  Only the geometry shell on either side of
    an unresolved boundary is joined.
    """
    shells = build_shell_slices(dims, full_dim=eigenvalues.numel())
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    endpoints = [end for _, end in shells]
    kept = [0]
    merged: list[int] = []
    for boundary in endpoints[:-1]:
        left = float(eigenvalues[boundary - 1])
        right = float(eigenvalues[boundary])
        relative_gap = (left - right) / max(abs(left), eps)
        if relative_gap <= tolerance:
            merged.append(boundary)
        else:
            kept.append(boundary)
    kept.append(endpoints[-1])
    return list(zip(kept[:-1], kept[1:])), merged


def gsr_shell_loss(
    student: torch.Tensor,
    teacher_scores: torch.Tensor,
    shells: Sequence[Shell],
    c_teacher: float | torch.Tensor,
    eps: float = 1e-8,
) -> GSRShellLossOutput:
    """Unbiased U-statistic for the prefix-risk-majorized GSR objective."""
    if student.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError("student and teacher_scores must both be [batch, dim]")
    if student.shape != teacher_scores.shape:
        raise ValueError(
            f"student and teacher_scores must match, got {tuple(student.shape)} "
            f"and {tuple(teacher_scores.shape)}"
        )
    if not shells:
        raise ValueError("GSR needs at least one shell")
    if shells[0][0] != 0 or shells[-1][1] != student.size(1):
        raise ValueError(
            f"shells must cover [0, {student.size(1)}), got {list(shells)}"
        )
    for previous, current in zip(shells, shells[1:]):
        if previous[1] != current[0]:
            raise ValueError(f"shells must be contiguous, got {list(shells)}")

    denominator = torch.as_tensor(
        c_teacher, dtype=torch.float32, device=student.device
    ).detach()
    if not torch.isfinite(denominator) or denominator.item() <= eps:
        raise ValueError(
            f"c_teacher must be finite and greater than eps={eps}, "
            f"got {denominator.item()}"
        )

    teacher_scores = teacher_scores.detach().float()
    student = student.float()
    shell_losses: Dict[str, torch.Tensor] = {}
    shell_weights: Dict[str, float] = {}
    majorized_shell_losses: Dict[str, torch.Tensor] = {}
    student_distances: Dict[str, torch.Tensor] = {}
    teacher_distances: Dict[str, torch.Tensor] = {}
    majorizer_weights = prefix_risk_majorizer_weights(len(shells))

    for (start, end), weight in zip(shells, majorizer_weights):
        if not 0 <= start < end <= student.size(1):
            raise ValueError(f"invalid shell {(start, end)} for dim {student.size(1)}")
        key = shell_key((start, end))
        student_pair = condensed_squared_distances(student[:, start:end])
        teacher_pair = condensed_squared_distances(teacher_scores[:, start:end])
        student_distances[key] = student_pair
        teacher_distances[key] = teacher_pair
        shell_loss = (student_pair - teacher_pair).square().mean() / (
            denominator + eps
        )
        shell_losses[key] = shell_loss
        shell_weights[key] = weight
        majorized_shell_losses[key] = weight * shell_loss

    total = torch.stack(list(majorized_shell_losses.values())).sum()
    return GSRShellLossOutput(
        total_loss=total,
        shell_losses=shell_losses,
        shell_weights=shell_weights,
        majorized_shell_losses=majorized_shell_losses,
        student_distances=student_distances,
        teacher_distances=teacher_distances,
    )
