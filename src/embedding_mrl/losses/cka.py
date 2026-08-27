"""Centered Kernel Alignment losses.

Two variants exist because the notebooks used two:

* :class:`CKALoss` - flattens ``[B, L, D]`` into one big ``[B*L, D]`` matrix and
  computes a single CKA (the MRL notebook's self-distillation helper).
* :class:`PerExampleCKALoss` - computes CKA per example over its ``k`` selected
  tokens and averages, which is what MIPIC's submatrix alignment needs.
"""

from __future__ import annotations

import random

import torch
from torch import nn

from .infonce import log_dimension_weights


class CKALoss(nn.Module):
    """Batch-level CKA distance ``1 - CKA(SH, TH)``.

    Inputs are flattened to 2D and computed in float64 for numerical stability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        d_student = student_hidden.size(-1)
        d_teacher = teacher_hidden.size(-1)

        sh = student_hidden.reshape(-1, d_student).to(torch.float64)
        th = teacher_hidden.reshape(-1, d_teacher).to(sh.device, torch.float64)

        sh = sh - sh.mean(0, keepdim=True)
        th = th - th.mean(0, keepdim=True)

        numerator = torch.norm(sh.t().matmul(th), "fro")
        den_s = torch.norm(sh.t().matmul(sh), "fro") + self.eps
        den_t = torch.norm(th.t().matmul(th), "fro") + self.eps

        cka_sim = numerator / torch.sqrt(den_s * den_t)
        return (1.0 - cka_sim).float()


class PerExampleCKALoss(nn.Module):
    """CKA computed per example (tokens are the samples), then averaged over the batch.

    Accepts ``[B, k, D]`` or ``[k, D]``.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def compute_cka_single(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CKA similarity for one example. ``x``: ``[n, p1]``, ``y``: ``[n, p2]``."""
        x = x.to(torch.float64)
        y = y.to(torch.float64)

        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)

        xty = x.t().matmul(y)
        xtx = x.t().matmul(x)
        yty = y.t().matmul(y)

        numerator = torch.norm(xty, "fro") ** 2
        denominator = torch.norm(xtx, "fro") * torch.norm(yty, "fro") + self.eps
        return numerator / denominator

    def forward(
        self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        if student_hidden.dim() == 3:
            sims = torch.stack(
                [
                    self.compute_cka_single(student_hidden[i], teacher_hidden[i])
                    for i in range(student_hidden.size(0))
                ]
            )
            cka_sim = sims.mean()
        elif student_hidden.dim() == 2:
            cka_sim = self.compute_cka_single(student_hidden, teacher_hidden)
        else:
            raise ValueError(
                f"expected a 2D or 3D tensor, got shape {tuple(student_hidden.shape)}"
            )

        return 1.0 - cka_sim.float()


class MatryoshkaCKASelfDistiller(nn.Module):
    """Every truncated prefix is pulled towards the full-width representation.

    Kept from the MRL notebook where it was defined but commented out of the
    training step; useful as an ablation (``losses.cka.MatryoshkaCKASelfDistiller``).
    """

    def __init__(
        self,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float] | None = None,
        n_dims_per_step: int = -1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.matryoshka_dims = sorted(matryoshka_dims, reverse=True)
        self.matryoshka_weights = matryoshka_weights or log_dimension_weights(
            len(matryoshka_dims)
        )
        if len(self.matryoshka_weights) != len(self.matryoshka_dims):
            raise ValueError("matryoshka_weights must match matryoshka_dims in length")
        self.n_dims_per_step = n_dims_per_step
        self.cka = CKALoss(eps=eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """``hidden_state``: ``[B, L, D]`` or ``[B, D]``. Returns the averaged CKA distance."""
        teacher = hidden_state[..., : self.matryoshka_dims[0]]

        # Index 0 is the teacher (full width), so only the smaller dims are students.
        indices = list(range(1, len(self.matryoshka_dims)))
        if 0 < self.n_dims_per_step < len(indices):
            indices = sorted(random.sample(indices, self.n_dims_per_step))

        total = hidden_state.new_zeros(())
        count = 0
        for idx in indices:
            dim = self.matryoshka_dims[idx]
            if dim > hidden_state.size(-1):
                continue
            total = total + self.matryoshka_weights[idx] * self.cka(
                hidden_state[..., :dim], teacher
            )
            count += 1

        return total / count if count else total
