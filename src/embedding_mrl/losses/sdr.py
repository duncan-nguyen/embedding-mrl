"""SDR-MRL: semantic distortion-rate objectives for Matryoshka representations.

Equation numbers refer to *Beyond Representation Alignment: Semantic
Distortion-Rate Learning for Matryoshka Representations* (``docs/SDR-MRL.pdf``).

The method treats every Matryoshka prefix ``Z_k`` as a representation available
at one deployment rate ``R_k = b d_k`` (Eq 18) and asks how much of the
*semantic neighborhood* induced by the full-dimensional teacher is still
recoverable from that prefix::

    p_T(S = j | x_i) = softmax_j cos(z_i^(K), z_j^(K)) / tau_T      (Eq 22-23)
    q_k(S = j | z_i) = softmax_j cos(z_i^(k), z_j^(k)) / tau_S      (Eq 24-25)
    D_k              = E_X KL(p_T(S|X) || q_k(S|Z_k))               (Eq 26)

Proposition 1 (Eq 30) turns ``D_k`` into a variational upper bound on the
conditional semantic information ``I_T(S; X | Z_k)``, so minimising it across a
deployment-rate prior ``pi`` (Eq 48) is the entire training signal::

    L_SDR = L_task + lambda_sem * sum_k pi_k D_k
                   + lambda_mono * sum_k [D_k - D_{k-1}]_+          (Eq 55)

Everything the ablation table (Sec 8) asks to vary is a constructor argument:
the divergence (A4), the geometry objective (A5), the candidate neighborhood
(A6), the rate prior (A7), full-prefix versus sampled-rate training (A8) and the
temperatures (A9).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CANDIDATE_MODES, DIVERGENCES, GEOMETRIES, RATE_PRIORS
from .cka import CKALoss


# --------------------------------------------------------------------------- #
# Deployment-rate prior (Eq 47-50)
# --------------------------------------------------------------------------- #
def rate_prior(
    prefix_dims: Sequence[int],
    kind: str = "uniform",
    weights: Sequence[float] | None = None,
) -> list[float]:
    """``pi_k = P(R = R_k)`` over the *truncated* prefixes ``d_1 < ... < d_{K-1}``.

    Args:
        prefix_dims: the nested dims the semantic loss is applied at, i.e. every
            dim except the full width (Eq 48 sums to ``K - 1``).
        kind: ``"uniform"`` (Eq 49), ``"inverse_dim"`` (Eq 50, low-rate focused)
            or ``"custom"``.
        weights: unnormalised weights, required when ``kind == "custom"``.
    Returns:
        A probability vector of the same length as ``prefix_dims``.
    """
    if not prefix_dims:
        raise ValueError("rate_prior needs at least one prefix dimension")
    if kind not in RATE_PRIORS:
        raise ValueError(f"rate_prior kind must be one of {RATE_PRIORS}, got {kind!r}")

    if kind == "uniform":
        raw = [1.0] * len(prefix_dims)
    elif kind == "inverse_dim":
        raw = [1.0 / float(d) for d in prefix_dims]
    else:
        if weights is None:
            raise ValueError("rate_prior kind='custom' requires explicit weights")
        if len(weights) != len(prefix_dims):
            raise ValueError(
                f"rate_prior needs {len(prefix_dims)} weights for prefixes "
                f"{list(prefix_dims)}, got {len(weights)}"
            )
        raw = [float(w) for w in weights]
        if any(w < 0 for w in raw):
            raise ValueError(f"rate_prior weights must be non-negative, got {raw}")

    total = sum(raw)
    if total <= 0:
        raise ValueError(
            f"rate_prior weights sum to {total}, which cannot be normalised"
        )
    return [w / total for w in raw]


# --------------------------------------------------------------------------- #
# Neighborhood distributions (Eq 22-25)
# --------------------------------------------------------------------------- #
def neighborhood_logits(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """``a_ij = cos(z_i, z_j) / tau`` with the self-pair removed.

    ``C_i = {x_j : j != i}`` (Eq 20), so the diagonal is pushed to the dtype
    minimum rather than ``-inf``: it still receives exactly zero probability but
    can never produce ``nan`` gradients.

    Args:
        z: ``[B, d]``, normalised or not - it is L2-normalised here.
    Returns:
        ``[B, B]`` logits.
    """
    if z.dim() != 2:
        raise ValueError(f"expected a 2D [batch, dim] tensor, got {tuple(z.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    z = F.normalize(z.float(), dim=-1)
    logits = (z @ z.t()) / temperature
    return logits.masked_fill(_self_mask(logits), torch.finfo(logits.dtype).min)


def _self_mask(logits: torch.Tensor) -> torch.Tensor:
    return torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)


def candidate_mask(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor | None = None,
    mode: str = "all",
    top_m: int = 32,
) -> torch.Tensor:
    """Ablation A6: which references stay in ``C_i``.

    Args:
        teacher_logits: ``[B, B]`` from :func:`neighborhood_logits`.
        student_logits: needed only by ``"teacher_topm_student_hard"``, which
            adds the prefix's own top-``M`` so a compressed student cannot hide
            a false neighbor outside the teacher's support.
        mode: see :data:`CANDIDATE_MODES`.
        top_m: ``M``; clamped to the number of available candidates.
    Returns:
        ``[B, B]`` boolean mask, always excluding the diagonal.
    """
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"candidates must be one of {CANDIDATE_MODES}, got {mode!r}")

    keep = ~_self_mask(teacher_logits)
    if mode == "all":
        return keep

    batch_size = teacher_logits.size(0)
    m = max(1, min(int(top_m), batch_size - 1))

    selected = torch.zeros_like(keep)
    selected.scatter_(1, teacher_logits.detach().topk(m, dim=-1).indices, True)

    if mode == "teacher_topm_student_hard":
        if student_logits is None:
            raise ValueError("'teacher_topm_student_hard' needs student_logits")
        selected.scatter_(1, student_logits.detach().topk(m, dim=-1).indices, True)

    return selected & keep


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(
        logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1
    )


def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """``mean_i sum_j p_ij (log p_ij - log q_ij)`` - the Eq 27 estimator.

    ``p`` is exactly zero outside the candidate set while ``log_p``/``log_q``
    stay finite there, so the excluded terms contribute ``0`` to both the value
    and the gradient.
    """
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


def divergence_from_log_probs(
    log_p: torch.Tensor, log_q: torch.Tensor, divergence: str = "forward_kl"
) -> torch.Tensor:
    """Ablation A4. ``forward_kl`` is the default because it is mass-covering:
    it punishes a prefix for starving a neighbor the teacher considers important.
    """
    if divergence == "forward_kl":
        return _kl(log_p, log_q)
    if divergence == "reverse_kl":
        return _kl(log_q, log_p)
    if divergence == "js":
        log_m = torch.logaddexp(log_p, log_q) - torch.log(
            torch.tensor(2.0, dtype=log_p.dtype, device=log_p.device)
        )
        return 0.5 * _kl(log_p, log_m) + 0.5 * _kl(log_q, log_m)
    raise ValueError(f"divergence must be one of {DIVERGENCES}, got {divergence!r}")


def semantic_neighborhood_distortion(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    teacher_temperature: float = 0.05,
    student_temperature: float = 0.05,
    divergence: str = "forward_kl",
    candidates: str = "all",
    top_m: int = 32,
) -> torch.Tensor:
    """``D_k`` for one prefix (Eq 26, estimated by Eq 27).

    Args:
        student_repr: ``[B, d_k]`` raw prefix (normalisation happens here, so
            this is ``h_{1:d_k}`` from Eq 13).
        teacher_repr: ``[B, D]`` full-dimensional teacher; pass it already
            detached - Eq 21 makes it a stop-gradient target.
    Returns:
        A scalar; ``0`` exactly when the prefix reproduces the teacher's
        neighborhood distribution.
    """
    teacher_logits = neighborhood_logits(teacher_repr, teacher_temperature)
    student_logits = neighborhood_logits(student_repr, student_temperature)

    mask = candidate_mask(teacher_logits, student_logits, candidates, top_m)
    log_p = _masked_log_softmax(teacher_logits, mask)
    log_q = _masked_log_softmax(student_logits, mask)
    return divergence_from_log_probs(log_p, log_q, divergence)


# --------------------------------------------------------------------------- #
# Alternative geometry objectives (Ablation A5)
# --------------------------------------------------------------------------- #
def gram_mse_distortion(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    candidates: str = "all",
    top_m: int = 32,
    teacher_temperature: float = 0.05,
) -> torch.Tensor:
    """Eq 105: ``||G_T - G_k||_F^2`` over cosine Gram matrices.

    Averaged over the retained entries rather than summed, so the scale does not
    move with the batch size; the constant is absorbed by ``lambda_sem``.
    """
    student = F.normalize(student_repr.float(), dim=-1)
    teacher = F.normalize(teacher_repr.float(), dim=-1)

    gram_student = student @ student.t()
    gram_teacher = teacher @ teacher.t()

    mask = candidate_mask(
        neighborhood_logits(teacher_repr, teacher_temperature),
        neighborhood_logits(student_repr, teacher_temperature),
        candidates,
        top_m,
    )
    squared = (gram_student - gram_teacher).pow(2)
    return squared[mask].mean()


def hard_neighbor_cross_entropy(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    teacher_temperature: float = 0.05,
    student_temperature: float = 0.05,
) -> torch.Tensor:
    """Cross entropy against the teacher's single nearest neighbor (Sec 8.2).

    The degenerate, non-distributional limit of the semantic decoder: it keeps
    only ``argmax_j p_T(S = j | x_i)``.
    """
    teacher_logits = neighborhood_logits(teacher_repr, teacher_temperature)
    student_logits = neighborhood_logits(student_repr, student_temperature)
    target = teacher_logits.detach().argmax(dim=-1)
    return F.cross_entropy(student_logits, target)


# --------------------------------------------------------------------------- #
# The multi-rate objective (Eq 48, 54, 55) and Algorithm 1
# --------------------------------------------------------------------------- #
class SemanticDistortionLoss(nn.Module):
    """``sum_k pi_k D_k`` plus the optional monotonic refinement regulariser.

    The module holds no parameters - the semantic decoder of Eq 25 is the prefix
    cosine geometry itself, which is exactly why SDR-MRL adds nothing at
    inference time.
    """

    def __init__(
        self,
        dims: Sequence[int],
        full_dim: int,
        teacher_temperature: float = 0.05,
        student_temperature: float = 0.05,
        divergence: str = "forward_kl",
        geometry: str = "snd",
        candidates: str = "all",
        top_m: int = 32,
        rate_prior_kind: str = "uniform",
        rate_weights: Sequence[float] | None = None,
        lambda_mono: float = 0.0,
        stochastic_rate: bool = False,
    ):
        """
        Args:
            dims: every nested dimension, including ``full_dim``.
            full_dim: ``D``; its own prefix is the teacher, so Eq 48 stops at
                ``K - 1`` and it is excluded from the loss.
            stochastic_rate: Sec 4.13 - draw one ``k ~ pi`` per step instead of
                walking every prefix. Unbiased for Eq 48 (Eq 68) and much cheaper.
        """
        super().__init__()

        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if divergence not in DIVERGENCES:
            raise ValueError(
                f"divergence must be one of {DIVERGENCES}, got {divergence!r}"
            )

        usable = sorted({int(d) for d in dims if d <= full_dim})
        self.prefix_dims: list[int] = [d for d in usable if d < full_dim]
        if not self.prefix_dims:
            raise ValueError(
                f"no truncated prefix below full_dim={full_dim} in dims={sorted(dims)}"
            )

        self.full_dim = int(full_dim)
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.divergence = divergence
        self.geometry = geometry
        self.candidates = candidates
        self.top_m = top_m
        self.lambda_mono = lambda_mono
        self.stochastic_rate = stochastic_rate

        self.rate_prior: list[float] = rate_prior(
            self.prefix_dims, rate_prior_kind, rate_weights
        )
        self.register_buffer(
            "_prior", torch.tensor(self.rate_prior, dtype=torch.float), persistent=False
        )
        self.cka = CKALoss() if geometry == "cka" else None

    # -- one prefix --------------------------------------------------------- #
    def distortion_at(
        self, student_repr: torch.Tensor, teacher_repr: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """``D_k`` under the configured geometry (A5)."""
        prefix = student_repr[:, :dim]

        if self.geometry == "snd":
            return semantic_neighborhood_distortion(
                prefix,
                teacher_repr,
                teacher_temperature=self.teacher_temperature,
                student_temperature=self.student_temperature,
                divergence=self.divergence,
                candidates=self.candidates,
                top_m=self.top_m,
            )
        if self.geometry == "gram_mse":
            return gram_mse_distortion(
                prefix,
                teacher_repr,
                candidates=self.candidates,
                top_m=self.top_m,
                teacher_temperature=self.teacher_temperature,
            )
        if self.geometry == "hard_neighbor":
            return hard_neighbor_cross_entropy(
                prefix,
                teacher_repr,
                teacher_temperature=self.teacher_temperature,
                student_temperature=self.student_temperature,
            )
        return self.cka(F.normalize(prefix, dim=-1), F.normalize(teacher_repr, dim=-1))

    # -- rate sampling (Eq 66-69) ------------------------------------------- #
    def sample_rate_index(self, generator: torch.Generator | None = None) -> int:
        """Draw ``k ~ pi``. Returns an index into :attr:`prefix_dims`."""
        return int(
            torch.multinomial(self._prior, num_samples=1, generator=generator).item()
        )

    def rates_for_step(self, rate_index: int | None) -> list[int]:
        """Which prefixes this step has to evaluate.

        Full-prefix training walks all of them; the stochastic variant needs only
        the sampled ``k``, plus its neighbour ``k - 1`` when the monotonic edge
        ``(k - 1, k)`` is being regularised (Eq 69).
        """
        if rate_index is None:
            return list(range(len(self.prefix_dims)))
        indices = [rate_index]
        if self.lambda_mono > 0 and rate_index > 0:
            indices.insert(0, rate_index - 1)
        return indices

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
        rate_index: int | None = None,
    ) -> dict[str, object]:
        """
        Args:
            student_repr: ``[B, D]`` pooled representation with gradients.
            teacher_repr: ``[B, D_T]`` stop-gradient teacher (Eq 21). Its width
                is free: only its neighborhood distribution is ever used, so an
                EMA or independently trained teacher may have any dimensionality.
            rate_index: index into :attr:`prefix_dims` for the sampled-rate
                variant; ``None`` evaluates every prefix. Ignored (treated as
                ``None``) when :attr:`stochastic_rate` is off.
        Returns:
            ``{"sem_loss", "mono_loss", "distortions": {dim: tensor}}`` where
            ``sem_loss`` estimates ``sum_k pi_k D_k`` (Eq 48).
        """
        if teacher_repr.requires_grad:
            teacher_repr = teacher_repr.detach()
        if not self.stochastic_rate:
            rate_index = None

        indices = self.rates_for_step(rate_index)
        distortions = {
            self.prefix_dims[i]: self.distortion_at(
                student_repr, teacher_repr, self.prefix_dims[i]
            )
            for i in indices
        }

        if rate_index is None:
            # Eq 48 exactly.
            sem_loss = student_repr.new_zeros(())
            for weight, dim in zip(self.rate_prior, self.prefix_dims):
                sem_loss = sem_loss + weight * distortions[dim]
        else:
            # Eq 67: D_{K_t} with K_t ~ pi is unbiased for the weighted sum.
            sem_loss = distortions[self.prefix_dims[rate_index]]

        return {
            "sem_loss": sem_loss,
            "mono_loss": self.monotonic_penalty(distortions),
            "distortions": distortions,
        }

    def monotonic_penalty(self, distortions: dict[int, torch.Tensor]) -> torch.Tensor:
        """Eq 54: ``sum_k [D_k - D_{k-1}]_+``.

        Optimal distortion is monotone in the rate (Prop 2, Eq 42) but the
        practical cosine decoder can violate it; this discourages the cases where
        *adding* coordinates makes the neighborhood harder to recover.
        """
        dims = sorted(distortions)
        first = distortions[dims[0]]
        if self.lambda_mono <= 0 or len(dims) < 2:
            return first.new_zeros(())

        penalty = first.new_zeros(())
        for previous, current in zip(dims, dims[1:]):
            penalty = penalty + torch.clamp(
                distortions[current] - distortions[previous], min=0.0
            )
        return penalty


__all__ = [
    "CANDIDATE_MODES",
    "DIVERGENCES",
    "GEOMETRIES",
    "RATE_PRIORS",
    "SemanticDistortionLoss",
    "candidate_mask",
    "divergence_from_log_probs",
    "gram_mse_distortion",
    "hard_neighbor_cross_entropy",
    "neighborhood_logits",
    "rate_prior",
    "semantic_neighborhood_distortion",
]
