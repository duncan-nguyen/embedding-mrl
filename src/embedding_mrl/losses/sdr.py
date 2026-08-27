"""SDR-MRL: semantic distortion-rate objectives for Matryoshka representations.

Equation numbers refer to *Beyond Representation Alignment: Semantic
Distortion-Rate Learning for Matryoshka Representations* (``docs/latex/main.pdf``).

The method treats every Matryoshka prefix ``Z_k`` as a representation available
at one deployment rate ``R_k = b d_k`` (Eq 18) and asks how much of the
*semantic neighborhood* induced by the full-dimensional teacher is still
recoverable from that prefix::

    p_T(S = j | x_i) = softmax_j cos(z_i^(K), z_j^(K)) / tau_T      (Eq 22-23)
    q_k(S = j | z_i) = softmax_j cos(z_i^(k), z_j^(k)) / tau_k      (Eq 24-25)
    D_k              = E_X KL(p_T(S|X) || q_k(S|Z_k))               (Eq 26)

Proposition 1 (Eq 30) turns ``D_k`` into a variational upper bound on the
conditional semantic information ``I_T(S; X | Z_k)``, so minimising it across a
deployment-rate prior ``pi`` (Eq 48) is the entire training signal::

    L_SDR = L_task + lambda_sem * sum_k pi_k D_k
                   + lambda_mono * sum_k [D_k - sg(D_{k-1})]_+      (Eq 55)

Three details matter for the bound to mean what the paper says it means:

* **The decoder temperature is per prefix** (Eq 24 writes ``tau_k``). The
  optimal decoder ``q_k^* = p_T(S | Z_k)`` is a Bayesian average over every
  input sharing the prefix, hence *flatter* than ``p_T(S | X)``. A cosine
  decoder tied to ``tau_T`` cannot express that, which inflates ``epsilon_k``
  at low rates. Minimising over ``tau_k`` is the infimum over the decoder
  family, so a learnable ``tau_k`` only ever tightens the bound; ``tau_K =
  tau_T`` keeps ``D_K = 0``.
* **The candidate set can extend beyond the batch** (Sec 4.2). With ``B = 16``
  the semantic variable has at most ``log 15`` nats; a memory queue of past
  embeddings (``extra`` below) makes ``p_T`` peak on genuine neighbours.
* **The monotonic hinge stops gradient into the lower prefix.** Without it
  ``d[D_k - D_{k-1}]_+ / dD_{k-1} = -1`` and the regulariser is happy to make
  the *smaller* prefix worse.

Everything the ablation table (Sec 8) asks to vary is a constructor argument:
the divergence (A4), the geometry objective (A5), the candidate neighborhood
(A6), the rate prior (A7), full-prefix versus sampled-rate training (A8) and the
temperatures (A9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CANDIDATE_MODES, DIVERGENCES, GEOMETRIES, RATE_PRIORS
from .cka import CKALoss

Temperature = "float | torch.Tensor"


def _temperature_value(temperature) -> float:
    if torch.is_tensor(temperature):
        return float(temperature.detach())
    return float(temperature)


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
def neighborhood_logits(
    z: torch.Tensor,
    temperature,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a_ij = cos(z_i, c_j) / tau`` over the candidate set, self-pair removed.

    The candidates are the batch itself plus, optionally, ``extra`` - a memory
    queue of previously seen embeddings (Sec 4.2). ``C_i = {x_j : j != i}``
    (Eq 20), so the anchor's own column is pushed to the dtype minimum rather
    than ``-inf``: it still receives exactly zero probability but can never
    produce ``nan`` gradients.

    Args:
        z: ``[B, d]``, normalised or not - it is L2-normalised here.
        temperature: a float or a 0-d tensor (learnable ``tau_k``).
        extra: ``[Q, d]`` additional candidates that receive no gradient.
    Returns:
        ``[B, B + Q]`` logits.
    """
    if z.dim() != 2:
        raise ValueError(f"expected a 2D [batch, dim] tensor, got {tuple(z.shape)}")
    if _temperature_value(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    z = F.normalize(z.float(), dim=-1)
    if extra is None:
        references = z
    else:
        if extra.dim() != 2 or extra.size(1) != z.size(1):
            raise ValueError(
                f"extra candidates must be [Q, {z.size(1)}], got {tuple(extra.shape)}"
            )
        references = torch.cat([z, F.normalize(extra.detach().float(), dim=-1)], dim=0)

    logits = (z @ references.t()) / temperature
    return logits.masked_fill(_self_mask(logits), torch.finfo(logits.dtype).min)


def _self_mask(logits: torch.Tensor) -> torch.Tensor:
    """``[B, N]`` mask of the anchor's own column (the first ``B`` columns are the batch)."""
    batch = logits.size(0)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    index = torch.arange(batch, device=logits.device)
    mask[index, index] = True
    return mask


def candidate_mask(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor | None = None,
    mode: str = "all",
    top_m: int = 32,
) -> torch.Tensor:
    """Ablation A6: which references stay in ``C_i``.

    Args:
        teacher_logits: ``[B, N]`` from :func:`neighborhood_logits`.
        student_logits: needed only by ``"teacher_topm_student_hard"``, which
            adds the prefix's own top-``M`` so a compressed student cannot hide
            a false neighbor outside the teacher's support.
        mode: see :data:`CANDIDATE_MODES`.
        top_m: ``M``; clamped to the number of available candidates.
    Returns:
        ``[B, N]`` boolean mask, always excluding the anchor itself.
    """
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"candidates must be one of {CANDIDATE_MODES}, got {mode!r}")

    keep = ~_self_mask(teacher_logits)
    if mode == "all":
        return keep

    available = teacher_logits.size(1) - 1
    m = max(1, min(int(top_m), available))

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
    teacher_temperature=0.05,
    student_temperature=0.05,
    divergence: str = "forward_kl",
    candidates: str = "all",
    top_m: int = 32,
    student_extra: torch.Tensor | None = None,
    teacher_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """``D_k`` for one prefix (Eq 26, estimated by Eq 27).

    Args:
        student_repr: ``[B, d_k]`` raw prefix (normalisation happens here, so
            this is ``h_{1:d_k}`` from Eq 13).
        teacher_repr: ``[B, D]`` full-dimensional teacher; pass it already
            detached - Eq 21 makes it a stop-gradient target.
        student_temperature: ``tau_k`` - float or learnable 0-d tensor.
        student_extra / teacher_extra: ``[Q, d_k]`` / ``[Q, D]`` queue entries
            for the same ``Q`` past items, appended to the candidate set on
            both sides. Either both or neither must be given.
    Returns:
        A scalar; ``0`` exactly when the prefix reproduces the teacher's
        neighborhood distribution.
    """
    if (student_extra is None) != (teacher_extra is None):
        raise ValueError("student_extra and teacher_extra must be given together")
    if student_extra is not None and student_extra.size(0) != teacher_extra.size(0):
        raise ValueError(
            f"queue sides disagree: {student_extra.size(0)} student rows vs "
            f"{teacher_extra.size(0)} teacher rows"
        )

    teacher_logits = neighborhood_logits(teacher_repr, teacher_temperature, teacher_extra)
    student_logits = neighborhood_logits(student_repr, student_temperature, student_extra)

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
    teacher_temperature=0.05,
    student_extra: torch.Tensor | None = None,
    teacher_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq 105: ``||G_T - G_k||_F^2`` over cosine Gram matrices.

    Averaged over the retained entries rather than summed, so the scale does not
    move with the batch size; the constant is absorbed by ``lambda_sem``.
    """
    # Cosine Gram matrices are the logits at temperature 1.
    gram_teacher = neighborhood_logits(teacher_repr, 1.0, teacher_extra)
    gram_student = neighborhood_logits(student_repr, 1.0, student_extra)

    mask = candidate_mask(
        neighborhood_logits(teacher_repr, teacher_temperature, teacher_extra),
        neighborhood_logits(student_repr, teacher_temperature, student_extra),
        candidates,
        top_m,
    )
    squared = (gram_student - gram_teacher).pow(2)
    return squared[mask].mean()


def hard_neighbor_cross_entropy(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    teacher_temperature=0.05,
    student_temperature=0.05,
    student_extra: torch.Tensor | None = None,
    teacher_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy against the teacher's single nearest neighbor (Sec 8.2).

    The degenerate, non-distributional limit of the semantic decoder: it keeps
    only ``argmax_j p_T(S = j | x_i)``.
    """
    teacher_logits = neighborhood_logits(teacher_repr, teacher_temperature, teacher_extra)
    student_logits = neighborhood_logits(student_repr, student_temperature, student_extra)
    target = teacher_logits.detach().argmax(dim=-1)
    return F.cross_entropy(student_logits, target)


# --------------------------------------------------------------------------- #
# The multi-rate objective (Eq 48, 54, 55) and Algorithm 1
# --------------------------------------------------------------------------- #
class SemanticDistortionLoss(nn.Module):
    """``sum_k pi_k D_k`` plus the optional monotonic refinement regulariser.

    The only parameters are the ``K - 1`` decoder temperatures ``tau_k``
    (Eq 24), and only when ``learnable_temperature`` is on. They are part of
    the *decoder*, not of the representation: nothing is added at inference
    time, where the prefix cosine geometry is used as is.
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
        learnable_temperature: bool = False,
        temperature_bounds: tuple[float, float] = (0.01, 1.0),
    ):
        """
        Args:
            dims: every nested dimension, including ``full_dim``.
            full_dim: ``D``; its own prefix is the teacher, so Eq 48 stops at
                ``K - 1`` and it is excluded from the loss.
            student_temperature: the initial ``tau_k`` for every prefix.
            stochastic_rate: Sec 4.13 - draw one ``k ~ pi`` per step instead of
                walking every prefix. Unbiased for Eq 48 (Eq 68) and much cheaper.
            learnable_temperature: let each ``tau_k`` be optimised (Eq 24). The
                infimum over the decoder family can only lower ``D_k``, so this
                tightens Prop 1's bound rather than changing the objective.
            temperature_bounds: ``tau_k`` is clamped to this range.
        """
        super().__init__()

        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if divergence not in DIVERGENCES:
            raise ValueError(
                f"divergence must be one of {DIVERGENCES}, got {divergence!r}"
            )
        low, high = (float(temperature_bounds[0]), float(temperature_bounds[1]))
        if not 0 < low <= student_temperature <= high:
            raise ValueError(
                f"student_temperature={student_temperature} must lie within "
                f"temperature_bounds={temperature_bounds}"
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
        self.learnable_temperature = learnable_temperature
        self.temperature_bounds = (low, high)

        self.rate_prior: list[float] = rate_prior(
            self.prefix_dims, rate_prior_kind, rate_weights
        )
        self.register_buffer(
            "_prior", torch.tensor(self.rate_prior, dtype=torch.float), persistent=False
        )

        # Eq 24: one decoder temperature per truncated prefix, parametrised in
        # log space so the clamp is the only constraint.
        initial = torch.full(
            (len(self.prefix_dims),), math.log(student_temperature), dtype=torch.float
        )
        if learnable_temperature:
            self.log_tau = nn.Parameter(initial)
        else:
            self.register_buffer("log_tau", initial)

        self.cka = CKALoss() if geometry == "cka" else None

    # -- temperatures ------------------------------------------------------- #
    def temperature_at(self, dim: int) -> torch.Tensor:
        """``tau_k`` for prefix ``dim`` as a 0-d tensor (differentiable if learnable)."""
        index = self.prefix_dims.index(int(dim))
        low, high = self.temperature_bounds
        return self.log_tau[index].clamp(math.log(low), math.log(high)).exp()

    @property
    def student_temperatures(self) -> dict[int, float]:
        """``{d_k: tau_k}`` for logging and diagnostics."""
        return {dim: float(self.temperature_at(dim).detach()) for dim in self.prefix_dims}

    # -- one prefix --------------------------------------------------------- #
    def distortion_at(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
        dim: int,
        student_extra: torch.Tensor | None = None,
        teacher_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``D_k`` under the configured geometry (A5)."""
        prefix = student_repr[:, :dim]
        prefix_extra = None if student_extra is None else student_extra[:, :dim]
        tau_k = self.temperature_at(dim)

        if self.geometry == "snd":
            return semantic_neighborhood_distortion(
                prefix,
                teacher_repr,
                teacher_temperature=self.teacher_temperature,
                student_temperature=tau_k,
                divergence=self.divergence,
                candidates=self.candidates,
                top_m=self.top_m,
                student_extra=prefix_extra,
                teacher_extra=teacher_extra,
            )
        if self.geometry == "gram_mse":
            return gram_mse_distortion(
                prefix,
                teacher_repr,
                candidates=self.candidates,
                top_m=self.top_m,
                teacher_temperature=self.teacher_temperature,
                student_extra=prefix_extra,
                teacher_extra=teacher_extra,
            )
        if self.geometry == "hard_neighbor":
            return hard_neighbor_cross_entropy(
                prefix,
                teacher_repr,
                teacher_temperature=self.teacher_temperature,
                student_temperature=tau_k,
                student_extra=prefix_extra,
                teacher_extra=teacher_extra,
            )
        # CKA is a whole-batch statistic between two feature spaces; a queue of
        # unrelated rows has no place in it.
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
        student_extra: torch.Tensor | None = None,
        teacher_extra: torch.Tensor | None = None,
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
            student_extra / teacher_extra: memory-queue candidates, ``[Q, D]``
                and ``[Q, D_T]`` for the same ``Q`` items (Sec 4.2).
        Returns:
            ``{"sem_loss", "mono_loss", "distortions": {dim: tensor},
            "temperatures": {dim: float}}`` where ``sem_loss`` estimates
            ``sum_k pi_k D_k`` (Eq 48).
        """
        if teacher_repr.requires_grad:
            teacher_repr = teacher_repr.detach()
        if not self.stochastic_rate:
            rate_index = None

        indices = self.rates_for_step(rate_index)
        distortions = {
            self.prefix_dims[i]: self.distortion_at(
                student_repr,
                teacher_repr,
                self.prefix_dims[i],
                student_extra=student_extra,
                teacher_extra=teacher_extra,
            )
            for i in indices
        }

        if rate_index is None:
            # Eq 48 exactly, and Eq 54 exactly.
            sem_loss = student_repr.new_zeros(())
            for weight, dim in zip(self.rate_prior, self.prefix_dims):
                sem_loss = sem_loss + weight * distortions[dim]
            mono_loss = self.monotonic_penalty(distortions)
        else:
            # Eq 67: D_{K_t} with K_t ~ pi is unbiased for the weighted sum.
            sem_loss = distortions[self.prefix_dims[rate_index]]
            # Eq 69: the edge (k-1, k) is drawn with probability pi_k, so the
            # importance weight 1/pi_k makes it unbiased for Eq 54's plain sum.
            mono_loss = self.monotonic_penalty(
                distortions, scale=1.0 / self.rate_prior[rate_index]
            )

        return {
            "sem_loss": sem_loss,
            "mono_loss": mono_loss,
            "distortions": distortions,
            "temperatures": {
                dim: float(self.temperature_at(dim).detach()) for dim in distortions
            },
        }

    def monotonic_penalty(
        self, distortions: dict[int, torch.Tensor], scale: float = 1.0
    ) -> torch.Tensor:
        """Eq 54: ``sum_k [D_k - sg(D_{k-1})]_+``.

        Optimal distortion is monotone in the rate (Prop 2, Eq 42) but the
        practical cosine decoder can violate it. The lower prefix is a
        stop-gradient *target*: the hinge may only pull ``D_k`` down, never push
        ``D_{k-1}`` up, otherwise its cheapest fix is to degrade the smaller
        prefix - the opposite of what a Matryoshka objective wants.
        """
        dims = sorted(distortions)
        first = distortions[dims[0]]
        if self.lambda_mono <= 0 or len(dims) < 2:
            return first.new_zeros(())

        penalty = first.new_zeros(())
        for previous, current in zip(dims, dims[1:]):
            penalty = penalty + torch.clamp(
                distortions[current] - distortions[previous].detach(), min=0.0
            )
        return scale * penalty


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
