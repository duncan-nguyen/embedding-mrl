"""Semantic distortion-rate evaluation (SDR-MRL Sec 6 and Sec 9).

Everything here is pure numerics over already-extracted embeddings: no model, no
tokenizer, no data loading. :mod:`embedding_mrl.evaluation` supplies the
embeddings; this module turns them into the diagnostics the proposal asks for:

* the fixed-teacher distortion-rate profile ``P = {(R_k, D_k^ref)}`` (Eq 86-87);
* the normalised distortion area ``SDRA`` (Eq 89-91);
* neighborhood preservation - kNN recall (Eq 96), similarity rank correlation,
  trustworthiness and continuity (Sec 6.4);
* the monotonicity violation rate ``V_mono`` (Eq 108);
* semantic gain per added coordinate ``eta_k`` (Eq 122);
* the price of nestedness ``P_nest`` (Eq 92-93);
* ``Spearman(G_k, dQ_k)``, does distortion reduction predict utility (Eq 119-121);
* the random rotation stress test (Eq 114-118).

All distortions use the same estimator the training loss does, so a number
printed by the evaluator is directly comparable to a number logged during
training - provided the same teacher and temperatures are used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .losses.sdr import _masked_log_softmax, divergence_from_log_probs

#: Rows of the ``N x N`` neighborhood matrix processed at once.
DEFAULT_CHUNK = 512


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _chunks(total: int, size: int):
    for start in range(0, total, max(1, size)):
        yield start, min(start + max(1, size), total)


def _logits_block(
    anchors: torch.Tensor, corpus: torch.Tensor, temperature: float, offset: int
) -> torch.Tensor:
    """``[rows, N]`` cosine logits with the anchors' own column masked out."""
    logits = (anchors @ corpus.t()) / temperature
    rows = torch.arange(anchors.size(0), device=logits.device)
    logits[rows, rows + offset] = torch.finfo(logits.dtype).min
    return logits


# --------------------------------------------------------------------------- #
# Distortion-rate profile (Sec 6.1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def semantic_distortion(
    student: torch.Tensor,
    teacher: torch.Tensor,
    teacher_temperature: float = 0.05,
    student_temperature: float = 0.05,
    divergence: str = "forward_kl",
    chunk: int = DEFAULT_CHUNK,
) -> float:
    """``D_k`` over a whole corpus, with every other item as a candidate (Eq 26).

    Args:
        student: ``[N, d]`` prefix embeddings.
        teacher: ``[N, D]`` reference-teacher embeddings for the same items.
    """
    if student.size(0) != teacher.size(0):
        raise ValueError(
            f"student has {student.size(0)} rows but teacher has {teacher.size(0)}"
        )
    if student.size(0) < 3:
        raise ValueError("semantic distortion needs at least 3 items")

    student = _normalize(student)
    teacher = _normalize(teacher)

    total, count = 0.0, 0
    for start, end in _chunks(student.size(0), chunk):
        teacher_logits = _logits_block(
            teacher[start:end], teacher, teacher_temperature, start
        )
        student_logits = _logits_block(
            student[start:end], student, student_temperature, start
        )
        mask = teacher_logits > torch.finfo(teacher_logits.dtype).min

        value = divergence_from_log_probs(
            _masked_log_softmax(teacher_logits, mask),
            _masked_log_softmax(student_logits, mask),
            divergence,
        )
        rows = end - start
        total += float(value) * rows
        count += rows
    return total / count


@torch.no_grad()
def zero_rate_distortion(
    teacher: torch.Tensor, temperature: float = 0.05, chunk: int = DEFAULT_CHUNK
) -> float:
    """Eq 88: ``D_0^ref = E KL(p_T || q_0)`` against the uniform decoder.

    With ``q_0`` uniform over the ``N - 1`` candidates this is exactly
    ``log(N - 1) - H(p_T)``, the semantic information a zero-rate representation
    fails to convey. It is the normaliser of Eq 89.
    """
    teacher = _normalize(teacher)
    n = teacher.size(0)

    entropy, count = 0.0, 0
    for start, end in _chunks(n, chunk):
        logits = _logits_block(teacher[start:end], teacher, temperature, start)
        mask = logits > torch.finfo(logits.dtype).min
        log_p = _masked_log_softmax(logits, mask)
        rows = end - start
        entropy += float(-(log_p.exp() * log_p).sum(dim=-1).sum())
        count += rows
    return float(np.log(n - 1) - entropy / count)


@torch.no_grad()
def distortion_profile(
    student: torch.Tensor,
    teacher: torch.Tensor,
    dims: Sequence[int],
    teacher_temperature: float = 0.05,
    student_temperature: float = 0.05,
    divergence: str = "forward_kl",
    chunk: int = DEFAULT_CHUNK,
) -> dict[int, float]:
    """Eq 86-87: ``D_k^ref`` at every requested prefix, against a fixed teacher."""
    return {
        int(dim): semantic_distortion(
            student[:, :dim],
            teacher,
            teacher_temperature=teacher_temperature,
            student_temperature=student_temperature,
            divergence=divergence,
            chunk=chunk,
        )
        for dim in sorted(dims)
        if dim <= student.size(1)
    }


def normalized_distortion(
    profile: dict[int, float], zero_rate: float, eps: float = 1e-8
) -> dict[int, float]:
    """Eq 89: rescale so the zero-rate decoder is ``1`` and the full width ``0``.

    Making the endpoints comparable is what lets distortion curves from models of
    different absolute quality be plotted on one axis.
    """
    if not profile:
        return {}
    full = profile[max(profile)]
    span = zero_rate - full + eps
    return {dim: (value - full) / span for dim, value in profile.items()}


def sdra(normalized: dict[int, float], full_dim: int) -> float:
    """Eq 91: trapezoidal area under the normalised distortion-rate curve.

    The curve starts at the zero-rate point ``(r = 0, D~ = 1)``. Lower is better:
    it means useful semantic structure becomes recoverable earlier in the code.
    """
    if not normalized:
        raise ValueError("SDRA needs a non-empty normalised profile")

    rates = [0.0] + [dim / float(full_dim) for dim in sorted(normalized)]
    values = [1.0] + [normalized[dim] for dim in sorted(normalized)]

    return float(
        sum(
            (values[i - 1] + values[i]) / 2.0 * (rates[i] - rates[i - 1])
            for i in range(1, len(rates))
        )
    )


def monotonicity_violation_rate(
    profile: dict[int, float], exclude_full: bool = True
) -> float:
    """Eq 108: the fraction of adjacent prefixes where ``D_k > D_{k-1}``.

    Prop 2 guarantees monotonicity only for the *optimal* decoder, so a non-zero
    rate here is exactly what ``lambda_mono`` is meant to reduce.

    Args:
        profile: ``dim -> D_k``.
        exclude_full: Eq 108 runs over ``k = 2 .. K - 1``, i.e. the truncated
            prefixes only. Leave this on when ``profile`` includes the full
            width, whose distortion is ~0 by construction and would otherwise
            make every model look perfectly monotone on its last edge.
    """
    dims = sorted(profile)
    if exclude_full and len(dims) > 1:
        dims = dims[:-1]
    if len(dims) < 2:
        return 0.0

    edges = list(zip(dims[:-1], dims[1:]))
    violations = sum(
        1 for previous, current in edges if profile[current] > profile[previous]
    )
    return violations / len(edges)


def refinement_gain(profile: dict[int, float]) -> dict[int, float]:
    """Eq 122: ``eta_k = (D_{k-1} - D_k) / (d_k - d_{k-1})``.

    Semantic value bought per extra coordinate. Purely diagnostic - the method
    deliberately does not force it to decrease (Sec 4.10).
    """
    dims = sorted(profile)
    return {
        current: (profile[previous] - profile[current]) / float(current - previous)
        for previous, current in zip(dims[:-1], dims[1:])
    }


def distortion_reduction(profile: dict[int, float]) -> dict[int, float]:
    """Eq 119: ``G_k = D_{k-1}^ref - D_k^ref`` for each adjacent prefix pair."""
    dims = sorted(profile)
    return {
        current: profile[previous] - profile[current]
        for previous, current in zip(dims[:-1], dims[1:])
    }


# --------------------------------------------------------------------------- #
# Neighborhood preservation (Sec 6.4)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def knn_recall(
    student: torch.Tensor,
    teacher: torch.Tensor,
    k: int = 10,
    chunk: int = DEFAULT_CHUNK,
) -> float:
    """Eq 96: overlap between the teacher's top-``k`` and the prefix's top-``k``.

    Unlike the distortion this is a hard, rank-based quantity, so it cannot be
    improved by merely matching the teacher's similarity *scale*.
    """
    student = _normalize(student)
    teacher = _normalize(teacher)
    n = student.size(0)
    k = max(1, min(k, n - 1))

    hits, count = 0.0, 0
    for start, end in _chunks(n, chunk):
        teacher_top = (
            _logits_block(teacher[start:end], teacher, 1.0, start)
            .topk(k, dim=-1)
            .indices
        )
        student_top = (
            _logits_block(student[start:end], student, 1.0, start)
            .topk(k, dim=-1)
            .indices
        )
        for row in range(end - start):
            overlap = len(
                set(teacher_top[row].tolist()) & set(student_top[row].tolist())
            )
            hits += overlap / k
        count += end - start
    return hits / count


@torch.no_grad()
def similarity_spearman(
    student: torch.Tensor,
    teacher: torch.Tensor,
    max_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Rank correlation between teacher and prefix pairwise cosine similarities.

    Sampled rather than exhaustive: an ``N x N`` upper triangle is quadratic and
    the correlation is stable well before it is fully enumerated.
    """
    student = _normalize(student)
    teacher = _normalize(teacher)
    n = student.size(0)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randint(0, n, (max_pairs,), generator=generator)
    cols = torch.randint(0, n, (max_pairs,), generator=generator)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]

    teacher_sim = (teacher[rows] * teacher[cols]).sum(dim=-1).cpu().numpy()
    student_sim = (student[rows] * student[cols]).sum(dim=-1).cpu().numpy()
    correlation, _ = spearmanr(teacher_sim, student_sim)
    return float(correlation)


@torch.no_grad()
def trustworthiness_and_continuity(
    student: torch.Tensor, teacher: torch.Tensor, k: int = 10
) -> tuple[float, float]:
    """Sec 6.4. Trustworthiness penalises *new* neighbors the prefix invents;
    continuity penalises *true* neighbors it loses.

    Both are computed on L2-normalised vectors with Euclidean distance, which is
    a monotone function of cosine on the unit sphere and therefore induces the
    same neighbor ranking.
    """
    from sklearn.manifold import trustworthiness

    student_np = _normalize(student).cpu().numpy()
    teacher_np = _normalize(teacher).cpu().numpy()
    k = max(1, min(k, (student_np.shape[0] - 1) // 2))

    return (
        float(trustworthiness(teacher_np, student_np, n_neighbors=k)),
        float(trustworthiness(student_np, teacher_np, n_neighbors=k)),
    )


# --------------------------------------------------------------------------- #
# Price of nestedness (Sec 6.3) and the RQ4 diagnostic (Sec 9.2)
# --------------------------------------------------------------------------- #
def price_of_nestedness(
    nested: dict[int, float],
    independent: dict[int, float],
    prior: dict[int, float] | None = None,
) -> dict[str, object]:
    """Eq 92-93: quality lost by making one representation serve every rate.

    Args:
        nested: ``dim -> Q(f_nested^{1:d})``, the prefix of a Matryoshka model.
        independent: ``dim -> Q(f_d^*)``, models trained at that width only.
        prior: optional deployment prior for the aggregate (Eq 93); uniform over
            the shared dims otherwise.
    Returns:
        ``{"per_dim": {dim: P_nest(d)}, "aggregate": float}``. Positive values
        mean the independent model is better, i.e. nestedness costs something.
    """
    shared = sorted(set(nested) & set(independent))
    if not shared:
        raise ValueError("nested and independent results share no dimension")

    per_dim = {dim: independent[dim] - nested[dim] for dim in shared}
    if prior is None:
        weights = {dim: 1.0 / len(shared) for dim in shared}
    else:
        total = sum(prior.get(dim, 0.0) for dim in shared)
        if total <= 0:
            raise ValueError("the deployment prior puts no mass on the shared dims")
        weights = {dim: prior.get(dim, 0.0) / total for dim in shared}

    return {
        "per_dim": per_dim,
        "aggregate": float(sum(weights[dim] * per_dim[dim] for dim in shared)),
    }


def distortion_gain_correlation(
    distortion_reductions: dict[int, float], quality_gains: dict[int, float]
) -> dict[str, float]:
    """Eq 121: ``rho = Spearman(G_k, dQ_k)`` across adjacent prefix pairs.

    A strong positive correlation is what licenses reading a distortion drop as
    "this block bought real semantic information"; RQ4 lives or dies here.
    """
    shared = sorted(set(distortion_reductions) & set(quality_gains))
    if len(shared) < 3:
        return {"spearman": float("nan"), "p_value": float("nan"), "n": len(shared)}

    correlation, p_value = spearmanr(
        [distortion_reductions[d] for d in shared], [quality_gains[d] for d in shared]
    )
    return {"spearman": float(correlation), "p_value": float(p_value), "n": len(shared)}


# --------------------------------------------------------------------------- #
# Random rotation stress test (Sec 9.1)
# --------------------------------------------------------------------------- #
def random_orthogonal(dim: int, seed: int = 0, device=None) -> torch.Tensor:
    """A Haar-ish orthogonal ``Q`` from the QR decomposition of a Gaussian."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    # Fix the sign convention so Q is uniformly distributed, not QR-biased.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device) if device is not None else q


@torch.no_grad()
def rotation_stress_test(
    embeddings: torch.Tensor,
    dims: Sequence[int],
    num_rotations: int = 3,
    k: int = 10,
    seed: int = 0,
) -> dict[str, object]:
    """Eq 114-118: full-space geometry is rotation-invariant, prefixes are not.

    Rotating an embedding leaves ``Z Z^T`` untouched (so every full-dimensional
    retrieval metric is unchanged) while scrambling which semantic directions
    land in the early coordinates. The gap between ``baseline`` and ``rotated``
    is the part of prefix quality that *coordinate ordering* alone is responsible
    for - the motivation for the whole method.

    Returns:
        ``{"full_dim_gram_shift", "baseline": {dim: kNN recall},
        "rotated": {dim: [recall per rotation]}, "mean_drop": {dim: float}}``
    """
    teacher = _normalize(embeddings)
    full_dim = teacher.size(1)
    usable = [int(d) for d in sorted(dims) if d < full_dim]

    baseline = {dim: knn_recall(teacher[:, :dim], teacher, k=k) for dim in usable}
    rotated: dict[int, list[float]] = {dim: [] for dim in usable}
    gram_shift = 0.0

    for index in range(num_rotations):
        rotation = random_orthogonal(full_dim, seed=seed + index, device=teacher.device)
        turned = teacher @ rotation

        # Eq 116: the full-space Gram matrix must survive the rotation exactly.
        sample = slice(0, min(256, teacher.size(0)))
        gram_shift = max(
            gram_shift,
            float(
                (
                    turned[sample] @ turned[sample].t()
                    - teacher[sample] @ teacher[sample].t()
                )
                .abs()
                .max()
            ),
        )
        for dim in usable:
            rotated[dim].append(knn_recall(turned[:, :dim], teacher, k=k))

    return {
        "full_dim_gram_shift": gram_shift,
        "baseline": baseline,
        "rotated": rotated,
        "mean_drop": {
            dim: baseline[dim] - float(np.mean(rotated[dim])) for dim in usable
        },
    }


__all__ = [
    "distortion_gain_correlation",
    "distortion_profile",
    "distortion_reduction",
    "knn_recall",
    "monotonicity_violation_rate",
    "normalized_distortion",
    "price_of_nestedness",
    "random_orthogonal",
    "refinement_gain",
    "rotation_stress_test",
    "sdra",
    "semantic_distortion",
    "similarity_spearman",
    "trustworthiness_and_continuity",
    "zero_rate_distortion",
]
