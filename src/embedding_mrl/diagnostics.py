"""Training-time instrumentation for SDR-MRL's mathematical claims.

The training loss is a single scalar, which hides everything that actually tells
you whether the method is working *for the reason the paper says it does*. This
module recomputes, per batch, the quantities the derivations are written in
terms of - so a run can be debugged against the maths rather than against the
loss curve.

What each number is for
-----------------------

``teacher/entropy`` (Eq 112)
    ``H(p_T)`` in nats. Sec 8.7 sweeps ``tau_T`` precisely because this is the
    knob that decides whether the semantic variable carries any signal:
    ``H -> 0`` degenerates to hard-neighbor supervision, ``H -> log(B-1)`` makes
    the teacher uniform and ``D_k`` uninformative for every prefix.

``teacher/perplexity``
    ``exp(H)``: the effective number of semantic neighbors per anchor. The
    readable form of the entropy - "the teacher is really pointing at ~4 of the
    15 candidates".

``teacher/mean_cosine``
    Collapse detector for the failure mode of Sec 4.9. A teacher whose points
    all coincide has a trivially reproducible neighborhood, so ``D_k -> 0``
    while the representation is worthless. If this climbs towards 1 the semantic
    term is being satisfied by collapse, not by information ordering.

``zero_rate_distortion`` (Eq 88)
    ``D_0 = log(B-1) - H(p_T)``: what a decoder that ignores the embedding
    entirely pays. It is the yardstick, so ``D_k / D_0 > 1`` means the prefix is
    *worse than knowing nothing* - a fact the raw ``D_k`` never makes obvious.

``dim_*/distortion`` and ``dim_*/marginal_gain`` (Eq 45)
    The in-batch distortion-rate profile and its differences. At an optimal
    decoder ``D_{k-1} - D_k = I_T(S; B_k | Z_{k-1})``, so the gains are the
    per-block conditional semantic information - the successive-refinement
    reading of the method.

``dim_*/eta`` (Eq 122)
    Gain per *added coordinate*. Where a model chooses to spend its capacity;
    MRL, ESE, MIPIC and SDR-MRL should differ here even at equal accuracy.

``dim_*/barycenter_gap`` (Eq 61-63)
    ``|| sum_j (q_ij - p_ij) z_j ||``: the gradient of the semantic loss up to
    ``1/tau``. Geometrically it is the distance between the student's local
    semantic barycenter and the teacher-weighted one, i.e. how far this anchor
    still has to travel on the sphere. It should shrink as training proceeds.

``full_dim_distortion``
    ``D_K``, which must be ~0 when ``tau_S == tau_T`` because the student and
    the teacher are then the same distribution. Any other value is an
    irreducible floor under the objective and almost always means mismatched
    temperatures rather than a modelling choice.

``monotonicity_violation_rate`` (Eq 108)
    Prop 2 guarantees ``D_k <= D_{k-1}`` only for the *optimal* decoder; the
    practical cosine decoder can break it. This is the quantity ``lambda_mono``
    exists to reduce, and the evidence for whether it is needed at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .losses.sdr import (
    _masked_log_softmax,
    candidate_mask,
    divergence_from_log_probs,
    neighborhood_logits,
)


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """``H(p)`` per row, in nats. Masked entries have ``p = 0`` and drop out."""
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@torch.no_grad()
def batch_knn_recall(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor, k: int
) -> float:
    """Eq 96 restricted to the batch: top-``k`` overlap, teacher versus prefix.

    A rank-based counterpart to ``D_k``. Distortion can be lowered by matching
    the teacher's similarity *scale*; recall only moves when the actual
    neighbor ordering improves, so the two disagreeing is itself informative.
    """
    k = max(1, min(k, teacher_logits.size(0) - 1))
    teacher_top = teacher_logits.topk(k, dim=-1).indices
    student_top = student_logits.topk(k, dim=-1).indices

    overlap = sum(
        len(set(teacher_top[row].tolist()) & set(student_top[row].tolist()))
        for row in range(teacher_top.size(0))
    )
    return overlap / (teacher_top.size(0) * k)


@torch.no_grad()
def semantic_barycenter_gap(
    probs_teacher: torch.Tensor, probs_student: torch.Tensor, student_z: torch.Tensor
) -> float:
    """Eq 61-63: ``mean_i || sum_j (q_ij - p_ij) z_j ||``.

    Eq 61 makes this the anchor gradient up to the ``1/tau_k`` factor, and
    Eq 62-63 give it its meaning: the student's semantic barycenter is being
    pulled onto the teacher-weighted one.
    """
    return float(((probs_student - probs_teacher) @ student_z).norm(dim=-1).mean())


@torch.no_grad()
def neighborhood_diagnostics(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    dims: Sequence[int],
    teacher_temperature: float = 0.05,
    student_temperature: float = 0.05,
    divergence: str = "forward_kl",
    candidates: str = "all",
    top_m: int = 32,
    knn_k: int = 5,
) -> dict[str, object]:
    """Every quantity above, for one batch.

    Args:
        student_repr: ``[B, D]`` pooled student embeddings.
        teacher_repr: ``[B, D_T]`` the teacher that training actually used, so
            the numbers here are comparable to the logged loss.
        dims: nested dimensions to profile; the full width is included so the
            ``D_K ~ 0`` sanity check is always available.
    """
    batch_size = student_repr.size(0)
    if batch_size < 3:
        return {}

    teacher_logits = neighborhood_logits(teacher_repr, teacher_temperature)
    full_student_logits = neighborhood_logits(student_repr, student_temperature)
    mask = candidate_mask(teacher_logits, full_student_logits, candidates, top_m)

    log_p = _masked_log_softmax(teacher_logits, mask)
    probs_teacher = log_p.exp()
    teacher_entropy = float(_entropy(log_p).mean())

    # Eq 88 with q_0 uniform over the retained candidates.
    support = float(mask.sum(dim=-1).float().mean())
    zero_rate = float(torch.log(torch.tensor(max(support, 1.0)))) - teacher_entropy

    teacher_unit = F.normalize(teacher_repr.float(), dim=-1)
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=teacher_unit.device)
    mean_cosine = float((teacher_unit @ teacher_unit.t())[off_diagonal].mean())

    per_dim: dict[str, dict[str, float]] = {}
    profile: dict[int, float] = {}

    for dim in sorted({int(d) for d in dims if d <= student_repr.size(1)}):
        prefix = F.normalize(student_repr[:, :dim].float(), dim=-1)
        student_logits = neighborhood_logits(prefix, student_temperature)
        log_q = _masked_log_softmax(student_logits, mask)

        distortion = float(divergence_from_log_probs(log_p, log_q, divergence))
        profile[dim] = distortion
        per_dim[f"dim_{dim}"] = {
            "distortion": distortion,
            # >1 means the prefix is worse than the zero-rate uniform decoder.
            "distortion_vs_zero_rate": distortion / zero_rate if zero_rate > 0 else float("nan"),
            "student_entropy": float(_entropy(log_q).mean()),
            "knn_recall": batch_knn_recall(teacher_logits, student_logits, knn_k),
            "barycenter_gap": semantic_barycenter_gap(probs_teacher, log_q.exp(), prefix),
        }

    ordered = sorted(profile)
    for previous, current in zip(ordered[:-1], ordered[1:]):
        gain = profile[previous] - profile[current]
        per_dim[f"dim_{current}"]["marginal_gain"] = gain          # Eq 45
        per_dim[f"dim_{current}"]["eta"] = gain / (current - previous)  # Eq 122

    truncated = ordered[:-1] if len(ordered) > 1 else ordered
    edges = list(zip(truncated[:-1], truncated[1:]))
    violations = sum(1 for a, b in edges if profile[b] > profile[a])

    return {
        "batch_size": batch_size,
        "candidate_support": support,
        "teacher": {
            "entropy": teacher_entropy,
            "perplexity": float(torch.exp(torch.tensor(teacher_entropy))),
            "max_prob": float(probs_teacher.max(dim=-1).values.mean()),
            "mean_cosine": mean_cosine,
        },
        "zero_rate_distortion": zero_rate,
        "full_dim_distortion": profile[ordered[-1]],
        "monotonicity_violation_rate": violations / len(edges) if edges else 0.0,
        "per_dim": per_dim,
    }


def format_diagnostics(record: dict[str, object], knn_k: int = 5) -> str:
    """A compact, readable table for the training log."""
    if not record:
        return "diagnostics unavailable (batch too small)"

    teacher = record["teacher"]
    header = (
        f"teacher: H={teacher['entropy']:.3f} nats "
        f"(perplexity {teacher['perplexity']:.1f} of {record['candidate_support']:.0f} "
        f"candidates, max p={teacher['max_prob']:.3f}, mean cos={teacher['mean_cosine']:+.3f})\n"
        f"D_0={record['zero_rate_distortion']:.3f}  "
        f"D_full={record['full_dim_distortion']:.2e}  "
        f"V_mono={record['monotonicity_violation_rate']:.2f}"
    )

    columns = f"{'dim':>6} | {'D_k':>8} | {'D_k/D_0':>8} | {'gain':>8} | {'eta*1e3':>8} | {'kNN@' + str(knn_k):>8} | {'H(q_k)':>7} | {'|dbary|':>8}"
    lines = [header, "", columns, "-" * len(columns)]

    for dim, scores in record["per_dim"].items():
        gain = scores.get("marginal_gain")
        eta = scores.get("eta")
        lines.append(
            f"{dim.split('_')[1]:>6} | {scores['distortion']:>8.4f} | "
            f"{scores['distortion_vs_zero_rate']:>8.3f} | "
            f"{'-' if gain is None else format(gain, '.4f'):>8} | "
            f"{'-' if eta is None else format(eta * 1e3, '.3f'):>8} | "
            f"{scores['knn_recall']:>8.3f} | {scores['student_entropy']:>7.3f} | "
            f"{scores['barycenter_gap']:>8.4f}"
        )
    return "\n".join(lines)


__all__ = [
    "batch_knn_recall",
    "format_diagnostics",
    "neighborhood_diagnostics",
    "semantic_barycenter_gap",
]
