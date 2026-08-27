#!/usr/bin/env python3
"""Check the SDR-MRL implementation against the derivations it claims to follow.

``docs/latex/main.pdf`` states its case as a chain of propositions rather than as a
loss formula, so the useful debugging question is not "does the loss go down"
but "does the code compute the object the proofs are about". Each check below
recomputes one equation from an independent construction and reports the
residual::

    python scripts/verify_sdr_math.py

Nothing here touches a model or the dataset: the discrete and Gaussian worlds
are built in-script precisely because their exact answers are known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_mrl.geometry import rotation_stress_test  # noqa: E402
from embedding_mrl.losses.sdr import (  # noqa: E402
    _masked_log_softmax,
    candidate_mask,
    divergence_from_log_probs,
    neighborhood_logits,
    rate_prior,
    semantic_neighborhood_distortion,
)

TOL = 1e-5
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
# 1. The distributions themselves (Eq 20-25)
# --------------------------------------------------------------------------- #
def check_distributions() -> None:
    section("Semantic neighborhood distributions (Eq 20-25)")
    torch.manual_seed(0)
    z = torch.randn(24, 16)

    logits = neighborhood_logits(z, 0.05)
    probs = torch.softmax(logits, dim=-1)

    check(
        "Eq 23  p_T rows are probability distributions",
        torch.allclose(probs.sum(-1), torch.ones(24), atol=TOL),
        f"max |sum - 1| = {float((probs.sum(-1) - 1).abs().max()):.2e}",
    )
    check(
        "Eq 20  C_i excludes the anchor itself",
        float(probs.diagonal().abs().max()) == 0.0,
        f"max p_ii = {float(probs.diagonal().abs().max()):.2e}",
    )

    # Restricting the candidate set must renormalise, not merely zero out mass.
    mask = candidate_mask(logits, logits, "teacher_topm", top_m=5)
    restricted = _masked_log_softmax(logits, mask).exp()
    check(
        "Eq 20  top-M candidate sets renormalise to 1",
        torch.allclose(restricted.sum(-1), torch.ones(24), atol=TOL),
        f"support = {float(mask.sum(-1).float().mean()):.1f} of 23",
    )
    check(
        "Eq 20  mass outside C_i is exactly 0",
        float(restricted[~mask].abs().max()) == 0.0,
    )

    # D_k = 0 exactly when the prefix reproduces the teacher's distribution.
    identical = semantic_neighborhood_distortion(z, z, 0.05, 0.05)
    check(
        "Eq 26  D_k = 0 when the prefix equals the teacher",
        abs(float(identical)) < 1e-6,
        f"D = {float(identical):.2e}",
    )
    check(
        "Eq 26  D_k >= 0 for arbitrary prefixes",
        all(
            float(semantic_neighborhood_distortion(torch.randn(24, d), z, 0.05, 0.05)) >= -TOL
            for d in (2, 4, 8)
        ),
    )

    # tau_S != tau_T leaves an irreducible floor: the implementation's own
    # optimum is then not 0, which is worth knowing before blaming the model.
    mismatched = float(semantic_neighborhood_distortion(z, z, 0.05, 0.10))
    check(
        "      tau_S != tau_T puts a floor under the objective",
        mismatched > 1e-3,
        f"D at full width = {mismatched:.4f} (0 when tau_S == tau_T)",
    )


# --------------------------------------------------------------------------- #
# 2. Gradient geometry (Eq 59-65)
# --------------------------------------------------------------------------- #
def check_gradients() -> None:
    section("Gradient geometry (Eq 59-65)")
    torch.manual_seed(1)
    batch, dim, tau = 12, 8, 0.05

    teacher = torch.randn(batch, dim)
    student = torch.randn(batch, dim)

    # -- Eq 59: dD_i / da_ij = q_ij - p_ij ---------------------------------- #
    teacher_logits = neighborhood_logits(teacher, tau)
    student_logits = neighborhood_logits(student, tau).clone().requires_grad_(True)
    mask = candidate_mask(teacher_logits, student_logits, "all")

    log_p = _masked_log_softmax(teacher_logits, mask)
    log_q = _masked_log_softmax(student_logits, mask)
    divergence_from_log_probs(log_p, log_q, "forward_kl").backward()

    # The estimator averages over anchors, hence the 1/B.
    expected = (log_q.exp() - log_p.exp()).detach() / batch
    residual = float((student_logits.grad - expected)[mask].abs().max())
    check("Eq 59  dD/da_ij = q_ij - p_ij", residual < TOL, f"max residual = {residual:.2e}")

    # -- Eq 61: the direct anchor gradient ---------------------------------- #
    unit = F.normalize(torch.randn(batch, dim), dim=-1)
    anchor = unit[0].clone().requires_grad_(True)
    others = unit[1:].detach()

    logits_row = (others @ anchor) / tau
    log_q_row = torch.log_softmax(logits_row, dim=-1)
    p_row = torch.softmax((others @ unit[0].detach()) / tau + torch.randn(batch - 1), dim=-1)
    (-(p_row * log_q_row).sum()).backward()

    q_row = log_q_row.exp().detach()
    closed_form = ((q_row - p_row) @ others) / tau
    residual = float((anchor.grad - closed_form).abs().max())
    check(
        "Eq 61  dD_i/dz_i = (1/tau) sum_j (q_ij - p_ij) z_j",
        residual < TOL,
        f"max residual = {residual:.2e}",
    )

    # -- Eq 65: the same gradient, projected onto the tangent space --------- #
    raw = torch.randn(dim, requires_grad=True)
    normalised = raw / raw.norm()
    logits_row = (others @ normalised) / tau
    log_q_row = torch.log_softmax(logits_row, dim=-1)
    (-(p_row * log_q_row).sum()).backward()

    z_i = (raw / raw.norm()).detach()
    q_row = log_q_row.exp().detach()
    inner = ((q_row - p_row) @ others)
    projected = (inner - (inner @ z_i) * z_i) / (tau * float(raw.detach().norm()))

    residual = float((raw.grad - projected).abs().max())
    check(
        "Eq 65  dD_i/dh_i = (I - z z^T) (...) / (tau ||h||)",
        residual < TOL,
        f"max residual = {residual:.2e}",
    )
    check(
        "Eq 65  the gradient is tangent to the unit sphere",
        abs(float(raw.grad @ z_i)) < TOL,
        f"<grad, z_i> = {float(raw.grad @ z_i):.2e}",
    )

    # -- Eq 62-63: the gradient moves one barycenter onto the other --------- #
    student_barycenter = q_row @ others
    teacher_barycenter = p_row @ others
    check(
        "Eq 62-63  gradient direction = student barycenter - teacher barycenter",
        float((inner - (student_barycenter - teacher_barycenter)).abs().max()) < TOL,
    )


# --------------------------------------------------------------------------- #
# 3. Variational decomposition and successive refinement (Prop 1, 2)
# --------------------------------------------------------------------------- #
def _discrete_world(seed: int = 0):
    """An exact S - X - Z Markov chain with nested deterministic quantisers.

    ``Z_1`` is a coarsening of ``Z_2``, which is a coarsening of ``Z_3``: the
    discrete analogue of Matryoshka prefixes (Eq 14-15), where every claim can
    be evaluated in closed form instead of estimated.
    """
    rng = np.random.default_rng(seed)
    n_x, n_s = 12, 5

    p_x = rng.dirichlet(np.ones(n_x))
    p_s_given_x = rng.dirichlet(np.ones(n_s) * 0.4, size=n_x)  # sharp, informative

    # Each level must be a *coarsening* of the next, or Eq 14-15's hierarchy
    # X -> Z_K -> ... -> Z_1 does not hold and Eq 45 is not even well posed.
    partitions = {
        3: np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),  # 6 cells
        2: np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),  # {0,1} {2,3} {4,5}
        1: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),  # {0,1} {2}
    }
    for coarse, fine in ((1, 2), (2, 3)):
        for cell in np.unique(partitions[fine]):
            members = partitions[coarse][partitions[fine] == cell]
            assert len(np.unique(members)) == 1, (
                f"Z_{fine} is not a refinement of Z_{coarse}; the test world is broken"
            )
    return p_x, p_s_given_x, partitions


def _posterior_given_z(p_x, p_s_given_x, assignment):
    """``p(z)`` and ``p_T(S|z)`` - the optimal decoder of Eq 40."""
    cells = np.unique(assignment)
    p_z = np.array([p_x[assignment == c].sum() for c in cells])
    p_s_given_z = np.stack(
        [(p_x[assignment == c, None] * p_s_given_x[assignment == c]).sum(0) / p_z[i]
         for i, c in enumerate(cells)]
    )
    return p_z, p_s_given_z


def _kl(p, q):
    mask = p > 0
    return float((p[mask] * (np.log(p[mask]) - np.log(q[mask]))).sum())


def _expected_distortion(p_x, p_s_given_x, assignment, decoder):
    """``D_k(q) = E_x KL(p_T(S|x) || q(S|z_k(x)))`` (Eq 26)."""
    return float(
        sum(p_x[x] * _kl(p_s_given_x[x], decoder[assignment[x]]) for x in range(len(p_x)))
    )


def _conditional_mi(p_x, p_s_given_x, assignment):
    """``I(S; X | Z) = H(S|Z) - H(S|X)``, exact."""
    def entropy(p):
        mask = p > 0
        return float(-(p[mask] * np.log(p[mask])).sum())

    h_s_given_x = sum(p_x[x] * entropy(p_s_given_x[x]) for x in range(len(p_x)))
    p_z, p_s_given_z = _posterior_given_z(p_x, p_s_given_x, assignment)
    h_s_given_z = sum(p_z[i] * entropy(p_s_given_z[i]) for i in range(len(p_z)))
    return float(h_s_given_z - h_s_given_x)


def check_propositions() -> None:
    section("Variational decomposition and successive refinement (Prop 1-2)")
    p_x, p_s_given_x, partitions = _discrete_world()
    rng = np.random.default_rng(7)

    optimal = {}
    for level, assignment in sorted(partitions.items()):
        _, decoder = _posterior_given_z(p_x, p_s_given_x, assignment)
        optimal[level] = _expected_distortion(p_x, p_s_given_x, assignment, decoder)

        mutual_information = _conditional_mi(p_x, p_s_given_x, assignment)
        check(
            f"Eq 39  D*_{level} = I_T(S; X | Z_{level})",
            abs(optimal[level] - mutual_information) < 1e-9,
            f"D* = {optimal[level]:.6f}, I = {mutual_information:.6f}",
        )

    # Eq 30-32: any other decoder pays exactly the extra term epsilon_k >= 0.
    assignment = partitions[2]
    p_z, posterior = _posterior_given_z(p_x, p_s_given_x, assignment)
    perturbed = rng.dirichlet(np.ones(p_s_given_x.shape[1]), size=posterior.shape[0])

    actual = _expected_distortion(p_x, p_s_given_x, assignment, perturbed)
    epsilon = sum(p_z[i] * _kl(posterior[i], perturbed[i]) for i in range(len(p_z)))
    check(
        "Eq 30  D_k = I_T(S; X | Z_k) + epsilon_k",
        abs(actual - (optimal[2] + epsilon)) < 1e-9,
        f"D = {actual:.6f} = {optimal[2]:.6f} + {epsilon:.6f}",
    )
    check("Eq 31  epsilon_k >= 0", epsilon >= 0, f"epsilon = {epsilon:.6f}")
    check(
        "Eq 32  D_k >= I_T(S; X | Z_k) for every decoder",
        all(
            _expected_distortion(
                p_x, p_s_given_x, assignment,
                rng.dirichlet(np.ones(p_s_given_x.shape[1]), size=posterior.shape[0]),
            ) >= optimal[2] - 1e-12
            for _ in range(200)
        ),
        "200 random decoders, none beat the posterior",
    )

    # Eq 42: coarser quantiser, never lower optimal distortion.
    check(
        "Eq 42  D*_k <= D*_{k-1} along the nested hierarchy",
        optimal[1] >= optimal[2] >= optimal[3] - 1e-12,
        f"D*_1={optimal[1]:.6f} >= D*_2={optimal[2]:.6f} >= D*_3={optimal[3]:.6f}",
    )

    # Eq 43/45: the drop is exactly the new block's conditional information.
    for coarse, fine in ((1, 2), (2, 3)):
        joint = partitions[coarse] * 100 + partitions[fine]  # Z_k = (Z_{k-1}, B_k)
        gain = _conditional_mi(p_x, p_s_given_x, partitions[coarse]) - _conditional_mi(
            p_x, p_s_given_x, joint
        )
        check(
            f"Eq 45  D*_{coarse} - D*_{fine} = I_T(S; B_{fine} | Z_{coarse})",
            abs((optimal[coarse] - optimal[fine]) - gain) < 1e-9,
            f"drop = {optimal[coarse] - optimal[fine]:.6f}, I = {gain:.6f}",
        )


# --------------------------------------------------------------------------- #
# 4. Stochastic rate sampling (Eq 66-68)
# --------------------------------------------------------------------------- #
def check_rate_sampling() -> None:
    section("Stochastic rate sampling (Eq 66-68)")
    dims = [16, 32, 64, 128, 256, 512]

    for kind in ("uniform", "inverse_dim"):
        prior = rate_prior(dims, kind)
        check(
            f"Eq 47  pi ({kind}) is a probability vector",
            abs(sum(prior) - 1.0) < TOL and all(w >= 0 for w in prior),
            f"pi = {[round(w, 4) for w in prior]}",
        )
    check(
        "Eq 50  the low-rate prior really favours small prefixes",
        rate_prior(dims, "inverse_dim")[0] > rate_prior(dims, "inverse_dim")[-1],
    )

    # Eq 68: E_{k ~ pi}[D_k] == sum_k pi_k D_k.
    prior = rate_prior(dims, "inverse_dim")
    distortions = torch.tensor([2.0, 1.4, 1.0, 0.7, 0.4, 0.1])
    exact = float((torch.tensor(prior) * distortions).sum())

    generator = torch.Generator().manual_seed(0)
    draws = torch.multinomial(
        torch.tensor(prior), 200_000, replacement=True, generator=generator
    )
    estimate = float(distortions[draws].mean())
    check(
        "Eq 68  sampled D_k is unbiased for sum_k pi_k D_k",
        abs(estimate - exact) < 5e-3,
        f"MC = {estimate:.5f}, exact = {exact:.5f}",
    )


# --------------------------------------------------------------------------- #
# 5. Linear-Gaussian nested optimum (Theorem 1, Eq 70-84)
# --------------------------------------------------------------------------- #
def check_linear_gaussian() -> None:
    section("Linear-Gaussian nested optimum (Theorem 1, Eq 70-84)")
    rng = np.random.default_rng(3)
    p, q = 8, 3

    root = rng.normal(size=(p, p))
    sigma_x = root @ root.T + 0.5 * np.eye(p)
    mixing = rng.normal(size=(q, p))
    sigma_noise = 0.3 * np.eye(q)

    sigma_sx = mixing @ sigma_x
    sigma_xs = sigma_sx.T
    sigma_s = mixing @ sigma_x @ mixing.T + sigma_noise

    eigenvalues, eigenvectors = np.linalg.eigh(sigma_x)
    inv_sqrt = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    sqrt = eigenvectors @ np.diag(eigenvalues ** 0.5) @ eigenvectors.T

    m_matrix = inv_sqrt @ sigma_xs @ sigma_sx @ inv_sqrt  # Eq 76
    spectrum, basis = np.linalg.eigh(m_matrix)
    order = np.argsort(spectrum)[::-1]
    spectrum, basis = spectrum[order], basis[:, order]

    def distortion(v):
        """Eq 75 via the MMSE decoder, computed without reference to Eq 82."""
        cov_sz = sigma_sx @ inv_sqrt @ v
        cov_z = v.T @ v
        return float(np.trace(sigma_s) - np.trace(cov_sz @ np.linalg.inv(cov_z) @ cov_sz.T))

    check(
        "Eq 75  D(V) = tr(Sigma_S) - tr(V^T M V) for orthonormal V",
        _matches_trace_form(distortion, m_matrix, sigma_s, p, rng),
        "MMSE decoder vs the closed form, at d = 1, 2, 4, 6",
    )

    optimal = {}
    for d in range(1, p + 1):
        v_star = basis[:, :d]
        optimal[d] = distortion(v_star)

        closed_form = float(np.trace(sigma_s) - spectrum[:d].sum())  # Eq 82
        if abs(optimal[d] - closed_form) > 1e-6:
            check(f"Eq 82  D*_{d} = tr(Sigma_S) - sum_j<=d lambda_j", False,
                  f"{optimal[d]:.6f} vs {closed_form:.6f}")
            break
    else:
        check(
            "Eq 82  D*_d = tr(Sigma_S) - sum_{j<=d} lambda_j for every d",
            True,
            f"D*_1 = {optimal[1]:.4f} ... D*_{p} = {optimal[p]:.4f}",
        )

    check(
        "Eq 83  the d-th coordinate's semantic gain is lambda_d",
        all(abs((optimal[d - 1] - optimal[d]) - spectrum[d - 1]) < 1e-6 for d in range(2, p + 1)),
        f"lambda = {np.round(spectrum[:4], 4)} ...",
    )

    # Ky Fan: no orthonormal V_d beats the top-d eigenvectors (Eq 81).
    beaten = 0
    for d in (2, 4):
        for _ in range(300):
            candidate = np.linalg.qr(rng.normal(size=(p, d)))[0]
            if distortion(candidate) < optimal[d] - 1e-9:
                beaten += 1
    check("Eq 79/81  top-d eigenvectors of M are optimal", beaten == 0,
          f"600 random subspaces, {beaten} beat V*")

    check(
        "Eq 80  span(V*_1) subset span(V*_2) subset ... (Matryoshka ordering)",
        all(
            np.allclose(
                basis[:, :d] @ (basis[:, :d].T @ basis[:, : d - 1]), basis[:, : d - 1], atol=1e-8
            )
            for d in range(2, p + 1)
        ),
    )

    # Eq 84: relevance is Sigma_XS Sigma_SX, not variance. PCA is not optimal.
    variance_order = np.argsort(eigenvalues)[::-1]
    for d in (2, 3):
        pca_subspace = np.linalg.qr(sqrt @ eigenvectors[:, variance_order[:d]])[0]
        check(
            f"Eq 84  PCA directions are suboptimal at d={d}",
            distortion(pca_subspace) > optimal[d] + 1e-8,
            f"D(PCA) = {distortion(pca_subspace):.4f} > D* = {optimal[d]:.4f}",
        )


def _matches_trace_form(distortion, m_matrix, sigma_s, p, rng) -> bool:
    for d in (1, 2, 4, 6):
        v = np.linalg.qr(rng.normal(size=(p, d)))[0]
        if abs(distortion(v) - (np.trace(sigma_s) - np.trace(v.T @ m_matrix @ v))) > 1e-6:
            return False
    return True


# --------------------------------------------------------------------------- #
# 6. The rotation argument (Eq 114-118)
# --------------------------------------------------------------------------- #
def check_rotation() -> None:
    section("Rotation stress test (Eq 114-118)")
    torch.manual_seed(5)
    embeddings = torch.randn(400, 64)
    embeddings[:, :8] *= 4.0  # semantic mass deliberately placed in the prefix

    report = rotation_stress_test(embeddings, [4, 8, 16, 32], num_rotations=3, k=10)
    check(
        "Eq 116  Z'Z'^T = ZZ^T: full-space geometry is rotation-invariant",
        report["full_dim_gram_shift"] < 1e-4,
        f"max |shift| = {report['full_dim_gram_shift']:.2e}",
    )
    drops = report["mean_drop"]
    check(
        "Eq 117-118  prefix quality is not rotation-invariant",
        all(value > 0.01 for value in drops.values()),
        "kNN recall drop " + ", ".join(f"d={d}: {v:+.3f}" for d, v in drops.items()),
    )

    rotated = semantic_neighborhood_distortion(
        (embeddings @ torch.linalg.qr(torch.randn(64, 64))[0])[:, :8], embeddings, 0.05, 0.05
    )
    original = semantic_neighborhood_distortion(embeddings[:, :8], embeddings, 0.05, 0.05)
    check(
        "      the same is visible in D_k, which is the point of the method",
        float(rotated) > float(original),
        f"D(rotated prefix) = {float(rotated):.4f} > D(prefix) = {float(original):.4f}",
    )


# --------------------------------------------------------------------------- #
# 7. The decoder temperature (Eq 24) and the practical gap epsilon_k (Eq 31)
# --------------------------------------------------------------------------- #
def check_decoder_temperature() -> None:
    section("Decoder temperature (Eq 24) and the practical gap (Eq 31)")
    torch.manual_seed(6)
    # A teacher whose semantic mass decays along the coordinates, i.e. what a
    # trained Matryoshka code is supposed to look like.
    teacher = torch.randn(256, 128) * torch.linspace(3.0, 0.3, 128)
    tau_t = 0.05
    grid = [0.05, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.3, 0.4]

    best = {}
    slack = {}
    for d in (8, 32, 96):
        values = {
            tau: float(semantic_neighborhood_distortion(teacher[:, :d], teacher, tau_t, tau))
            for tau in grid
        }
        best[d] = min(values, key=values.get)
        slack[d] = values[tau_t] - values[best[d]]

    check(
        "Eq 24  the optimal decoder is flatter than the teacher at low rate (tau_k > tau_T)",
        best[8] > tau_t,
        "argmin tau_k: " + ", ".join(f"d={d}: {t:.2f}" for d, t in best.items()),
    )
    check(
        "Eq 24  tau_k* decreases toward tau_T as the prefix grows",
        best[8] >= best[32] >= best[96],
    )
    check(
        "Eq 31  tying tau_k = tau_T leaves a removable part of epsilon_k on the table",
        slack[8] > 0.05 * max(slack[8], 1e-9) and slack[8] > 0,
        "D(tau_T) - min_tau D: " + ", ".join(f"d={d}: {s:.4f}" for d, s in slack.items()),
    )

    # A learnable tau_k is the infimum over the decoder family: it can only go down.
    from embedding_mrl.losses.sdr import SemanticDistortionLoss

    loss = SemanticDistortionLoss([8, 32, 128], 128, tau_t, tau_t, learnable_temperature=True)
    before = {d: float(v) for d, v in loss(teacher, teacher)["distortions"].items()}
    optimiser = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(80):
        optimiser.zero_grad()
        loss(teacher, teacher)["sem_loss"].backward()
        optimiser.step()
    after = {d: float(v) for d, v in loss(teacher, teacher)["distortions"].items()}
    check(
        "Eq 32  a learnable tau_k never raises D_k above the tied value",
        all(after[d] <= before[d] + 1e-6 for d in before),
        "D_8: " + f"{before[8]:.4f} -> {after[8]:.4f}, tau_8 = {loss.student_temperatures[8]:.3f}",
    )


# --------------------------------------------------------------------------- #
# 8. The monotonic hinge (Eq 54, Eq 69)
# --------------------------------------------------------------------------- #
def check_monotonic_hinge() -> None:
    section("Monotonic refinement hinge (Eq 54, Eq 69)")
    from embedding_mrl.losses.sdr import SemanticDistortionLoss

    lower = torch.tensor(0.2, requires_grad=True)
    upper = torch.tensor(0.5, requires_grad=True)
    loss = SemanticDistortionLoss([4, 8, 16], 16, lambda_mono=1.0)
    loss.monotonic_penalty({4: lower, 8: upper}).backward()
    check(
        "Eq 54  d hinge / d D_k = 1 on a violated edge",
        float(upper.grad) == 1.0,
    )
    check(
        "Eq 54  d hinge / d D_{k-1} = 0: the lower prefix is a stop-gradient target",
        lower.grad is None or float(lower.grad) == 0.0,
        "a naive hinge has -1 here and is minimised by degrading the smaller prefix",
    )

    torch.manual_seed(8)
    teacher = torch.randn(48, 16)
    student = teacher.clone()
    student[:, 4:8] = 6.0 * torch.randn(48, 4)  # break the middle block -> violations
    full = SemanticDistortionLoss([4, 8, 16], 16, lambda_mono=1.0, rate_prior_kind="inverse_dim")
    exact = float(full(student, teacher)["mono_loss"])
    sampled = SemanticDistortionLoss(
        [4, 8, 16], 16, lambda_mono=1.0, rate_prior_kind="inverse_dim", stochastic_rate=True
    )
    expectation = sum(
        w * float(sampled(student, teacher, rate_index=k)["mono_loss"])
        for k, w in enumerate(sampled.rate_prior)
    )
    check(
        "Eq 69  E_{k~pi}[hinge_k / pi_k] = sum_k hinge_k",
        exact > 0 and abs(expectation - exact) < 1e-5,
        f"sum = {exact:.5f}, E = {expectation:.5f}",
    )


# --------------------------------------------------------------------------- #
# 9. What the online self-teacher does *not* rule out (Sec 4.9)
# --------------------------------------------------------------------------- #
def check_tail_inert_minimiser() -> None:
    section("Online self-teacher: tail-inert minimisers (Sec 4.9)")
    from embedding_mrl.losses.infonce import matryoshka_info_nce
    from embedding_mrl.losses.sdr import SemanticDistortionLoss

    torch.manual_seed(9)
    batch, full = 16, 128
    dims = [8, 16, 32, 64, 128]
    loss = SemanticDistortionLoss(dims, full, 0.05, 0.05)
    prefix = F.normalize(torch.randn(batch, 8), dim=-1)
    tail = torch.randn(batch, full - 8) / (full - 8) ** 0.5

    rows = []
    for scale in (1.0, 0.3, 0.0):
        h = torch.cat([prefix, scale * tail], dim=1)
        view = h * (1 + 0.3 * torch.randn_like(h))  # dropout-like second view
        task = float(matryoshka_info_nce(h, view, dims, temperature=0.05)[0])
        sem = float(loss(h, h.detach())["sem_loss"])
        rows.append((scale, task, sem))

    check(
        "Sec 4.9  L_sem -> 0 as the tail goes inert while L_task is already saturated",
        rows[-1][2] < 1e-6 and rows[0][2] > rows[-1][2] and abs(rows[0][1] - rows[-1][1]) < 0.5,
        "; ".join(f"tail x{s:g}: L_task={t:.3f}, L_sem={m:.4f}" for s, t, m in rows)
        + "  -> use an EMA/frozen teacher or watch norm_share",
    )


def main() -> int:
    print(__doc__.strip().split("\n")[0])
    print("=" * 74)

    check_distributions()
    check_gradients()
    check_propositions()
    check_rate_sampling()
    check_linear_gaussian()
    check_rotation()
    check_decoder_temperature()
    check_monotonic_hinge()
    check_tail_inert_minimiser()

    failures = [name for name, passed, _ in _RESULTS if not passed]
    print("\n" + "=" * 74)
    print(f"{len(_RESULTS) - len(failures)}/{len(_RESULTS)} checks passed")
    for name in failures:
        print(f"  FAILED: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
