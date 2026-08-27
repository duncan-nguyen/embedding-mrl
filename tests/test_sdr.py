"""SDR-MRL: the properties the derivation claims, checked directly.

``scripts/verify_sdr_math.py`` covers the propositions end to end against
independently constructed worlds; this file pins the same behaviour at unit
level so a refactor cannot quietly break it.
"""

import math

import pytest
import torch

from embedding_mrl import geometry
from embedding_mrl.diagnostics import neighborhood_diagnostics
from embedding_mrl.losses.sdr import (
    SemanticDistortionLoss,
    _masked_log_softmax,
    candidate_mask,
    divergence_from_log_probs,
    gram_mse_distortion,
    hard_neighbor_cross_entropy,
    neighborhood_logits,
    rate_prior,
    semantic_neighborhood_distortion,
)

B, D = 24, 32
DIMS = [4, 8, 16, 32]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def structured(batch: int = B, dim: int = D) -> torch.Tensor:
    """An embedding whose semantic mass really does sit in the early coordinates."""
    h = torch.randn(batch, dim)
    h[:, :8] *= 4.0
    return h


# --------------------------------------------------------------------------- #
# The neighborhood distributions (Eq 20-25)
# --------------------------------------------------------------------------- #
def test_the_anchor_is_never_its_own_semantic_neighbour():
    probs = torch.softmax(neighborhood_logits(torch.randn(B, D), 0.05), dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    assert float(probs.diagonal().abs().max()) == 0.0


def test_temperature_controls_how_peaked_the_teacher_is():
    z = torch.randn(B, D)
    sharp = torch.softmax(neighborhood_logits(z, 0.01), dim=-1)
    flat = torch.softmax(neighborhood_logits(z, 1.0), dim=-1)
    assert float(sharp.max(-1).values.mean()) > float(flat.max(-1).values.mean())


def test_restricting_the_candidate_set_renormalises_it():
    z = torch.randn(B, D)
    logits = neighborhood_logits(z, 0.05)
    mask = candidate_mask(logits, logits, "teacher_topm", top_m=5)

    probs = _masked_log_softmax(logits, mask).exp()
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    assert float(probs[~mask].abs().max()) == 0.0
    assert int(mask.sum(-1).min()) == 5


def test_student_hard_negatives_only_ever_widen_the_candidate_set():
    teacher_logits = neighborhood_logits(torch.randn(B, D), 0.05)
    student_logits = neighborhood_logits(torch.randn(B, D), 0.05)

    teacher_only = candidate_mask(teacher_logits, student_logits, "teacher_topm", 4)
    with_hard = candidate_mask(
        teacher_logits, student_logits, "teacher_topm_student_hard", 4
    )
    assert bool((with_hard | teacher_only == with_hard).all())
    assert int(with_hard.sum()) >= int(teacher_only.sum())


# --------------------------------------------------------------------------- #
# Semantic neighborhood distortion (Eq 26-27)
# --------------------------------------------------------------------------- #
def test_distortion_vanishes_exactly_when_the_prefix_matches_the_teacher():
    z = structured()
    assert abs(float(semantic_neighborhood_distortion(z, z, 0.05, 0.05))) < 1e-6


def test_distortion_is_non_negative_and_falls_as_the_prefix_grows():
    teacher = structured()
    profile = [
        float(semantic_neighborhood_distortion(teacher[:, :d], teacher, 0.05, 0.05))
        for d in DIMS
    ]
    assert all(value >= -1e-6 for value in profile)
    assert profile == sorted(profile, reverse=True), profile


def test_mismatched_temperatures_leave_an_irreducible_floor():
    """tau_S != tau_T means even a perfect prefix cannot reach D = 0."""
    z = structured()
    assert float(semantic_neighborhood_distortion(z, z, 0.05, 0.20)) > 1e-3


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
def test_every_divergence_is_a_non_negative_differentiable_scalar(divergence):
    teacher = structured()
    student = torch.randn(B, 8, requires_grad=True)

    value = semantic_neighborhood_distortion(
        student, teacher, 0.05, 0.05, divergence=divergence
    )
    value.backward()
    assert value.ndim == 0 and float(value.detach()) >= 0
    assert torch.isfinite(student.grad).all()


def test_jensen_shannon_is_symmetric_and_bounded_by_log_two():
    logits_a = neighborhood_logits(torch.randn(B, D), 0.05)
    logits_b = neighborhood_logits(torch.randn(B, D), 0.05)
    mask = candidate_mask(logits_a, logits_b, "all")
    log_a, log_b = _masked_log_softmax(logits_a, mask), _masked_log_softmax(logits_b, mask)

    forward = divergence_from_log_probs(log_a, log_b, "js")
    backward = divergence_from_log_probs(log_b, log_a, "js")
    assert float(forward) == pytest.approx(float(backward), abs=1e-6)
    assert 0.0 <= float(forward) <= math.log(2) + 1e-6


def test_masked_candidates_contribute_no_gradient():
    """0 * log 0 must stay 0 in the backward pass, not become NaN."""
    teacher = structured()
    student = torch.randn(B, 8, requires_grad=True)
    semantic_neighborhood_distortion(
        student, teacher, 0.05, 0.05, candidates="teacher_topm", top_m=2
    ).backward()
    assert torch.isfinite(student.grad).all()


def test_the_logit_gradient_is_q_minus_p():
    """Eq 59 - the identity every geometric claim in Sec 4.12 rests on."""
    teacher_logits = neighborhood_logits(structured(), 0.05)
    student_logits = neighborhood_logits(torch.randn(B, D), 0.05).clone().requires_grad_(True)
    mask = candidate_mask(teacher_logits, student_logits, "all")

    log_p = _masked_log_softmax(teacher_logits, mask)
    log_q = _masked_log_softmax(student_logits, mask)
    divergence_from_log_probs(log_p, log_q, "forward_kl").backward()

    expected = (log_q.exp() - log_p.exp()).detach() / B  # the estimator averages anchors
    assert torch.allclose(student_logits.grad[mask], expected[mask], atol=1e-6)


# --------------------------------------------------------------------------- #
# Alternative geometries (Sec 8.2)
# --------------------------------------------------------------------------- #
def test_the_alternative_geometries_also_vanish_on_a_perfect_prefix():
    z = structured()
    assert float(gram_mse_distortion(z, z)) < 1e-6
    # Hard-neighbor CE cannot reach 0: it is a cross entropy, not a divergence.
    assert float(hard_neighbor_cross_entropy(z, z, 0.05, 0.05)) < float(
        hard_neighbor_cross_entropy(torch.randn(B, D), z, 0.05, 0.05)
    )


# --------------------------------------------------------------------------- #
# The deployment-rate prior (Eq 47-50)
# --------------------------------------------------------------------------- #
def test_uniform_and_low_rate_priors():
    dims = [16, 32, 64]
    assert rate_prior(dims, "uniform") == pytest.approx([1 / 3] * 3)

    low_rate = rate_prior(dims, "inverse_dim")
    assert sum(low_rate) == pytest.approx(1.0)
    assert low_rate == sorted(low_rate, reverse=True)  # Eq 50 favours small prefixes
    assert low_rate[0] == pytest.approx(4 * low_rate[2])  # pi ∝ 1/d


def test_custom_priors_must_be_supplied_and_well_formed():
    with pytest.raises(ValueError, match="requires explicit weights"):
        rate_prior([16, 32], "custom")
    with pytest.raises(ValueError, match="non-negative"):
        rate_prior([16, 32], "custom", [1.0, -1.0])
    assert rate_prior([16, 32], "custom", [3.0, 1.0]) == pytest.approx([0.75, 0.25])


# --------------------------------------------------------------------------- #
# The multi-rate objective (Eq 48, 54)
# --------------------------------------------------------------------------- #
def make_loss(**kwargs) -> SemanticDistortionLoss:
    return SemanticDistortionLoss(dims=DIMS, full_dim=D, **kwargs)


def test_the_full_width_is_the_teacher_and_never_a_student():
    loss = make_loss()
    assert loss.prefix_dims == [4, 8, 16]
    assert D not in loss.prefix_dims


def test_sem_loss_is_the_prior_weighted_sum_of_the_prefix_distortions():
    loss = make_loss(rate_prior_kind="inverse_dim")
    teacher = structured()
    outcome = loss(teacher, teacher.clone())

    expected = sum(
        weight * outcome["distortions"][dim]
        for weight, dim in zip(loss.rate_prior, loss.prefix_dims)
    )
    assert float(outcome["sem_loss"]) == pytest.approx(float(expected), abs=1e-6)


def test_stochastic_sampling_evaluates_one_rate_instead_of_all_of_them():
    loss = make_loss(stochastic_rate=True)
    outcome = loss(structured(), structured(), rate_index=1)

    assert list(outcome["distortions"]) == [loss.prefix_dims[1]]
    assert float(outcome["sem_loss"]) == pytest.approx(
        float(outcome["distortions"][loss.prefix_dims[1]])
    )


def test_a_sampled_monotonic_edge_pulls_in_its_neighbour():
    """Eq 69: the regulariser needs both ends of the edge (k-1, k)."""
    loss = make_loss(stochastic_rate=True, lambda_mono=0.1)
    outcome = loss(structured(), structured(), rate_index=2)
    assert list(outcome["distortions"]) == loss.prefix_dims[1:3]


def test_sampled_rates_follow_the_prior():
    loss = make_loss(rate_prior_kind="inverse_dim", stochastic_rate=True)
    generator = torch.Generator().manual_seed(0)
    draws = [loss.sample_rate_index(generator) for _ in range(4000)]

    for index, weight in enumerate(loss.rate_prior):
        assert draws.count(index) / len(draws) == pytest.approx(weight, abs=0.03)


def test_the_monotonic_penalty_only_fires_on_a_violation():
    loss = make_loss(lambda_mono=1.0)
    monotone = {4: torch.tensor(0.9), 8: torch.tensor(0.5), 16: torch.tensor(0.2)}
    violating = {4: torch.tensor(0.2), 8: torch.tensor(0.5), 16: torch.tensor(0.9)}

    assert float(loss.monotonic_penalty(monotone)) == 0.0
    # (0.5 - 0.2) + (0.9 - 0.5): every violated edge is charged.
    assert float(loss.monotonic_penalty(violating)) == pytest.approx(0.7)


def test_the_penalty_is_inert_when_lambda_mono_is_zero():
    loss = make_loss(lambda_mono=0.0)
    violating = {4: torch.tensor(0.2), 8: torch.tensor(0.9)}
    assert float(loss.monotonic_penalty(violating)) == 0.0


def test_the_objective_backpropagates_into_the_student_only():
    loss = make_loss()
    student = structured().requires_grad_(True)
    teacher = structured().requires_grad_(True)

    outcome = loss(student, teacher)
    (outcome["sem_loss"] + outcome["mono_loss"]).backward()

    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None, "Eq 21 makes the teacher a stop-gradient target"


def test_the_hinge_never_pushes_the_lower_prefix_up():
    """Eq 54 with sg(D_{k-1}): the cheapest way to satisfy [D_k - D_{k-1}]_+ must
    not be to make the *smaller* prefix worse."""
    loss = make_loss(lambda_mono=1.0)
    lower = torch.tensor(0.2, requires_grad=True)
    upper = torch.tensor(0.5, requires_grad=True)

    loss.monotonic_penalty({4: lower, 8: upper}).backward()
    assert lower.grad is None or float(lower.grad) == 0.0
    assert float(upper.grad) == pytest.approx(1.0)


def test_the_sampled_edge_is_unbiased_for_the_full_monotonic_sum():
    """Eq 69: E_{k ~ pi}[ hinge_k / pi_k ] == sum_k hinge_k (Eq 54)."""
    teacher = structured()
    student = teacher.clone()
    # Breaking the middle block of an otherwise perfect student makes D_8 > D_4.
    student[:, 4:8] = 6.0 * torch.randn(B, 4)

    full = make_loss(lambda_mono=1.0, rate_prior_kind="inverse_dim")
    exact = float(full(student, teacher)["mono_loss"])
    assert exact > 0, "the test needs at least one violated edge"

    sampled = make_loss(
        lambda_mono=1.0, rate_prior_kind="inverse_dim", stochastic_rate=True
    )
    expectation = sum(
        weight * float(sampled(student, teacher, rate_index=k)["mono_loss"])
        for k, weight in enumerate(sampled.rate_prior)
    )
    assert expectation == pytest.approx(exact, rel=1e-5)


# --------------------------------------------------------------------------- #
# Decoder temperature (Eq 24)
# --------------------------------------------------------------------------- #
def ordered_teacher(batch: int = 256, dim: int = 64) -> torch.Tensor:
    """Semantic mass decays smoothly along the coordinates, like a trained Matryoshka code."""
    return torch.randn(batch, dim) * torch.linspace(3.0, 0.3, dim)


def best_temperature(prefix: torch.Tensor, teacher: torch.Tensor) -> float:
    grid = [0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.4]
    return min(
        grid,
        key=lambda tau: float(semantic_neighborhood_distortion(prefix, teacher, 0.05, tau)),
    )


def test_the_optimal_decoder_is_flatter_the_smaller_the_prefix():
    """The optimal decoder p_T(S | Z_k) is a posterior average, hence flatter than
    p_T(S | X); the practical tau_k must be allowed above tau_T to express it."""
    teacher = ordered_teacher()
    tau_small = best_temperature(teacher[:, :4], teacher)
    tau_large = best_temperature(teacher[:, :32], teacher)

    assert tau_small > 0.05, "tied temperatures are not the optimal decoder at low rate"
    assert tau_small >= tau_large


def test_a_learnable_temperature_only_ever_tightens_the_bound():
    teacher = ordered_teacher()
    loss = make_loss(learnable_temperature=True)
    assert any(p.requires_grad for p in loss.parameters()), "tau_k must be a parameter"

    before = {d: float(v.detach()) for d, v in loss(teacher, teacher)["distortions"].items()}
    optimiser = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(60):
        optimiser.zero_grad()
        loss(teacher, teacher)["sem_loss"].backward()
        optimiser.step()
    after = {d: float(v.detach()) for d, v in loss(teacher, teacher)["distortions"].items()}

    assert all(after[d] <= before[d] + 1e-6 for d in before), (before, after)
    assert after[4] < before[4]
    temperatures = loss.student_temperatures
    assert temperatures[4] > 0.05, "the smallest prefix should have warmed its decoder"
    assert temperatures[4] >= temperatures[16] - 1e-6


def test_a_fixed_temperature_has_no_parameters_and_stays_put():
    loss = make_loss(learnable_temperature=False)
    assert list(loss.parameters()) == []
    assert loss.student_temperatures == {4: pytest.approx(0.05), 8: pytest.approx(0.05), 16: pytest.approx(0.05)}


def test_temperatures_are_clamped_to_their_bounds():
    loss = make_loss(learnable_temperature=True, temperature_bounds=(0.02, 0.2))
    with torch.no_grad():
        loss.log_tau.fill_(10.0)
    assert all(t == pytest.approx(0.2) for t in loss.student_temperatures.values())
    with pytest.raises(ValueError, match="temperature_bounds"):
        make_loss(student_temperature=0.5, temperature_bounds=(0.01, 0.1))


# --------------------------------------------------------------------------- #
# Memory queue (Sec 4.2)
# --------------------------------------------------------------------------- #
def test_queue_entries_widen_the_candidate_set_and_receive_no_gradient():
    z = structured().requires_grad_(True)
    queue = structured(batch=40).requires_grad_(True)
    logits = neighborhood_logits(z, 0.05, extra=queue)

    assert logits.shape == (B, B + 40)
    probs = torch.softmax(logits, dim=-1)
    assert float(probs.detach()[torch.arange(B), torch.arange(B)].abs().max()) == 0.0
    assert torch.allclose(probs.detach().sum(-1), torch.ones(B), atol=1e-5)
    (probs * torch.randn_like(probs)).sum().backward()
    assert torch.isfinite(z.grad).all()
    assert queue.grad is None, "queue rows are candidates, never anchors"


def test_the_full_width_still_has_zero_distortion_with_a_queue():
    z, queue = structured(), structured(batch=40)
    value = semantic_neighborhood_distortion(
        z, z, 0.05, 0.05, student_extra=queue, teacher_extra=queue
    )
    assert abs(float(value)) < 1e-6

    # The raw estimator wants queue rows at the prefix width; the loss module
    # does that slicing itself (see the test below).
    prefixed = semantic_neighborhood_distortion(
        z[:, :4], z, 0.05, 0.05, student_extra=queue[:, :4], teacher_extra=queue
    )
    assert float(prefixed) > 0
    with pytest.raises(ValueError, match="extra candidates must be"):
        semantic_neighborhood_distortion(
            z[:, :4], z, 0.05, 0.05, student_extra=queue, teacher_extra=queue
        )


def test_queue_sides_must_agree():
    z = structured()
    with pytest.raises(ValueError, match="together"):
        semantic_neighborhood_distortion(z, z, student_extra=structured(batch=5))
    with pytest.raises(ValueError, match="disagree"):
        semantic_neighborhood_distortion(
            z, z, student_extra=structured(batch=5), teacher_extra=structured(batch=6)
        )


def test_the_loss_module_slices_the_queue_to_each_prefix():
    loss = make_loss()
    z, queue = structured(), structured(batch=40)
    outcome = loss(z, z, student_extra=queue, teacher_extra=queue)
    assert set(outcome["distortions"]) == {4, 8, 16}
    assert set(outcome["temperatures"]) == {4, 8, 16}


def test_a_config_without_any_truncated_prefix_is_rejected():
    with pytest.raises(ValueError, match="no truncated prefix"):
        SemanticDistortionLoss(dims=[32], full_dim=32)


# --------------------------------------------------------------------------- #
# Evaluation metrics (Sec 6, Sec 9)
# --------------------------------------------------------------------------- #
def test_the_distortion_profile_bottoms_out_at_the_full_width():
    teacher = structured(batch=120, dim=D)
    profile = geometry.distortion_profile(teacher, teacher, DIMS)

    assert abs(profile[D]) < 1e-6
    assert list(profile.values()) == sorted(profile.values(), reverse=True)


def test_normalised_distortion_places_the_endpoints_at_one_and_zero():
    teacher = structured(batch=120, dim=D)
    profile = geometry.distortion_profile(teacher, teacher, DIMS)
    zero_rate = geometry.zero_rate_distortion(teacher)
    normalized = geometry.normalized_distortion(profile, zero_rate)

    assert normalized[D] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < geometry.sdra(normalized, D) < 1.0


def test_a_prefix_can_be_worse_than_knowing_nothing():
    """D~ > 1 means the prefix loses to the zero-rate uniform decoder (Eq 88-89)."""
    teacher = structured(batch=120, dim=D)
    profile = geometry.distortion_profile(teacher, teacher, [1, D])
    normalized = geometry.normalized_distortion(
        profile, geometry.zero_rate_distortion(teacher)
    )
    assert normalized[1] > 1.0


def test_knn_recall_is_perfect_at_the_full_width_and_worse_when_truncated():
    teacher = structured(batch=120, dim=D)
    assert geometry.knn_recall(teacher, teacher, k=10) == pytest.approx(1.0)
    assert geometry.knn_recall(teacher[:, :4], teacher, k=10) < 0.9


def test_monotonicity_violations_ignore_the_full_width_edge():
    profile = {4: 0.5, 8: 0.9, 16: 0.3, 32: 0.0}
    assert geometry.monotonicity_violation_rate(profile) == pytest.approx(0.5)
    assert geometry.monotonicity_violation_rate(profile, exclude_full=False) == pytest.approx(
        1 / 3
    )


def test_refinement_gain_is_the_distortion_drop_per_added_coordinate():
    gains = geometry.refinement_gain({4: 1.0, 8: 0.6})
    assert gains[8] == pytest.approx((1.0 - 0.6) / 4)


def test_price_of_nestedness_is_positive_when_the_independent_model_wins():
    result = geometry.price_of_nestedness({16: 0.70, 32: 0.80}, {16: 0.74, 32: 0.81})
    assert result["per_dim"] == pytest.approx({16: 0.04, 32: 0.01})
    assert result["aggregate"] == pytest.approx(0.025)

    weighted = geometry.price_of_nestedness(
        {16: 0.70, 32: 0.80}, {16: 0.74, 32: 0.81}, prior={16: 3.0, 32: 1.0}
    )
    assert weighted["aggregate"] == pytest.approx(0.75 * 0.04 + 0.25 * 0.01)


def test_rotation_preserves_the_full_space_but_not_the_prefixes():
    """Eq 114-118 - the observation the whole method is built on."""
    report = geometry.rotation_stress_test(structured(batch=200), DIMS, num_rotations=2, k=10)
    assert report["full_dim_gram_shift"] < 1e-4
    assert all(drop > 0.0 for drop in report["mean_drop"].values())


def test_a_random_orthogonal_matrix_really_is_orthogonal():
    q = geometry.random_orthogonal(16, seed=1)
    assert torch.allclose(q.t() @ q, torch.eye(16), atol=1e-5)


# --------------------------------------------------------------------------- #
# Training diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_report_the_quantities_the_proofs_are_written_in():
    teacher = structured()
    record = neighborhood_diagnostics(teacher, teacher, DIMS, knn_k=5)

    assert abs(record["full_dim_distortion"]) < 1e-6
    assert record["monotonicity_violation_rate"] == 0.0
    assert record["teacher"]["perplexity"] == pytest.approx(
        math.exp(record["teacher"]["entropy"]), rel=1e-4
    )
    # D_0 = log(support) - H(p_T), the Eq 88 identity.
    assert record["zero_rate_distortion"] == pytest.approx(
        math.log(record["candidate_support"]) - record["teacher"]["entropy"], rel=1e-4
    )

    smallest = record["per_dim"]["dim_4"]
    assert smallest["distortion"] > record["per_dim"]["dim_16"]["distortion"]
    assert 0.0 <= smallest["knn_recall"] <= 1.0
    assert record["per_dim"]["dim_32"]["barycenter_gap"] == pytest.approx(0.0, abs=1e-5)


def test_marginal_gains_are_the_successive_distortion_drops():
    teacher = structured()
    per_dim = neighborhood_diagnostics(teacher, teacher, DIMS)["per_dim"]

    assert "marginal_gain" not in per_dim["dim_4"]  # nothing precedes the first prefix
    assert per_dim["dim_8"]["marginal_gain"] == pytest.approx(
        per_dim["dim_4"]["distortion"] - per_dim["dim_8"]["distortion"]
    )
    assert per_dim["dim_8"]["eta"] == pytest.approx(
        per_dim["dim_8"]["marginal_gain"] / 4
    )


def test_diagnostics_decline_to_run_on_a_batch_too_small_to_have_neighbours():
    assert neighborhood_diagnostics(torch.randn(2, D), torch.randn(2, D), DIMS) == {}


def test_diagnostics_report_norm_share_and_per_prefix_temperatures():
    teacher = structured()
    record = neighborhood_diagnostics(
        teacher, teacher, DIMS, student_temperature={4: 0.2, 8: 0.1, 16: 0.07}
    )
    per_dim = record["per_dim"]

    shares = [per_dim[f"dim_{d}"]["norm_share"] for d in DIMS]
    assert shares == sorted(shares) and shares[-1] == pytest.approx(1.0)
    assert per_dim["dim_8"]["norm_share"] > 0.7, "structured() puts its mass in 8 coords"

    assert per_dim["dim_4"]["decoder_temperature"] == 0.2
    # The full width is not in the mapping, so it decodes at tau_T and D_K = 0.
    assert per_dim["dim_32"]["decoder_temperature"] == 0.05
    assert abs(record["full_dim_distortion"]) < 1e-6


def test_diagnostics_use_the_same_queue_the_loss_did():
    teacher, queue = structured(), structured(batch=40)
    record = neighborhood_diagnostics(
        teacher, teacher, DIMS, student_extra=queue, teacher_extra=queue
    )
    assert record["candidate_support"] == pytest.approx(B - 1 + 40)
    assert abs(record["full_dim_distortion"]) < 1e-6


def test_calibrated_distortion_is_never_worse_than_the_tied_decoder():
    teacher = ordered_teacher(batch=200, dim=D)
    for dim in (4, 8):
        tied = geometry.semantic_distortion(teacher[:, :dim], teacher)
        best, tau = geometry.calibrated_semantic_distortion(teacher[:, :dim], teacher)
        assert best <= tied + 1e-9
        assert tau >= 0.05
    profile, temperatures = geometry.calibrated_distortion_profile(teacher, teacher, DIMS)
    assert profile[D] == pytest.approx(0.0, abs=1e-6)
    assert temperatures[4] >= temperatures[16]


# --------------------------------------------------------------------------- #
# Trainer wiring
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

from embedding_mrl.config import ExperimentConfig  # noqa: E402
from embedding_mrl.trainers import build_trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def sdr_config(output_dir: Path, **sdr) -> ExperimentConfig:
    cfg = ExperimentConfig.from_dict(
        {
            "name": "test_sdr",
            "method": "sdr",
            "model": {"name_or_path": "dummy", "hidden_dim": 32, "pooling": "cls"},
            "data": {
                "root": str(REPO_ROOT / "data"),
                "max_length": 32,
                "num_workers": 0,
                "max_train_samples": 32,
            },
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "fp16": False,
                "save_model": False,
                "output_dir": str(output_dir),
            },
            "matryoshka": {"dims": [8, 16, 32]},
            "eval": {"enabled": False},
            "sdr": sdr,
        }
    )
    cfg.data = cfg.data.resolve(REPO_ROOT)
    return cfg


def test_the_online_teacher_adds_no_second_model(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path))
    assert trainer.teacher_model is None, "the online teacher is the student itself"


def test_the_ema_teacher_is_frozen_and_trails_the_student(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path, teacher="ema", teacher_momentum=0.5))
    teacher = trainer.teacher_model

    assert teacher is not None
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training

    before = [p.detach().clone() for p in teacher.parameters()]
    trainer.train()

    after = list(teacher.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "EMA never updated"
    assert any(
        not torch.equal(t, s)
        for t, s in zip(teacher.parameters(), trainer.model.parameters())
    ), "the EMA teacher should trail the student, not track it exactly"


def test_the_ema_teacher_stays_out_of_the_optimiser(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path, teacher="ema"))
    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in optimised for p in trainer.teacher_model.parameters())


def test_a_frozen_teacher_needs_a_checkpoint_to_freeze():
    with pytest.raises(ValueError, match="requires sdr.teacher_model"):
        sdr_config(Path("unused"), teacher="frozen")


def test_decoder_temperatures_are_optimised_by_the_trainer(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path, learnable_temperature=True))
    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert id(trainer.sdr_loss.log_tau) in optimised

    before = trainer.sdr_loss.log_tau.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.sdr_loss.log_tau.detach()), "tau_k never moved"


def test_fixed_decoder_temperatures_stay_out_of_the_optimiser(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path, learnable_temperature=False))
    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert id(trainer.sdr_loss.log_tau) not in optimised
    assert trainer.sdr_loss not in trainer.extra_modules


def test_the_memory_queue_fills_and_feeds_the_loss(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path, queue_size=20))
    assert trainer.queue_extras() == (None, None)

    trainer.train()  # 32 samples / batch 8 = 4 steps x 2 views = 64 rows > 20
    student_extra, teacher_extra = trainer.queue_extras()
    assert student_extra.shape == (20, 32) and teacher_extra.shape == (20, 32)
    assert torch.equal(student_extra, teacher_extra), "online: the teacher is the student"


def test_the_queue_refuses_the_cka_geometry():
    with pytest.raises(ValueError, match="incompatible with geometry='cka'"):
        sdr_config(Path("unused"), queue_size=8, geometry="cka")


def test_diagnostics_are_appended_to_a_jsonl_log(tmp_path, offline_backbone):
    import json

    trainer = build_trainer(sdr_config(tmp_path, diagnostics_every=1))
    trainer.train()

    records = [
        json.loads(line)
        for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(trainer.train_loader)
    assert [r["step"] for r in records] == list(range(1, len(records) + 1))
    assert set(records[0]["per_dim"]) == {"dim_8", "dim_16", "dim_32"}
    assert abs(records[0]["full_dim_distortion"]) < 1e-5


def test_diagnostics_stay_off_by_default(tmp_path, offline_backbone):
    trainer = build_trainer(sdr_config(tmp_path))
    trainer.train()
    assert not (tmp_path / "diagnostics.jsonl").exists()
    assert trainer.diagnostics == []


def test_the_semantic_protocol_lands_in_the_report(tmp_path, offline_backbone):
    import json

    from test_trainers import _write_tiny_eval_corpus

    data_root = tmp_path / "data"
    _write_tiny_eval_corpus(data_root)

    cfg = sdr_config(tmp_path / "run")
    cfg.data.root = str(data_root)
    cfg.eval.enabled = True
    cfg.eval.cls_tasks = []
    cfg.eval.sts_tasks = ["stsb"]
    cfg.eval.pair_tasks = []
    cfg.eval.semantic_distortion = True
    cfg.eval.distortion_tasks = ["stsb"]
    cfg.eval.knn_k = 5
    cfg.eval.rotation_trials = 2

    report = build_trainer(cfg).evaluate_only()
    semantic = report["semantic"]

    assert semantic["reference"] == "self"
    assert set(semantic["distortion"]) == {"dim_8", "dim_16", "dim_32"}
    assert abs(semantic["distortion"]["dim_32"]) < 1e-5
    assert semantic["student_temperature_calibrated"] is True
    assert set(semantic["student_temperatures"]) == {"dim_8", "dim_16", "dim_32"}
    assert all(t >= 0.05 for t in semantic["student_temperatures"].values())
    assert semantic["normalized_distortion"]["dim_32"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= semantic["sdra"]
    assert semantic["preservation"]["dim_32"]["knn_recall"] == pytest.approx(1.0)
    assert semantic["rotation_stress_test"]["full_dim_gram_shift"] < 1e-3

    # The flat table carries distortion next to quality, per dimension.
    assert report["table"]["dim_8"]["semantic/distortion"] == semantic["distortion"]["dim_8"]

    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert saved["semantic"]["sdra"] == pytest.approx(semantic["sdra"])


def test_the_math_verification_script_still_passes():
    """``scripts/verify_sdr_math.py`` is the method's real regression net."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_sdr_math.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
