"""Loss maths: shapes, gradients, and the properties each objective claims."""

import math

import pytest
import torch

from embedding_mrl.losses import (
    AttentionDistributionMatching,
    CKALoss,
    MIPICAlignmentLoss,
    PerExampleCKALoss,
    PipelineInfoNCELoss,
    TopKCKAAlignment,
    build_shell_slices,
    condensed_squared_distances,
    epresso_simcse,
    epresso_simcse_from_hidden_states,
    info_nce,
    full_normalize,
    gsr_shell_loss,
    matryoshka_info_nce,
    merge_tied_shells,
)

DIMS = [4, 8, 16, 32]
B, L, D = 6, 12, 32


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _hidden(n_layers: int = 5):
    return [torch.randn(B, L, D, requires_grad=True) for _ in range(n_layers)]


def test_info_nce_is_minimised_by_matching_views():
    x = torch.randn(B, D)
    aligned, _ = info_nce(x, x.clone())
    shuffled, _ = info_nce(x, x[torch.randperm(B)])
    assert aligned < shuffled


def test_matryoshka_info_nce_sums_over_every_nested_dim():
    a, b = torch.randn(B, D), torch.randn(B, D)
    total, logits = matryoshka_info_nce(a, b, nested_dims=DIMS)
    assert set(logits) == {f"dim_{d}" for d in DIMS}
    expected = sum(info_nce(a[:, :d], b[:, :d])[0] for d in DIMS)
    assert torch.allclose(total, expected)


def test_matryoshka_info_nce_skips_dims_wider_than_the_embedding():
    a, b = torch.randn(B, 16), torch.randn(B, 16)
    total, logits = matryoshka_info_nce(a, b, nested_dims=[8, 16, 64])
    assert set(logits) == {"dim_8", "dim_16"}
    assert torch.isfinite(total)


def test_matryoshka_info_nce_rejects_impossible_dims():
    with pytest.raises(ValueError, match="no nested dim fits"):
        matryoshka_info_nce(torch.randn(B, 8), torch.randn(B, 8), nested_dims=[64])


def test_matryoshka_info_nce_backpropagates():
    a = torch.randn(B, D, requires_grad=True)
    b = torch.randn(B, D, requires_grad=True)
    matryoshka_info_nce(a, b, nested_dims=DIMS)[0].backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_full_normalization_preserves_prefix_cosine():
    x = torch.randn(B, D)
    y = torch.randn(B, D)
    qx = full_normalize(x)
    qy = full_normalize(y)
    for dim in DIMS:
        expected = torch.nn.functional.cosine_similarity(x[:, :dim], y[:, :dim])
        actual = torch.nn.functional.cosine_similarity(qx[:, :dim], qy[:, :dim])
        assert torch.allclose(actual, expected, atol=1e-6)


def test_shell_distances_add_to_the_full_distance():
    points = torch.randn(B, D)
    shells = build_shell_slices(DIMS, full_dim=D)
    shell_distances = sum(
        condensed_squared_distances(points[:, start:end]) for start, end in shells
    )
    assert torch.allclose(
        shell_distances, condensed_squared_distances(points), atol=1e-5
    )


def test_gsr_is_zero_for_matching_shell_geometries_and_only_grads_student():
    teacher = torch.randn(B, D)
    student = teacher.clone().requires_grad_()
    out = gsr_shell_loss(student, teacher, build_shell_slices(DIMS), c_teacher=1.0)
    assert out.total_loss.item() == pytest.approx(0.0, abs=1e-8)
    out.total_loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_gsr_supports_shells_wider_than_the_batch():
    student = torch.randn(3, D, requires_grad=True)
    teacher = torch.randn(3, D)
    out = gsr_shell_loss(student, teacher, [(0, 4), (4, 32)], c_teacher=1.0)
    assert torch.isfinite(out.total_loss)
    assert out.teacher_distances["dim_4_32"].abs().sum() > 0
    out.total_loss.backward()
    assert torch.isfinite(student.grad).all()


def test_gsr_batch_loss_is_an_unbiased_pair_estimator():
    student = torch.randn(4, 4)
    teacher = torch.randn(4, 4)
    shells = [(0, 2), (2, 4)]
    corpus_loss = gsr_shell_loss(student, teacher, shells, c_teacher=1.0).total_loss

    pair_losses = []
    for left in range(4):
        for right in range(left + 1, 4):
            ids = torch.tensor([left, right])
            pair_losses.append(
                gsr_shell_loss(
                    student[ids], teacher[ids], shells, c_teacher=1.0
                ).total_loss
            )
    assert torch.allclose(torch.stack(pair_losses).mean(), corpus_loss, atol=1e-6)


def test_tied_eigenvalue_boundary_is_merged():
    eigenvalues = torch.tensor([3.0, 2.0, 2.0, 1.0])
    shells, merged = merge_tied_shells([2, 4], eigenvalues, tolerance=0.0)
    assert shells == [(0, 4)]
    assert merged == [2]


def test_epresso_weights_smaller_dims_more():
    a, b = torch.randn(B, D), torch.randn(B, D)
    weighted, loss_dict, acc_dict = epresso_simcse(a, b, matryoshka_dims=DIMS)
    unweighted, _, _ = epresso_simcse(a, b, matryoshka_dims=DIMS, use_layer_weight=False)

    assert weighted < unweighted  # log weights are all <= 1
    manual = sum(
        loss_dict[f"loss_dim_{d}"] / (1 + math.log(i + 1)) for i, d in enumerate(DIMS)
    )
    assert weighted.item() == pytest.approx(manual, rel=1e-5)
    assert all(0.0 <= v <= 1.0 for v in acc_dict.values())


def test_epresso_from_hidden_states_adds_intermediate_layers():
    hs1, hs2 = _hidden(), _hidden()
    mask = torch.ones(B, L, dtype=torch.long)

    final_only, _, _ = epresso_simcse_from_hidden_states(
        hs1, hs2, mask, mask, matryoshka_dims=DIMS, use_intermediate_layers=False
    )
    with_layers, _, _ = epresso_simcse_from_hidden_states(
        hs1, hs2, mask, mask, matryoshka_dims=DIMS, n_layers_per_step=-1
    )
    assert with_layers > final_only
    with_layers.backward()
    # index 0 is the embedding layer, which EPRESSO deliberately skips
    assert hs1[0].grad is None
    assert hs1[-1].grad is not None and torch.isfinite(hs1[-1].grad).all()


def test_cka_of_identical_representations_is_zero_distance():
    x = torch.randn(B * L, D)
    assert CKALoss()(x, x.clone()).item() == pytest.approx(0.0, abs=1e-5)
    y = torch.randn(B, 8, D)
    assert PerExampleCKALoss()(y, y.clone()).item() == pytest.approx(0.0, abs=1e-5)


def test_cka_is_invariant_to_isotropic_scaling():
    x = torch.randn(B, 8, D)
    assert PerExampleCKALoss()(x, 3.7 * x).item() == pytest.approx(0.0, abs=1e-5)


def _probs(module, hidden, mask=None, temperature=0.05):
    return module._scores_to_probs(module.teacher_scores(hidden), mask, temperature)


def test_attention_matching_masks_padding_out():
    module = AttentionDistributionMatching(d_small=8, d_full=D)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, L // 2 :] = 0
    hidden = torch.randn(B, L, D)

    probs = _probs(module, hidden, mask)
    assert torch.allclose(probs[:, L // 2 :], torch.zeros(B, L - L // 2), atol=1e-6)
    assert torch.allclose(probs.sum(dim=1), torch.ones(B), atol=1e-5)


def test_attention_kl_is_non_negative_and_zero_for_identical_distributions():
    module = AttentionDistributionMatching(d_small=D, d_full=D)
    hidden = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.long)

    teacher = _probs(module, hidden, mask)
    kl = module(hidden, hidden, teacher_probs=teacher, mask=mask)
    assert kl.item() >= -1e-6

    # Force the student's lift to be the identity -> the two distributions match.
    with torch.no_grad():
        module.up_project.weight.copy_(torch.eye(D))
    kl_identity = module(hidden, hidden, teacher_probs=teacher, mask=mask)
    assert kl_identity.item() == pytest.approx(0.0, abs=1e-4)


def test_attention_scores_use_the_full_dimensional_cls_as_query():
    """Eq 4: s_j = h_CLS . h_j / sqrt(D)."""
    module = AttentionDistributionMatching(d_small=8, d_full=D)
    hidden = torch.randn(B, L, D)
    expected = (hidden[:, 0, :].unsqueeze(1) * hidden).sum(-1) / math.sqrt(D)
    assert torch.allclose(module.teacher_scores(hidden), expected, atol=1e-5)


def test_attention_kl_flows_into_the_projection():
    module = AttentionDistributionMatching(d_small=8, d_full=D)
    hidden = torch.randn(B, L, D)
    teacher = _probs(module, hidden)
    module(hidden[..., :8], hidden, teacher_probs=teacher).backward()
    assert module.up_project.weight.grad is not None
    assert torch.isfinite(module.up_project.weight.grad).all()


def test_top_k_selection_picks_the_highest_ranked_tokens():
    hidden = torch.randn(B, L, D)
    scores = torch.randn(B, L)
    selected = TopKCKAAlignment.select_top_k(hidden, scores, k=4)
    assert selected.shape == (B, 4, D)

    expected_idx = scores.topk(4, dim=1).indices
    for i in range(B):
        assert torch.allclose(selected[i], hidden[i, expected_idx[i]])


def test_top_k_selection_never_picks_padding():
    hidden = torch.randn(B, L, D)
    scores = torch.full((B, L), -5.0)
    scores[:, :3] = 5.0
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, :3] = 1
    selected = TopKCKAAlignment.select_top_k(hidden, scores, k=3, mask=mask)
    for row in range(B):
        for token in selected[row]:
            assert any(torch.allclose(token, hidden[row, i]) for i in range(3))


def test_top_k_sets_are_nested_because_ranking_is_shared():
    """Sec 3.2.2: S_k1 subset S_k2 subset ... - the teacher ranking is used for all dims."""
    hidden = torch.randn(1, L, D)
    scores = torch.randn(1, L)
    order = scores.topk(L, dim=1).indices[0].tolist()
    for k_small, k_large in [(2, 4), (4, 7)]:
        small = set(order[:k_small])
        large = set(order[:k_large])
        assert small < large
        assert TopKCKAAlignment.select_top_k(hidden, scores, k=k_small).shape[1] == k_small


def test_pipeline_infonce_stops_gradient_on_the_target():
    src = torch.randn(B, L, 8, requires_grad=True)
    tgt = torch.randn(B, L, 16, requires_grad=True)
    PipelineInfoNCELoss(d_src=8, d_tgt=16, d_hidden=8)(src, tgt).backward()
    assert src.grad is not None and src.grad.abs().sum() > 0
    assert tgt.grad is None or tgt.grad.abs().sum() == 0


def test_pipeline_infonce_can_propagate_into_the_target():
    src = torch.randn(B, L, 8, requires_grad=True)
    tgt = torch.randn(B, L, 16, requires_grad=True)
    PipelineInfoNCELoss(d_src=8, d_tgt=16, d_hidden=8, detach_target=False)(src, tgt).backward()
    assert tgt.grad is not None and tgt.grad.abs().sum() > 0


def _alignment(**kwargs):
    defaults = dict(
        d_full=D,
        matryoshka_dims=DIMS,
        layers=[1, 3],
        checkpoints=[(4, 1), (8, 2), (32, 4)],
        gamma_schedule=[0.2, 0.4, 0.6],
        k_min=2,
        temperature=0.05,
    )
    defaults.update(kwargs)
    return MIPICAlignmentLoss(**defaults)


def test_alignment_returns_all_components_and_backpropagates():
    module = _alignment()
    hidden = _hidden()
    out = module(hidden, mask=torch.ones(B, L, dtype=torch.long))

    assert set(out) == {"total_loss", "sia_loss", "att_loss", "cka_loss", "pic_loss"}
    # Eq 13: SIA sums the attention and CKA terms with equal weight.
    assert torch.allclose(out["sia_loss"], out["att_loss"] + out["cka_loss"])
    assert torch.allclose(out["total_loss"], out["sia_loss"] + out["pic_loss"])

    out["total_loss"].backward()
    assert hidden[1].grad is not None and torch.isfinite(hidden[1].grad).all()


def test_sum_aggregation_scales_with_the_number_of_layers():
    """Eq 14 sums over layers, so more layers means a larger SIA term."""
    hidden = _hidden()
    mask = torch.ones(B, L, dtype=torch.long)
    torch.manual_seed(1)
    one = _alignment(layers=[1])(hidden, mask)["att_loss"].item()
    torch.manual_seed(1)
    two = _alignment(layers=[1, 3])(hidden, mask)["att_loss"].item()
    assert two > one


def test_mean_aggregation_averages_instead_of_summing():
    hidden = _hidden()
    mask = torch.ones(B, L, dtype=torch.long)
    summed = _alignment(aggregate="sum")(hidden, mask)
    averaged = _alignment(aggregate="mean")(hidden, mask)
    n_terms = len(DIMS[:-1]) * 2  # 3 truncated prefixes x 2 layers
    assert summed["cka_loss"].item() > averaged["cka_loss"].item()
    assert averaged["cka_loss"].item() == pytest.approx(
        summed["cka_loss"].item() / n_terms, rel=1e-4
    )


def test_top_k_schedule_follows_the_gamma_ratios():
    """Appendix A.5: k_i = max(k_min, ceil(gamma_i * m))."""
    module = _alignment(gamma_schedule=[0.2, 0.5, 0.7], k_min=8)
    assert module.top_k_for(4, seq_len=100) == 20
    assert module.top_k_for(8, seq_len=100) == 50
    assert module.top_k_for(16, seq_len=100) == 70


def test_top_k_schedule_respects_the_floor_and_the_sequence_length():
    module = _alignment(gamma_schedule=[0.2, 0.5, 0.7], k_min=8)
    assert module.top_k_for(4, seq_len=10) == 8  # ceil(0.2*10)=2 -> floored to k_min
    assert module.top_k_for(16, seq_len=5) == 5  # never exceeds the sequence
    assert module.top_k_for(16, seq_len=100, min_real_tokens=12) == 12


def test_gamma_schedule_must_cover_every_truncated_prefix():
    with pytest.raises(ValueError, match="gamma_schedule has"):
        _alignment(gamma_schedule=[0.2, 0.4])


def test_alignment_needs_at_least_two_checkpoints():
    with pytest.raises(ValueError, match="at least two checkpoints"):
        _alignment(checkpoints=[(4, 1)])


def test_out_of_range_layer_indices_fail_fast():
    with pytest.raises(ValueError, match="reference hidden-state indices"):
        _alignment(layers=[1, 99]).validate_against(num_hidden_states=5)


def test_checkpoint_layers_are_validated_too():
    with pytest.raises(ValueError, match="reference hidden-state indices"):
        _alignment(checkpoints=[(4, 1), (32, 42)]).validate_against(num_hidden_states=5)
