"""Loss maths: shapes, gradients, and the properties each objective claims."""

import math

import pytest
import torch

from embedding_mrl.losses import (
    CKALoss,
    HorizontalAttentionAlignment,
    PerExampleCKALoss,
    PipelineInfoNCELoss,
    SubmatrixCKALoss,
    TotalAlignmentLoss,
    epresso_simcse,
    epresso_simcse_from_hidden_states,
    info_nce,
    matryoshka_info_nce,
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


def test_attention_alignment_masks_padding_out():
    module = HorizontalAttentionAlignment(d_small=8, d_full=D, d_att=16, enabled=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, L // 2 :] = 0
    hidden = torch.randn(B, L, D)

    kl, scores = module(hidden[..., :8], hidden, mask=mask)
    probs, _ = module.compute_attention_dist(hidden[..., :8], module.proj_small, mask)
    assert kl.item() >= 0.0
    assert torch.allclose(probs[:, L // 2 :], torch.zeros(B, L - L // 2), atol=1e-6)
    assert scores.shape == (B, L)


def test_attention_alignment_disabled_returns_zero_but_still_scores():
    module = HorizontalAttentionAlignment(d_small=8, d_full=D, d_att=16, enabled=False)
    hidden = torch.randn(B, L, D)
    kl, scores = module(hidden[..., :8], hidden, mask=torch.ones(B, L, dtype=torch.long))
    assert kl.item() == 0.0
    assert scores.shape == (B, L)


def test_submatrix_cka_selects_the_top_k_tokens():
    hidden = torch.randn(B, L, D)
    scores = torch.randn(B, L)
    selected = SubmatrixCKALoss.select_top_k_tokens(hidden, scores, k=4)
    assert selected.shape == (B, 4, D)

    expected_idx = scores.topk(4, dim=1).indices
    for i in range(B):
        assert torch.allclose(selected[i], hidden[i, expected_idx[i]])


def test_submatrix_cka_never_selects_padding():
    hidden = torch.randn(B, L, D)
    scores = torch.full((B, L), -5.0)
    scores[:, :3] = 5.0  # only the first three tokens are informative
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, :3] = 1
    selected = SubmatrixCKALoss.select_top_k_tokens(hidden, scores, k=3, mask=mask)
    # topk orders by score, not position, so compare as sets of token vectors
    for row in range(B):
        for token in selected[row]:
            assert any(torch.allclose(token, hidden[row, i]) for i in range(3))


def test_pipeline_infonce_stops_gradient_on_the_target():
    src = torch.randn(B, L, 8, requires_grad=True)
    tgt = torch.randn(B, L, 16, requires_grad=True)
    PipelineInfoNCELoss(d_src=8, d_tgt=16, d_hidden=8)(src, tgt).backward()
    assert src.grad is not None and src.grad.abs().sum() > 0
    assert tgt.grad is None or tgt.grad.abs().sum() == 0


def _alignment(**kwargs):
    defaults = dict(
        d_full=D,
        matryoshka_dims=DIMS,
        align_layers=[1, 3],
        pipeline_pairs=[(1, 4, 3, 32)],
        d_att=16,
        base_k=8,
    )
    defaults.update(kwargs)
    return TotalAlignmentLoss(**defaults)


def test_total_alignment_returns_all_components_and_backpropagates():
    module = _alignment(use_attention_kl=True)
    hidden = _hidden()
    out = module(hidden, mask=torch.ones(B, L, dtype=torch.long))

    assert set(out) == {"total_loss", "att_loss", "cka_loss", "chain_loss"}
    expected = 0.4 * out["att_loss"] + 0.4 * out["cka_loss"] + 0.2 * out["chain_loss"]
    assert torch.allclose(out["total_loss"], expected)

    out["total_loss"].backward()
    assert hidden[1].grad is not None and torch.isfinite(hidden[1].grad).all()


def test_attention_term_is_zero_when_the_kl_is_disabled():
    out = _alignment(use_attention_kl=False)(_hidden(), mask=torch.ones(B, L, dtype=torch.long))
    assert out["att_loss"].item() == 0.0
    assert out["cka_loss"].item() != 0.0


def test_default_k_map_grows_with_the_dimension():
    module = _alignment(base_k=64)
    assert module.k_map(D) == 64
    assert module.k_map(4) == 8  # floor of 8
    assert module.k_map(16) <= module.k_map(32)


def test_explicit_k_map_is_honoured():
    module = _alignment(k_map={4: 2, 8: 3})
    assert module.k_map(4) == 2 and module.k_map(8) == 3


def test_out_of_range_layer_indices_fail_fast():
    with pytest.raises(ValueError, match="reference hidden-state indices"):
        _alignment(align_layers=[1, 99]).validate_against(num_hidden_states=5)
