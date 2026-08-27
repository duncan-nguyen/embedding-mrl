import pytest
import torch

from embedding_mrl.pooling import cls_pooling, get_pooler, last_token_pooling, mean_pooling, pool

B, L, D = 4, 6, 8


def test_cls_pooling_takes_the_first_token():
    hidden = torch.randn(B, L, D)
    assert torch.allclose(cls_pooling(hidden, torch.ones(B, L)), hidden[:, 0, :])


def test_mean_pooling_ignores_padding():
    hidden = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 3:] = 0
    assert torch.allclose(mean_pooling(hidden, mask), hidden[:, :3, :].mean(dim=1), atol=1e-6)


def test_mean_pooling_survives_an_all_padding_row():
    hidden = torch.randn(B, L, D)
    assert torch.isfinite(mean_pooling(hidden, torch.zeros(B, L, dtype=torch.long))).all()


def test_last_token_pooling_picks_the_final_real_token():
    hidden = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.long)
    lengths = [1, 3, 6, 2]
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    pooled = last_token_pooling(hidden, mask)
    for row, length in enumerate(lengths):
        assert torch.allclose(pooled[row], hidden[row, length - 1])


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        get_pooler("magic")


@pytest.mark.parametrize("mode", ["cls", "mean", "last"])
def test_pool_dispatch_returns_two_dimensional_output(mode):
    assert pool(torch.randn(B, L, D), torch.ones(B, L, dtype=torch.long), mode).shape == (B, D)
