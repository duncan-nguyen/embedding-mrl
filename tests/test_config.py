"""Config loading, inheritance and validation (no torch required)."""

from pathlib import Path

import pytest

from embedding_mrl.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(p for p in (REPO_ROOT / "configs").rglob("*.yaml") if p.name != "base.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_configs_load(path):
    cfg = ExperimentConfig.load(path)
    assert cfg.method in ("mrl", "ese", "mipic", "gsr")
    assert max(cfg.matryoshka.dims) == cfg.model.hidden_dim
    assert Path(cfg.data.train_path).exists()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_configs_use_the_paper_training_settings(path):
    """Table 7: 5 epochs, lr 2e-5, max length 256, batch 16, cosine, AdamW, tau=0.05."""
    cfg = ExperimentConfig.load(path)
    assert cfg.train.epochs == 5
    assert cfg.train.lr == pytest.approx(2e-5)
    assert cfg.train.batch_size == 16
    assert cfg.data.max_length == 256
    assert cfg.train.scheduler.startswith("cosine")
    assert cfg.matryoshka.temperature == pytest.approx(0.05)
    assert cfg.matryoshka.dims == [16, 32, 64, 128, 256, 512, cfg.model.hidden_dim]


@pytest.mark.parametrize(
    "name,alpha", [("tinybert_6l", 0.4), ("bert", 0.4), ("qwen3_0.6b", 0.5), ("bgem3", 0.5)]
)
def test_mipic_alpha_matches_table_7(name, alpha):
    cfg = ExperimentConfig.load(REPO_ROOT / "configs" / "mipic" / f"{name}.yaml")
    assert cfg.mipic.alpha == pytest.approx(alpha)


@pytest.mark.parametrize(
    "name,layers,checkpoints",
    [
        ("tinybert_6l", [1, 2, 3, 4, 5, 6],
         [(16, 1), (32, 2), (64, 3), (256, 4), (512, 5), (768, 6)]),
        ("bert", [2, 4, 6, 8, 9, 10, 12],
         [(16, 2), (32, 4), (64, 6), (128, 8), (256, 9), (512, 10), (768, 12)]),
        ("bgem3", [1, 4, 7, 11, 15, 19, 24],
         [(16, 1), (32, 4), (64, 7), (128, 11), (256, 15), (512, 19), (1024, 24)]),
        ("qwen3_0.6b", [2, 6, 12, 16, 20, 24, 28],
         [(16, 2), (32, 6), (64, 12), (128, 16), (256, 20), (512, 24), (1024, 28)]),
    ],
)
def test_mipic_layers_and_checkpoints_match_appendix_a5(name, layers, checkpoints):
    cfg = ExperimentConfig.load(REPO_ROOT / "configs" / "mipic" / f"{name}.yaml")
    assert cfg.mipic.layers == layers
    assert cfg.mipic.checkpoint_pairs() == checkpoints
    assert cfg.mipic.gamma_schedule == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert cfg.mipic.k_min == 8
    assert cfg.mipic.aggregate == "sum"


def test_there_is_one_config_per_method_and_model():
    assert len(CONFIGS) == 16  # 4 methods x 4 backbones


def test_base_inheritance_and_override(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(
        f"_base_: {REPO_ROOT / 'configs' / 'base.yaml'}\n"
        "name: child\n"
        "train:\n  epochs: 1\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.load(child, overrides=["train.batch_size=4", "eval.enabled=false"])
    assert cfg.name == "child"
    assert cfg.train.epochs == 1
    assert cfg.train.batch_size == 4
    assert cfg.eval.enabled is False
    # untouched keys still come from base.yaml
    assert cfg.train.lr == pytest.approx(2e-5)


def test_dims_above_hidden_dim_are_rejected():
    with pytest.raises(ValueError, match="exceed model.hidden_dim"):
        ExperimentConfig.from_dict({"model": {"hidden_dim": 768}, "matryoshka": {"dims": [16, 1024]}})


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentConfig.from_dict({"trian": {"epochs": 1}})


def test_checkpoints_parse_as_dim_layer_pairs():
    cfg = ExperimentConfig.from_dict({"method": "mipic"})
    assert cfg.mipic.checkpoint_pairs()[0] == (16, 2)
    assert cfg.mipic.checkpoint_pairs()[-1] == (768, 12)


def test_alpha_splits_the_objective_as_equation_18():
    cfg = ExperimentConfig.from_dict({"method": "mipic", "mipic": {"alpha": 0.4}})
    assert cfg.mipic.w_matryoshka == pytest.approx(0.4)
    assert cfg.mipic.w_align == pytest.approx(0.6)


def test_alpha_must_be_a_valid_mixing_weight():
    with pytest.raises(ValueError, match="alpha must be in"):
        ExperimentConfig.from_dict({"method": "mipic", "mipic": {"alpha": 1.5}})


def test_mipic_checkpoint_dims_must_fit_the_backbone():
    with pytest.raises(ValueError, match="checkpoints reference dims"):
        ExperimentConfig.from_dict(
            {
                "method": "mipic",
                "model": {"hidden_dim": 768},
                "mipic": {"checkpoints": [[16, 3], [1024, 11]]},
            }
        )


def test_gamma_schedule_must_match_the_truncated_prefixes():
    with pytest.raises(ValueError, match="gamma_schedule has"):
        ExperimentConfig.from_dict(
            {"method": "mipic", "mipic": {"gamma_schedule": [0.2, 0.3]}}
        )


def test_gsr_requires_an_active_epoch_and_full_geometry_endpoint():
    with pytest.raises(ValueError, match="warmup_epochs must be smaller"):
        ExperimentConfig.from_dict(
            {"method": "gsr", "train": {"epochs": 1}, "gsr": {"warmup_epochs": 1}}
        )
    with pytest.raises(ValueError, match="largest GSR geometry dimension"):
        ExperimentConfig.from_dict(
            {
                "method": "gsr",
                "model": {"hidden_dim": 768},
                "matryoshka": {"dims": [16, 32, 768]},
                "gsr": {"geometry_dims": [16, 32], "warmup_epochs": 0},
            }
        )


def test_gsr_can_be_intentionally_disabled_during_all_epochs():
    cfg = ExperimentConfig.from_dict(
        {
            "method": "gsr",
            "train": {"epochs": 1},
            "gsr": {"weight": 0.0, "warmup_epochs": 1},
        }
    )
    assert cfg.gsr.weight == 0.0


def test_gsr_weight_is_a_bounded_tradeoff():
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        ExperimentConfig.from_dict(
            {"method": "gsr", "gsr": {"weight": 1.01, "warmup_epochs": 0}}
        )


# --------------------------------------------------------------------------- #
# CLI override parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("2e-5", 2e-5),      # PyYAML 1.1 reads this as a *string* on its own
        ("2e-05", 2e-5),
        ("2E-5", 2e-5),
        ("2.0e-5", 2e-5),    # the form YAML does accept
        ("1e10", 1e10),
        ("0.05", 0.05),
    ],
)
def test_scientific_notation_overrides_parse_as_numbers(text, expected):
    """`--set train.lr=2e-5` must not silently arrive as a string."""
    from embedding_mrl.config import _parse_override_value

    value = _parse_override_value(text)
    assert isinstance(value, float)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("16", 16),
        ("true", True),
        ("null", None),
        ("[16, 32]", [16, 32]),
        ("cosine_with_min_lr", "cosine_with_min_lr"),
        ("google-bert/bert-base-uncased", "google-bert/bert-base-uncased"),
        ("e5-large", "e5-large"),  # a model name, not a number
    ],
)
def test_other_override_values_keep_their_yaml_meaning(text, expected):
    from embedding_mrl.config import _parse_override_value

    assert _parse_override_value(text) == expected


def test_a_learning_rate_override_survives_a_full_config_load():
    cfg = ExperimentConfig.load(
        REPO_ROOT / "configs" / "mipic" / "bert.yaml", ["train.lr=2e-5"]
    )
    assert cfg.train.lr == pytest.approx(2e-5)
    # This is the multiplication that used to raise TypeError.
    assert cfg.train.lr * cfg.mipic.module_lr_scale == pytest.approx(4e-5)


def test_a_string_in_a_numeric_field_is_rejected_at_config_time():
    """The failure must name the field, not surface later inside the optimiser."""
    with pytest.raises(TypeError, match="train.lr must be a number"):
        ExperimentConfig.from_dict({"train": {"lr": "2e-5"}})
    with pytest.raises(TypeError, match="train.epochs must be an integer"):
        ExperimentConfig.from_dict({"train": {"epochs": 1.5}})
