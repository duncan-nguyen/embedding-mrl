"""Config loading, inheritance and validation (no torch required)."""

from pathlib import Path

import pytest

from embedding_mrl.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(p for p in (REPO_ROOT / "configs").rglob("*.yaml") if p.name != "base.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_shipped_configs_load(path):
    cfg = ExperimentConfig.load(path)
    assert cfg.method in ("mrl", "ese", "mipic")
    assert max(cfg.matryoshka.dims) == cfg.model.hidden_dim
    assert Path(cfg.data.train_path).exists()


def test_there_is_one_config_per_method_and_model():
    assert len(CONFIGS) == 12


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


def test_pipeline_pairs_expand_into_consecutive_transitions():
    cfg = ExperimentConfig.from_dict(
        {"method": "mipic", "mipic": {"pipeline_pairs": [[3, 16, 7, 128, 11, 768]]}}
    )
    assert cfg.mipic.parsed_pipeline_pairs() == [(3, 16, 7, 128), (7, 128, 11, 768)]


def test_mipic_pipeline_dims_must_fit_the_backbone():
    with pytest.raises(ValueError, match="pipeline_pairs reference dims"):
        ExperimentConfig.from_dict(
            {
                "method": "mipic",
                "model": {"hidden_dim": 768},
                "mipic": {"pipeline_pairs": [[3, 16, 11, 1024]]},
            }
        )
