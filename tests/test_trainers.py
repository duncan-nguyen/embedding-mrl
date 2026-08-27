"""End-to-end pipeline smoke tests on a locally built encoder (no downloads)."""

import json
from pathlib import Path

import pytest
import torch

from embedding_mrl.config import ExperimentConfig
from embedding_mrl.trainers import TRAINERS, build_trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["mrl", "ese", "mipic"]


def make_config(method: str, output_dir: Path, **extra) -> ExperimentConfig:
    """A one-epoch, few-sample version of the shipped configs."""
    raw = {
        "name": f"test_{method}",
        "method": method,
        "model": {
            "name_or_path": "dummy",
            "hidden_dim": 32,
            "pooling": "mean" if method == "ese" else "cls",
        },
        "data": {
            "root": str(REPO_ROOT / "data"),
            "max_length": 32,
            "num_workers": 0,
            "max_train_samples": 16,
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
        "mipic": {
            "layers": [1, 3],
            "checkpoints": [[8, 1], [16, 3], [32, 4]],
            "gamma_schedule": [0.3, 0.5],
            "k_min": 2,
        },
    }
    for key, value in extra.items():
        raw.setdefault(key, {}).update(value) if isinstance(
            value, dict
        ) else raw.update({key: value})
    cfg = ExperimentConfig.from_dict(raw)
    cfg.data = cfg.data.resolve(REPO_ROOT)
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_one_epoch_runs_and_updates_the_encoder(method, tmp_path, offline_backbone):
    trainer = build_trainer(make_config(method, tmp_path))
    before = [p.detach().clone() for p in trainer.model.parameters()]

    summary = trainer.train()

    assert len(summary["history"]) == 1
    loss = summary["history"][0]["train_loss"]
    assert loss == pytest.approx(loss)  # not NaN
    assert loss > 0

    after = list(trainer.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "no weights moved"
    assert json.loads((tmp_path / "history.json").read_text())[0]["epoch"] == 1
    assert (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("method", METHODS)
def test_compute_loss_produces_a_finite_differentiable_scalar(
    method, tmp_path, offline_backbone
):
    trainer = build_trainer(make_config(method, tmp_path))
    batch = trainer._to_device(next(iter(trainer.train_loader)))

    loss, logs = trainer.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss.requires_grad
    assert all(isinstance(v, float) for v in logs.values())

    loss.backward()
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mipic_optimiser_includes_the_alignment_module(tmp_path, offline_backbone):
    trainer = build_trainer(make_config("mipic", tmp_path))
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(
        groups[0]["lr"] * trainer.cfg.mipic.module_lr_scale
    )
    assert any(p.requires_grad for p in trainer.alignment_loss.parameters())


def test_mipic_alignment_module_also_trains(tmp_path, offline_backbone):
    trainer = build_trainer(make_config("mipic", tmp_path))
    before = [p.detach().clone() for p in trainer.alignment_loss.parameters()]
    trainer.train()
    after = list(trainer.alignment_loss.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_mipic_rejects_layer_indices_the_backbone_cannot_provide(
    tmp_path, offline_backbone
):
    cfg = make_config("mipic", tmp_path)
    cfg.mipic.layers = [1, 99]  # the dummy encoder exposes 5 hidden states
    with pytest.raises(ValueError, match="reference hidden-state indices"):
        build_trainer(cfg)


def test_hidden_dim_mismatch_is_caught_before_training(tmp_path, offline_backbone):
    cfg = make_config("mrl", tmp_path)
    cfg.model.hidden_dim = 64  # the dummy encoder is 32-wide
    cfg.matryoshka.dims = [8, 16, 64]
    with pytest.raises(ValueError, match="hidden_size"):
        build_trainer(cfg)


def test_scheduler_covers_exactly_one_pass_over_the_data(tmp_path, offline_backbone):
    trainer = build_trainer(make_config("mrl", tmp_path))
    assert trainer.total_steps == len(trainer.train_loader)
    trainer.train()
    assert trainer.scheduler.last_epoch == trainer.total_steps


def test_every_method_has_a_trainer():
    assert sorted(TRAINERS) == sorted(METHODS)


def _write_tiny_eval_corpus(root: Path) -> None:
    """A miniature stand-in for data/test with the filenames the registry expects."""
    import pandas as pd

    test_dir = root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (root / "train").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"text": [f"train sentence number {i}" for i in range(16)]}).to_csv(
        root / "train" / "final_data.csv", index=False
    )

    cls_frame = pd.DataFrame(
        {
            "text": [
                f"utterance {i} about topic {i % 3} with filler {i * 7 % 11}"
                for i in range(24)
            ],
            "label": [i % 3 for i in range(24)],
        }
    )
    cls_frame.to_csv(test_dir / "emotion_train.csv", index=False)
    cls_frame.to_csv(test_dir / "emotion_test.csv", index=False)

    # The stub encoder is a bag-of-words model, so the two sides need genuinely
    # different vocabulary (not a rotation) to produce varying similarities.
    pair_frame = pd.DataFrame(
        {
            "sentence1": [f"w{i} w{i + 1} w{i + 2} shared" for i in range(20)],
            "sentence2": [f"w{i * 3 % 20} w{i * 5 % 20} shared" for i in range(20)],
            "score": [(i % 6) for i in range(20)],
            "label": [i % 2 for i in range(20)],
        }
    )
    pair_frame.to_csv(test_dir / "stsb_test.csv", index=False)
    pair_frame.to_csv(test_dir / "mrpc_test.csv", index=False)


def test_evaluation_suite_reports_every_task_at_every_dimension(
    tmp_path, offline_backbone
):
    data_root = tmp_path / "data"
    _write_tiny_eval_corpus(data_root)

    cfg = make_config("mrl", tmp_path / "run")
    cfg.data.root = str(data_root)
    cfg.eval.enabled = True
    cfg.eval.cls_tasks = ["emotion"]
    cfg.eval.sts_tasks = ["stsb"]
    cfg.eval.pair_tasks = ["mrpc"]
    cfg.eval.batch_size = 8
    cfg.eval.logreg_max_iter = 20

    trainer = build_trainer(cfg)
    results = trainer.evaluate_only()

    dims = {f"dim_{d}" for d in cfg.matryoshka.dims}
    assert set(results["classification"]["emotion"]) == dims
    assert set(results["sts"]["stsb"]) == dims
    assert set(results["pair"]["mrpc"]) == dims
    assert set(results["summary"]) == {"classification", "sts", "pair"}

    metrics = results["pair"]["mrpc"]["dim_32"]
    assert {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "best_threshold",
    } <= set(metrics)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    for family, scores in results["summary"].items():
        assert all(v == v for v in scores.values()), f"{family} produced NaN"

    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert saved["summary"] == results["summary"]


def test_evaluation_leaves_the_model_in_training_mode(tmp_path, offline_backbone):
    data_root = tmp_path / "data"
    _write_tiny_eval_corpus(data_root)

    cfg = make_config("mrl", tmp_path / "run")
    cfg.data.root = str(data_root)
    cfg.eval.enabled = True
    cfg.eval.cls_tasks = []
    cfg.eval.sts_tasks = ["stsb"]
    cfg.eval.pair_tasks = []

    trainer = build_trainer(cfg)
    trainer.model.train()
    trainer.evaluator.evaluate(trainer.model)
    assert trainer.model.training


@pytest.mark.parametrize("method", METHODS)
def test_gradient_clipping_path_runs_with_a_disabled_scaler(
    method, tmp_path, offline_backbone
):
    """fp16 is off on CPU, so the GradScaler is disabled - unscale_ must still be safe."""
    cfg = make_config(method, tmp_path)
    cfg.train.max_grad_norm = 1.0
    trainer = build_trainer(cfg)
    trainer.train()

    for param in trainer.model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("method", METHODS)
def test_training_ends_with_an_evaluation_report_on_disk(method, tmp_path, offline_backbone):
    """After the last epoch the run must evaluate and persist results.json."""
    data_root = tmp_path / "data"
    _write_tiny_eval_corpus(data_root)

    cfg = make_config(method, tmp_path / "run")
    cfg.data.root = str(data_root)
    cfg.eval.enabled = True
    cfg.eval.every_epoch = False  # evaluate only once, at the end
    cfg.eval.cls_tasks = ["emotion"]
    cfg.eval.sts_tasks = ["stsb"]
    cfg.eval.pair_tasks = ["mrpc"]
    cfg.eval.batch_size = 8
    cfg.eval.logreg_max_iter = 20

    outcome = build_trainer(cfg).train()
    run_dir = tmp_path / "run"

    assert (run_dir / "results.json").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "results_epoch1.json").exists()

    report = json.loads((run_dir / "results.json").read_text())
    assert report["experiment"]["method"] == method
    assert report["training"]["final_loss"] == pytest.approx(outcome["history"][0]["train_loss"])
    assert set(report["table"]) == {f"dim_{d}" for d in cfg.matryoshka.dims}
    assert report["table"]["dim_32"]["mean/sts"] == report["summary"]["sts"]["dim_32"]


def test_eval_only_writes_the_same_report(tmp_path, offline_backbone):
    data_root = tmp_path / "data"
    _write_tiny_eval_corpus(data_root)

    cfg = make_config("mrl", tmp_path / "run")
    cfg.data.root = str(data_root)
    cfg.eval.enabled = True
    cfg.eval.cls_tasks = []
    cfg.eval.sts_tasks = ["stsb"]
    cfg.eval.pair_tasks = ["mrpc"]

    report = build_trainer(cfg).evaluate_only()
    assert (tmp_path / "run" / "results.json").exists()
    assert report["training"]["epochs_completed"] == 0
    assert report["summary"]["sts"]
