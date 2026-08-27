"""The final report: metadata, per-dimension table, JSON + CSV output."""

import csv
import json

import pytest

from embedding_mrl.config import ExperimentConfig
from embedding_mrl.reporting import (
    build_dimension_table,
    build_report,
    format_summary,
    write_report,
)

RESULTS = {
    "classification": {
        "emotion": {"dim_16": {"accuracy": 0.40, "f1": 0.31}, "dim_32": {"accuracy": 0.60, "f1": 0.55}}
    },
    "sts": {"stsb": {"dim_16": 0.55, "dim_32": 0.71}},
    "pair": {
        "mrpc": {
            "dim_16": {"accuracy": 0.66, "f1": 0.60, "average_precision": 0.70, "best_threshold": 0.5},
            "dim_32": {"accuracy": 0.72, "f1": 0.68, "average_precision": 0.75, "best_threshold": 0.5},
        }
    },
    "summary": {
        "classification": {"dim_16": 0.40, "dim_32": 0.60},
        "sts": {"dim_16": 0.55, "dim_32": 0.71},
        "pair": {"dim_16": 0.66, "dim_32": 0.72},
    },
}

HISTORY = [{"epoch": 1, "train_loss": 3.2}, {"epoch": 2, "train_loss": 2.1}]


@pytest.fixture
def cfg():
    return ExperimentConfig.from_dict(
        {
            "name": "demo",
            "method": "mipic",
            "model": {"hidden_dim": 32},
            "matryoshka": {"dims": [16, 32]},
            "mipic": {"layers": [1], "checkpoints": [[16, 1], [32, 3]],
                      "gamma_schedule": [0.3]},
            "train": {"epochs": 2},
        }
    )


def test_table_has_one_row_per_dimension_with_task_and_mean_columns():
    table = build_dimension_table(RESULTS)
    assert list(table) == ["dim_16", "dim_32"]
    assert table["dim_32"]["classification/emotion"] == pytest.approx(0.60)
    assert table["dim_32"]["sts/stsb"] == pytest.approx(0.71)
    assert table["dim_32"]["pair/mrpc"] == pytest.approx(0.72)
    assert table["dim_32"]["mean/sts"] == pytest.approx(0.71)


def test_table_rows_are_ordered_numerically_not_lexically():
    results = {"sts": {"t": {"dim_128": 0.1, "dim_16": 0.2, "dim_1024": 0.3}}, "summary": {}}
    assert list(build_dimension_table(results)) == ["dim_16", "dim_128", "dim_1024"]


def test_report_keeps_the_historical_top_level_keys(cfg):
    report = build_report(cfg, RESULTS, HISTORY, duration_seconds=12.34)
    assert {"classification", "sts", "pair", "summary"} <= set(report)
    assert report["classification"] == RESULTS["classification"]


def test_report_records_the_run_metadata(cfg):
    report = build_report(cfg, RESULTS, HISTORY, duration_seconds=12.34)
    experiment = report["experiment"]
    assert experiment["name"] == "demo"
    assert experiment["method"] == "mipic"
    assert experiment["matryoshka_dims"] == [16, 32]
    assert experiment["eval_split"] == "test"
    assert experiment["finished_at"]

    training = report["training"]
    assert training["epochs_completed"] == 2
    assert training["final_loss"] == pytest.approx(2.1)
    assert training["loss_per_epoch"] == [3.2, 2.1]
    assert training["duration_seconds"] == pytest.approx(12.3)


def test_report_survives_an_empty_history(cfg):
    report = build_report(cfg, RESULTS, history=[])
    assert report["training"]["final_loss"] is None
    assert report["training"]["epochs_completed"] == 0


def test_write_report_emits_valid_json_and_csv(cfg, tmp_path):
    report = build_report(cfg, RESULTS, HISTORY)
    path = write_report(report, tmp_path)

    assert path == tmp_path / "results.json"
    loaded = json.loads(path.read_text())
    assert loaded["summary"] == RESULTS["summary"]
    assert loaded["experiment"]["name"] == "demo"

    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [row["dim"] for row in rows] == ["16", "32"]
    assert float(rows[1]["mean/sts"]) == pytest.approx(0.71)
    assert float(rows[1]["classification/emotion"]) == pytest.approx(0.60)


def test_write_report_accepts_a_custom_stem(cfg, tmp_path):
    write_report(build_report(cfg, RESULTS, HISTORY), tmp_path, stem="results_epoch2")
    assert (tmp_path / "results_epoch2.json").exists()
    assert (tmp_path / "results_epoch2.csv").exists()


def test_format_summary_lists_every_dimension(cfg):
    text = format_summary(build_report(cfg, RESULTS, HISTORY))
    assert "classification" in text and "sts" in text and "pair" in text
    assert "16" in text and "32" in text


def test_format_summary_handles_no_results(cfg):
    assert format_summary(build_report(cfg, {}, [])) == "no evaluation results"
