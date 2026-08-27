"""Metric helpers and the summary aggregation."""

import numpy as np
import pytest

from embedding_mrl.evaluation import pair_classification_metrics, summarize


def test_perfectly_separable_scores_reach_full_accuracy():
    scores = np.array([0.05, 0.1, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    metrics = pair_classification_metrics(scores, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert 0.1 < metrics["best_threshold"] <= 0.9


def test_metrics_accept_python_lists():
    """The notebooks passed lists here and crashed on `list >= float`."""
    metrics = pair_classification_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_accuracy_never_falls_below_the_majority_class():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.3).astype(int)
    metrics = pair_classification_metrics(scores, labels)
    assert metrics["accuracy"] >= 0.7 - 1e-9


def test_summary_averages_each_family_per_dimension():
    results = {
        "classification": {
            "a": {"dim_16": {"accuracy": 0.4, "f1": 0.3}, "dim_32": {"accuracy": 0.6, "f1": 0.5}},
            "b": {"dim_16": {"accuracy": 0.6, "f1": 0.5}, "dim_32": {"accuracy": 0.8, "f1": 0.7}},
        },
        "sts": {"s1": {"dim_16": 0.5, "dim_32": 0.7}},
        "pair": {"p1": {"dim_16": {"accuracy": 0.9}, "dim_32": {"accuracy": 1.0}}},
    }
    summary = summarize(results)
    assert summary["classification"] == {"dim_16": pytest.approx(0.5), "dim_32": pytest.approx(0.7)}
    assert summary["sts"] == {"dim_16": pytest.approx(0.5), "dim_32": pytest.approx(0.7)}
    assert summary["pair"] == {"dim_16": pytest.approx(0.9), "dim_32": pytest.approx(1.0)}


def test_summary_of_an_empty_run_is_empty():
    assert summarize({"classification": {}, "sts": {}, "pair": {}}) == {}
