"""Turn raw evaluation scores into the final ``results.json`` (and a flat CSV).

The report keeps the ``classification`` / ``sts`` / ``pair`` top-level keys the
project has always used, and adds the run metadata and a per-dimension table
that make a result file readable months later.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .utils import save_json

#: Metrics lifted into the flat per-dimension table, per family. The **first**
#: entry keeps the historical bare column name (``classification/emotion``);
#: later ones get a ``:metric`` suffix. Macro-F1 travels next to accuracy
#: because ``docs/MIPIC.pdf`` Tables 1-2 report macro-F1 for the classification
#: tasks - on a skewed label set the two differ by ~20 points, so lining an
#: accuracy column up against that table silently compares different statistics.
FAMILY_METRICS: Dict[str, tuple] = {
    "classification": ("accuracy", "f1"),
    "sts": (None,),  # STS scores are bare floats
    "pair": ("accuracy", "f1"),
}


def _metric_value(score: Any, key: Optional[str]) -> Optional[float]:
    if key is None:
        return None if score is None else float(score)
    if isinstance(score, dict) and key in score:
        return float(score[key])
    return None


def build_dimension_table(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """``dim -> {task: headline metric}`` across every family, for quick reading."""
    table: Dict[str, Dict[str, float]] = {}
    for family, metrics in FAMILY_METRICS.items():
        for task, per_dim in results.get(family, {}).items():
            for dim, score in per_dim.items():
                for index, key in enumerate(metrics):
                    value = _metric_value(score, key)
                    if value is None:
                        continue
                    column = f"{family}/{task}" if index == 0 else f"{family}/{task}:{key}"
                    table.setdefault(dim, {})[column] = value

    # SDR-MRL Sec 6: distortion sits next to quality so the two can be read off
    # the same row - that comparison is RQ4 (Eq 121).
    semantic = results.get("semantic", {})
    for dim, value in semantic.get("distortion", {}).items():
        table.setdefault(dim, {})["semantic/distortion"] = float(value)
    for dim, scores in semantic.get("preservation", {}).items():
        table.setdefault(dim, {})["semantic/knn_recall"] = float(scores["knn_recall"])

    # Append the family averages already computed by the evaluator.
    for family, per_dim in results.get("summary", {}).items():
        for dim, value in per_dim.items():
            table.setdefault(dim, {})[f"mean/{family}"] = float(value)

    # Order by dimension, numerically rather than lexically.
    return {
        dim: table[dim] for dim in sorted(table, key=lambda d: int(d.split("_")[1]))
    }


def build_report(
    cfg: ExperimentConfig,
    results: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the final report dict written to ``results.json``."""
    history = history or []
    trained_epochs = [record for record in history if "train_loss" in record]

    report: Dict[str, Any] = {
        "experiment": {
            "name": cfg.name,
            "method": cfg.method,
            "model": cfg.model.name_or_path,
            "hidden_dim": cfg.model.hidden_dim,
            "pooling": cfg.model.pooling,
            "matryoshka_dims": cfg.matryoshka.ascending,
            "eval_split": cfg.eval.split,
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "seed": cfg.train.seed,
            "finished_at": _dt.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        },
        "training": {
            "epochs_completed": len(trained_epochs),
            "final_loss": trained_epochs[-1]["train_loss"] if trained_epochs else None,
            "loss_per_epoch": [record["train_loss"] for record in trained_epochs],
            "duration_seconds": round(duration_seconds, 1)
            if duration_seconds
            else None,
        },
        # The historical layout, unchanged.
        "classification": results.get("classification", {}),
        "sts": results.get("sts", {}),
        "pair": results.get("pair", {}),
        "summary": results.get("summary", {}),
        "table": build_dimension_table(results),
    }
    if results.get("semantic"):
        report["semantic"] = results["semantic"]
    return report


def write_report(
    report: Dict[str, Any],
    output_dir: str | Path,
    stem: str = "results",
) -> Path:
    """Write ``<stem>.json`` plus a flat ``<stem>.csv``. Returns the JSON path."""
    output_dir = Path(output_dir)
    json_path = output_dir / f"{stem}.json"
    save_json(report, json_path)
    write_csv(report, output_dir / f"{stem}.csv")
    return json_path


def write_csv(report: Dict[str, Any], path: str | Path) -> None:
    """One row per Matryoshka dimension, one column per task."""
    table = report.get("table", {})
    if not table:
        return

    columns: List[str] = []
    for row in table.values():
        for key in row:
            if key not in columns:
                columns.append(key)
    columns.sort(key=lambda c: (c.startswith("semantic/"), c.startswith("mean/"), c))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dim"] + columns)
        for dim, row in table.items():
            writer.writerow(
                [dim.split("_")[1]]
                + [f"{row[c]:.4f}" if c in row else "" for c in columns]
            )


def format_summary(report: Dict[str, Any]) -> str:
    """A compact text table for the end of a training log."""
    table = report.get("table", {})
    if not table:
        return "no evaluation results"

    families = [c for c in next(iter(table.values())) if c.startswith("mean/")]
    if not families:
        return "no evaluation results"

    width = max(14, *(len(f.split("/")[1]) for f in families))
    header = f"{'dim':>6} | " + " | ".join(f"{f.split('/')[1]:>{width}}" for f in families)
    lines = [header, "-" * len(header)]
    for dim, row in table.items():
        cells = " | ".join(f"{row.get(f, float('nan')):>{width}.4f}" for f in families)
        lines.append(f"{dim.split('_')[1]:>6} | {cells}")
    return "\n".join(lines)
