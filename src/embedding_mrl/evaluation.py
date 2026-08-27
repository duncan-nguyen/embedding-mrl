"""Matryoshka evaluation suite: classification probe, STS, and pair classification.

Every task is scored at each nested dimension. Embeddings are extracted **once**
at full width and then truncated per dimension - mathematically identical to the
notebooks' per-dimension re-encoding, but ``len(dims)`` times cheaper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import build_pair_loader, build_single_text_loader, resolve_eval_paths
from .pooling import pool
from .utils import autocast

LOGGER = logging.getLogger("embedding_mrl.eval")


def pair_classification_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Sweep 200 thresholds for the best accuracy, then report the full metric set."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    best_acc, best_thr = 0.0, 0.0
    for threshold in np.linspace(0, 1, 200):
        acc = accuracy_score(labels, (scores >= threshold).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, threshold

    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": float(best_thr),
        "accuracy": float(best_acc),
        "f1": float(f1_score(labels, preds, average="macro")),
        "precision": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


class MatryoshkaEvaluator:
    """Runs the full suite for one encoder and returns a JSON-serialisable dict."""

    def __init__(self, tokenizer, cfg: ExperimentConfig, device: torch.device):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.dims: list[int] = cfg.matryoshka.ascending
        self.pooling = cfg.model.pooling
        self.paths = resolve_eval_paths(cfg.data, cfg.eval)

    # -- embedding extraction ---------------------------------------------- #
    @torch.no_grad()
    def _encode(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=False,
            return_dict=True,
        )
        return pool(
            outputs.last_hidden_state, attention_mask.to(self.device), self.pooling
        )

    @torch.no_grad()
    def _embed_pairs(
        self, model, loader, desc: str
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Full-width embeddings for both sides of every pair, plus the targets."""
        emb1_chunks, emb2_chunks, label_chunks = [], [], []
        with autocast(self.cfg.train.fp16, self.device):
            for batch in tqdm(loader, desc=desc, leave=False):
                emb1_chunks.append(
                    self._encode(model, batch["input_ids1"], batch["attention_mask1"])
                    .float()
                    .cpu()
                )
                emb2_chunks.append(
                    self._encode(model, batch["input_ids2"], batch["attention_mask2"])
                    .float()
                    .cpu()
                )
                label_chunks.append(batch["labels"].numpy())
        return (
            torch.cat(emb1_chunks),
            torch.cat(emb2_chunks),
            np.concatenate(label_chunks),
        )

    @torch.no_grad()
    def _embed_texts(self, model, loader, desc: str) -> tuple[np.ndarray, np.ndarray]:
        emb_chunks, label_chunks = [], []
        with autocast(self.cfg.train.fp16, self.device):
            for batch in tqdm(loader, desc=desc, leave=False):
                emb_chunks.append(
                    self._encode(model, batch["input_ids1"], batch["attention_mask1"])
                    .float()
                    .cpu()
                )
                label_chunks.append(batch["labels"].numpy())
        return torch.cat(emb_chunks).numpy(), np.concatenate(label_chunks)

    # -- individual task families ------------------------------------------ #
    def eval_sts(self, model, path: Path) -> dict[str, float]:
        """Spearman correlation between cosine similarity (rescaled to 0-5) and gold."""
        loader = build_pair_loader(
            path,
            self.tokenizer,
            self.cfg.eval.batch_size,
            self.cfg.eval.sts_max_length,
            "score",
        )
        emb1, emb2, labels = self._embed_pairs(model, loader, f"STS {path.stem}")

        results = {}
        for dim in self.dims:
            sim = torch.nn.functional.cosine_similarity(
                emb1[:, :dim], emb2[:, :dim]
            ).numpy()
            score = (sim + 1) * 2.5  # [-1, 1] -> [0, 5]
            corr, _ = spearmanr(score, labels)
            results[f"dim_{dim}"] = float(corr)
            LOGGER.info("    dim %-5d Spearman = %.4f", dim, corr)
        return results

    def eval_pair(self, model, path: Path) -> dict[str, dict[str, float]]:
        """Threshold-tuned accuracy/F1/AP on cosine similarity, per dimension."""
        loader = build_pair_loader(
            path,
            self.tokenizer,
            self.cfg.eval.batch_size,
            self.cfg.eval.sts_max_length,
            "label",
        )
        emb1, emb2, labels = self._embed_pairs(model, loader, f"Pair {path.stem}")

        results = {}
        for dim in self.dims:
            sim = torch.nn.functional.cosine_similarity(
                emb1[:, :dim], emb2[:, :dim]
            ).numpy()
            score = (sim + 1) / 2  # [-1, 1] -> [0, 1]
            metrics = pair_classification_metrics(score, labels)
            results[f"dim_{dim}"] = metrics
            LOGGER.info(
                "    dim %-5d acc = %.4f | f1 = %.4f | AP = %.4f",
                dim,
                metrics["accuracy"],
                metrics["f1"],
                metrics["average_precision"],
            )
        return results

    def eval_classification(
        self, model, train_path: Path, test_path: Path
    ) -> dict[str, dict[str, float]]:
        """Logistic-regression probe on frozen embeddings, per dimension."""
        train_loader = build_single_text_loader(
            train_path,
            self.tokenizer,
            self.cfg.eval.batch_size,
            self.cfg.eval.cls_max_length,
        )
        test_loader = build_single_text_loader(
            test_path,
            self.tokenizer,
            self.cfg.eval.batch_size,
            self.cfg.eval.cls_max_length,
        )
        x_train, y_train = self._embed_texts(
            model, train_loader, f"CLS train {train_path.stem}"
        )
        x_test, y_test = self._embed_texts(
            model, test_loader, f"CLS eval {test_path.stem}"
        )

        results = {}
        for dim in self.dims:
            clf = LogisticRegression(
                random_state=self.cfg.eval.logreg_seed,
                max_iter=self.cfg.eval.logreg_max_iter,
                verbose=0,
            )
            clf.fit(x_train[:, :dim], y_train)
            y_pred = clf.predict(x_test[:, :dim])

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="macro")),
            }
            results[f"dim_{dim}"] = metrics
            LOGGER.info(
                "    dim %-5d acc = %.4f | f1 = %.4f",
                dim,
                metrics["accuracy"],
                metrics["f1"],
            )
        return results

    # -- entry point -------------------------------------------------------- #
    def evaluate(self, model) -> dict[str, dict[str, object]]:
        """Run every configured task. Restores the model's original mode afterwards."""
        was_training = model.training
        model.eval()
        results: dict[str, dict[str, object]] = {
            "classification": {},
            "sts": {},
            "pair": {},
        }

        try:
            for name, train_path, test_path in self.paths["classification"]:
                LOGGER.info("  [cls]  %s", name)
                results["classification"][name] = self.eval_classification(
                    model, train_path, test_path
                )

            for name, path in self.paths["sts"]:
                LOGGER.info("  [sts]  %s", name)
                results["sts"][name] = self.eval_sts(model, path)

            for name, path in self.paths["pair"]:
                LOGGER.info("  [pair] %s", name)
                results["pair"][name] = self.eval_pair(model, path)
        finally:
            if was_training:
                model.train()

        results["summary"] = summarize(results)
        return results


def summarize(results: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    """Average the headline metric of each family across tasks, per dimension."""
    summary: dict[str, dict[str, float]] = {}

    def collect(family: str, extract) -> None:
        per_dim: dict[str, list[float]] = {}
        for task_scores in results.get(family, {}).values():
            for dim, value in task_scores.items():
                per_dim.setdefault(dim, []).append(extract(value))
        if per_dim:
            summary[family] = {
                dim: float(np.mean(vals)) for dim, vals in per_dim.items()
            }

    collect("classification", lambda v: v["accuracy"])
    collect("sts", lambda v: v)
    collect("pair", lambda v: v["accuracy"])
    return summary
