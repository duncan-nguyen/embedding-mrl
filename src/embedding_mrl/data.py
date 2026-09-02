"""Datasets, collators and the local CSV task registry.

Every path the notebooks read from ``/kaggle/input/...`` is resolved here against
the repo-local ``data/`` directory instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, EvalConfig

# --------------------------------------------------------------------------- #
# Task registry: friendly name -> CSV file names under ``data/test/``
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClsTask:
    """Logistic-regression probe: fit on ``train_file``, score on the eval split."""

    name: str
    train_file: str
    validation_file: str | None
    test_file: str | None

    def eval_file(self, split: str) -> str:
        file = self.validation_file if split == "validation" else self.test_file
        if file is None:
            raise KeyError(f"classification task {self.name!r} has no {split!r} split")
        return file


@dataclass(frozen=True)
class PairFileTask:
    """A sentence-pair task (STS regression or pair classification)."""

    name: str
    validation_file: str | None
    test_file: str | None

    def eval_file(self, split: str) -> str:
        file = self.validation_file if split == "validation" else self.test_file
        if file is None:
            # Several STS sets ship a single file; fall back to it either way.
            file = self.test_file or self.validation_file
        if file is None:
            raise KeyError(f"task {self.name!r} has no usable split")
        return file


CLS_TASKS: dict[str, ClsTask] = {
    "banking77": ClsTask(
        "banking77",
        "banking_train.csv",
        "banking77_validation.csv",
        "banking77_test.csv",
    ),
    "emotion": ClsTask(
        "emotion", "emotion_train.csv", "emotion_validation.csv", "emotion_test.csv"
    ),
    "tweet": ClsTask(
        "tweet", "tweet_train.csv", "tweet_validation.csv", "tweet_test.csv"
    ),
}

STS_TASKS: dict[str, PairFileTask] = {
    "sick": PairFileTask("sick", "sick_validation.csv", "sick_test.csv"),
    "sts12": PairFileTask("sts12", "sts12_validation.csv", "sts12_test.csv"),
    "stsb": PairFileTask("stsb", "stsb_validation.csv", "stsb_test.csv"),
    # Single-file STS sets (no separate validation split shipped in data/test).
    "sick_r": PairFileTask("sick_r", None, "sick_r.csv"),
    "sts13": PairFileTask("sts13", None, "sts13.csv"),
    "sts14": PairFileTask("sts14", None, "sts14.csv"),
    "sts15": PairFileTask("sts15", None, "sts15.csv"),
    "sts16": PairFileTask("sts16", None, "sts16.csv"),
}

PAIR_TASKS: dict[str, PairFileTask] = {
    "mrpc": PairFileTask("mrpc", "mrpc_validation.csv", "mrpc_test.csv"),
    "scitail": PairFileTask("scitail", "scitail_validation.csv", "scitail_test.csv"),
    "wic": PairFileTask("wic", "wic_validation.csv", "wic_test.csv"),
    # Available in data/test but not part of the notebooks' default suite.
    "rte": PairFileTask(
        "rte", "rte_validaion.csv", "rte_test.csv"
    ),  # sic: upstream typo
    "qnli": PairFileTask("qnli", "qnli_validation.csv", "qnli_test.csv"),
}


def resolve_eval_paths(data_cfg: DataConfig, eval_cfg: EvalConfig) -> dict[str, list]:
    """Turn task names in the config into concrete, existing file paths."""
    root = data_cfg.test_path

    cls_paths = []
    for name in eval_cfg.cls_tasks:
        task = _lookup(CLS_TASKS, name, "classification")
        cls_paths.append(
            (task.name, root / task.train_file, root / task.eval_file(eval_cfg.split))
        )

    sts_paths = [
        (task.name, root / task.eval_file(eval_cfg.split))
        for task in (_lookup(STS_TASKS, n, "STS") for n in eval_cfg.sts_tasks)
    ]
    pair_paths = [
        (task.name, root / task.eval_file(eval_cfg.split))
        for task in (_lookup(PAIR_TASKS, n, "pair") for n in eval_cfg.pair_tasks)
    ]

    for _, *paths in cls_paths:
        _require(paths)
    _require([p for _, p in sts_paths])
    _require([p for _, p in pair_paths])

    return {"classification": cls_paths, "sts": sts_paths, "pair": pair_paths}


def _lookup(registry: dict[str, object], name: str, kind: str):
    try:
        return registry[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown {kind} task {name!r}; known: {sorted(registry)}"
        ) from exc


def _require(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing evaluation CSV(s): " + ", ".join(missing))


# --------------------------------------------------------------------------- #
# Training data (unsupervised SimCSE: the same sentence forms both views)
# --------------------------------------------------------------------------- #
class SimCSEPairDataset(Dataset):
    """Yields ``(sentence, sentence)`` - dropout inside the encoder makes the views differ."""

    def __init__(self, frame: pd.DataFrame, text_column: str = "text"):
        if text_column not in frame.columns:
            raise KeyError(
                f"train file needs a {text_column!r} column, found {list(frame.columns)}"
            )
        texts = frame[text_column].astype(str).tolist()
        self.samples: list[tuple[str, str]] = [(t, t) for t in texts]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.samples[idx]


class SimCSECollator:
    """Tokenise both views with the student tokenizer, padding to the longest item."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        view1, view2 = zip(*batch)
        enc1 = self._encode(list(view1))
        enc2 = self._encode(list(view2))

        out = {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],
            "input_ids2": enc2["input_ids"],
            "attention_mask2": enc2["attention_mask"],
        }
        # Only BERT-style tokenizers emit these; Qwen3 does not.
        if "token_type_ids" in enc1:
            out["token_type_ids1"] = enc1["token_type_ids"]
        if "token_type_ids" in enc2:
            out["token_type_ids2"] = enc2["token_type_ids"]
        return out

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )


def build_train_loader(
    tokenizer, data_cfg: DataConfig, batch_size: int, shuffle: bool = True
) -> DataLoader:
    path = data_cfg.train_path
    if not path.exists():
        raise FileNotFoundError(f"training file not found: {path}")
    frame = pd.read_csv(path)
    if data_cfg.max_train_samples is not None:
        frame = frame.head(data_cfg.max_train_samples)

    dataset = SimCSEPairDataset(frame, data_cfg.text_column)
    collator = SimCSECollator(tokenizer, data_cfg.max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=data_cfg.num_workers,
        persistent_workers=data_cfg.num_workers > 0,
        drop_last=False,
    )


# --------------------------------------------------------------------------- #
# Evaluation data
# --------------------------------------------------------------------------- #
class SentencePairDataset(Dataset):
    """``sentence1`` / ``sentence2`` plus a float target (STS score or 0/1 label)."""

    def __init__(self, file_path: str | Path, label_column: str):
        frame = pd.read_csv(file_path)
        for column in ("sentence1", "sentence2", label_column):
            if column not in frame.columns:
                raise KeyError(
                    f"{file_path} needs a {column!r} column, found {list(frame.columns)}"
                )
        self.sentence1 = frame["sentence1"].astype(str).tolist()
        self.sentence2 = frame["sentence2"].astype(str).tolist()
        self.labels = frame[label_column].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class SingleTextDataset(Dataset):
    """``text`` plus an integer class label, for the logistic-regression probe."""

    def __init__(
        self,
        file_path: str | Path,
        text_column: str = "text",
        label_column: str = "label",
    ):
        frame = pd.read_csv(file_path)
        for column in (text_column, label_column):
            if column not in frame.columns:
                raise KeyError(
                    f"{file_path} needs a {column!r} column, found {list(frame.columns)}"
                )
        self.texts = frame[text_column].astype(str).tolist()
        self.labels = frame[label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PairEvalCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc1 = self._encode([item["sentence1"] for item in batch])
        enc2 = self._encode([item["sentence2"] for item in batch])
        return {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],
            "input_ids2": enc2["input_ids"],
            "attention_mask2": enc2["attention_mask"],
            "labels": torch.stack([item["label"] for item in batch]),
        }

    def _encode(self, texts):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class SingleTextCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc = self.tokenizer(
            [item["text"] for item in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids1": enc["input_ids"],
            "attention_mask1": enc["attention_mask"],
            "labels": torch.stack([item["label"] for item in batch]),
        }


def build_pair_loader(
    path, tokenizer, batch_size: int, max_length: int, label_column: str
) -> DataLoader:
    dataset = SentencePairDataset(path, label_column=label_column)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PairEvalCollator(tokenizer, max_length),
    )


def build_single_text_loader(
    path, tokenizer, batch_size: int, max_length: int
) -> DataLoader:
    dataset = SingleTextDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SingleTextCollator(tokenizer, max_length),
    )
