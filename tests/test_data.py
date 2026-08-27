"""The task registry must resolve to files that actually exist in data/."""

from pathlib import Path

import pytest

from embedding_mrl.config import DataConfig, EvalConfig
from embedding_mrl.data import CLS_TASKS, PAIR_TASKS, STS_TASKS, resolve_eval_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = DataConfig().resolve(REPO_ROOT)


@pytest.mark.parametrize("split", ["test", "validation"])
def test_default_task_suite_resolves(split):
    eval_cfg = EvalConfig(split=split)
    if split == "validation":
        # sick_r / sts13-16 ship a single file; keep the notebook's validation suite.
        eval_cfg.sts_tasks = ["sick", "sts12", "stsb"]
    paths = resolve_eval_paths(DATA, eval_cfg)
    assert [name for name, *_ in paths["classification"]] == ["banking77", "emotion", "tweet"]
    for _, *files in paths["classification"]:
        assert all(Path(f).exists() for f in files)


def test_every_registered_file_exists():
    root = DATA.test_path
    files = []
    for task in CLS_TASKS.values():
        files += [task.train_file, task.validation_file, task.test_file]
    for task in list(STS_TASKS.values()) + list(PAIR_TASKS.values()):
        files += [task.validation_file, task.test_file]
    missing = [f for f in files if f is not None and not (root / f).exists()]
    assert missing == []


def test_unknown_task_name_is_rejected():
    with pytest.raises(KeyError, match="unknown STS task"):
        resolve_eval_paths(DATA, EvalConfig(sts_tasks=["sts99"], cls_tasks=[], pair_tasks=[]))
