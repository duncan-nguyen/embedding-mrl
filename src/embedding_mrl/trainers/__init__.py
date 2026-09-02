"""Trainer registry, keyed by ``ExperimentConfig.method``."""

from ..config import ExperimentConfig
from .base import BaseTrainer
from .ese import ESETrainer
from .gsr import GSRTrainer
from .mipic import MIPICTrainer
from .mrl import MRLTrainer

TRAINERS = {
    "mrl": MRLTrainer,
    "ese": ESETrainer,
    "mipic": MIPICTrainer,
    "gsr": GSRTrainer,
}


def build_trainer(cfg: ExperimentConfig) -> BaseTrainer:
    try:
        trainer_cls = TRAINERS[cfg.method]
    except KeyError as exc:
        raise ValueError(f"unknown method {cfg.method!r}; known: {sorted(TRAINERS)}") from exc
    return trainer_cls(cfg)


__all__ = [
    "BaseTrainer",
    "MRLTrainer",
    "ESETrainer",
    "MIPICTrainer",
    "GSRTrainer",
    "TRAINERS",
    "build_trainer",
]
