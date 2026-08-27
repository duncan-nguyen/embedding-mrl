"""Matryoshka embedding experiments (MRL / ESE / MIPIC / SDR-MRL) as an importable package."""

from .config import ExperimentConfig

__version__ = "0.1.0"
__all__ = ["ExperimentConfig", "build_trainer"]


def build_trainer(cfg: "ExperimentConfig"):
    """Lazy re-export so ``import embedding_mrl`` does not pull in torch."""
    from .trainers import build_trainer as _build

    return _build(cfg)
