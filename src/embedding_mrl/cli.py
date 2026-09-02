"""Command-line entry point: ``python -m embedding_mrl.cli --config <path>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig

LOGGER = logging.getLogger("embedding_mrl")


def _setup_logging(level: int) -> None:
    """Local copy so ``--print-config`` works without numpy/torch installed."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedding-mrl",
        description=(
            "Train / evaluate Matryoshka embedding models (MRL, ESE, MIPIC, GSR)."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="path to a YAML config under configs/"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override, e.g. --set train.epochs=1 --set model.pooling=mean",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training and only run the evaluation suite",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve the config, print it, and exit without loading any model",
    )
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = ExperimentConfig.load(args.config, args.overrides)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    if args.print_config:
        import yaml

        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0

    LOGGER.info("Experiment %s (method=%s)", cfg.name, cfg.method)
    from .trainers import (
        build_trainer,  # imported late so --print-config needs no torch
    )

    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.evaluate_only()
    else:
        trainer.train()

    LOGGER.info("Done. Artifacts in %s", Path(cfg.train.output_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
