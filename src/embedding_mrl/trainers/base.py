"""Shared training scaffold: model loading, optimiser, schedule, loop, evaluation."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from ..config import ExperimentConfig
from ..data import build_train_loader
from ..evaluation import MatryoshkaEvaluator
from ..reporting import build_report, format_summary, write_report
from ..utils import (
    autocast,
    count_trainable_parameters,
    cuda_memory_info,
    make_grad_scaler,
    pick_device,
    resolve_dtype,
    save_json,
    set_seed,
)

LOGGER = logging.getLogger("embedding_mrl.train")


class BaseTrainer(ABC):
    """Template for every method; subclasses only implement :meth:`compute_loss`."""

    method: str = "base"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        set_seed(cfg.train.seed)

        self.device = pick_device()
        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._attach_file_logger()
        self.epoch_index = -1
        self.step_in_epoch = 0
        self.global_step = 0

        LOGGER.info("Loading %s", cfg.model.name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        self.model = self._load_model()

        self._check_hidden_dim()

        self.train_loader: DataLoader = build_train_loader(
            self.tokenizer, cfg.data, cfg.train.batch_size
        )
        self.total_steps = len(self.train_loader) * cfg.train.epochs

        self.extra_modules: List[torch.nn.Module] = []
        self.setup_modules()

        self.optimizer = self.build_optimizer()
        self.scheduler = get_scheduler(
            name=cfg.train.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.total_steps * cfg.train.warmup_ratio),
            num_training_steps=self.total_steps,
            scheduler_specific_kwargs={"min_lr": cfg.train.min_lr}
            if cfg.train.scheduler == "cosine_with_min_lr"
            else {},
        )
        self.scaler = make_grad_scaler(cfg.train.fp16, self.device)

        self.evaluator = (
            MatryoshkaEvaluator(self.tokenizer, cfg, self.device)
            if cfg.eval.enabled
            else None
        )
        self.history: List[Dict[str, Any]] = []

        LOGGER.info(
            "method=%s | %d train pairs | %d steps/epoch | %d trainable params",
            self.method,
            len(self.train_loader.dataset),
            len(self.train_loader),
            count_trainable_parameters(self.model)
            + sum(count_trainable_parameters(m) for m in self.extra_modules),
        )

    def _load_model(self) -> torch.nn.Module:
        """Load the backbone, tolerating the ``torch_dtype`` -> ``dtype`` rename in transformers v5."""
        cfg = self.cfg.model
        kwargs: Dict[str, Any] = {
            "output_hidden_states": True,
            "trust_remote_code": cfg.trust_remote_code,
        }
        dtype = resolve_dtype(cfg.torch_dtype)
        if dtype is not None:
            try:
                return AutoModel.from_pretrained(
                    cfg.name_or_path, dtype=dtype, **kwargs
                ).to(self.device)
            except TypeError:
                kwargs["torch_dtype"] = dtype
        return AutoModel.from_pretrained(cfg.name_or_path, **kwargs).to(self.device)

    # -- hooks -------------------------------------------------------------- #
    def setup_modules(self) -> None:
        """Instantiate any trainable loss modules (MIPIC alignment heads)."""

    def on_train_start(self) -> None:
        """Hook called once after config persistence and before the first epoch."""

    def on_epoch_start(self, epoch_index: int) -> None:
        """Hook called before modules are switched back to training mode."""

    def on_after_backward(
        self,
        epoch_index: int,
        step: int,
        loss: torch.Tensor,
        logs: Dict[str, float],
    ) -> None:
        """Hook called with unscaled gradients before clipping and optimiser step."""

    def on_after_step(
        self,
        epoch_index: int,
        step: int,
        loss: torch.Tensor,
        logs: Dict[str, float],
        step_seconds: float,
    ) -> None:
        """Hook called after the optimiser and scheduler have completed a step."""

    def on_epoch_end(self, epoch_index: int, record: Dict[str, Any]) -> None:
        """Hook allowed to append method-specific values to the epoch record."""

    def method_report_metadata(self) -> Dict[str, Any]:
        """Small method-specific summary included in the final report."""
        return {}

    def progress_metrics(self, logs: Dict[str, float]) -> Dict[str, float]:
        """Metrics shown by tqdm; subclasses may keep a large log dict compact."""
        return logs

    @abstractmethod
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return ``(loss, log_dict)`` for one batch already moved to the device."""

    def build_optimizer(self) -> torch.optim.Optimizer:
        groups = [{"params": list(self.model.parameters()), "lr": self.cfg.train.lr}]
        groups.extend(self.extra_param_groups())
        return torch.optim.AdamW(groups, weight_decay=self.cfg.train.weight_decay)

    def extra_param_groups(self) -> List[Dict[str, Any]]:
        return []

    def trainable_modules(self) -> Iterable[torch.nn.Module]:
        yield self.model
        yield from self.extra_modules

    # -- internals ---------------------------------------------------------- #
    def _check_hidden_dim(self) -> None:
        actual = getattr(self.model.config, "hidden_size", None)
        if actual is not None and actual != self.cfg.model.hidden_dim:
            raise ValueError(
                f"config says model.hidden_dim={self.cfg.model.hidden_dim} but "
                f"{self.cfg.model.name_or_path} has hidden_size={actual}"
            )

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value
            if key == "sample_ids"
            else value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    def _attach_file_logger(self) -> None:
        """Persist the package log without adding duplicate handlers in tests."""
        path = (self.output_dir / "train.log").resolve()
        package_logger = logging.getLogger("embedding_mrl")
        for handler in list(package_logger.handlers):
            if not getattr(handler, "_embedding_mrl_run_file", False):
                continue
            if Path(handler.baseFilename).resolve() == path:
                return
            package_logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        handler._embedding_mrl_run_file = True  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        package_logger.addHandler(handler)

    def _set_train_mode(self) -> None:
        for module in self.trainable_modules():
            module.train()

    def _clip_gradients(self) -> None:
        max_norm = self.cfg.train.max_grad_norm
        if not max_norm:
            return
        for module in self.trainable_modules():
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm)

    # -- main loop ---------------------------------------------------------- #
    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.cfg.dump(self.output_dir / "config.yaml")
        last_results: Optional[Dict[str, Any]] = None
        run_started = time.time()
        self.on_train_start()

        for epoch in range(cfg.train.epochs):
            self.epoch_index = epoch
            self.on_epoch_start(epoch)
            self._set_train_mode()
            running_loss, seen = 0.0, 0
            running_metrics: Dict[str, float] = {}
            running_metric_weights: Dict[str, int] = {}
            started = time.time()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

            for step, raw_batch in enumerate(pbar, start=1):
                step_started = time.time()
                self.step_in_epoch = step
                self.global_step += 1
                batch = self._to_device(raw_batch)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(cfg.train.fp16, self.device):
                    loss, logs = self.compute_loss(batch)
                    loss = loss.float()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.on_after_backward(epoch, step, loss, logs)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.on_after_step(
                    epoch,
                    step,
                    loss,
                    logs,
                    time.time() - step_started,
                )

                batch_size = batch["input_ids1"].size(0)
                running_loss += loss.item() * batch_size
                seen += batch_size
                for key, value in logs.items():
                    numeric = float(value)
                    if math.isfinite(numeric):
                        running_metrics[key] = (
                            running_metrics.get(key, 0.0) + numeric * batch_size
                        )
                        running_metric_weights[key] = (
                            running_metric_weights.get(key, 0) + batch_size
                        )

                pbar.set_postfix(
                    {
                        "avg_loss": f"{running_loss / max(1, seen):.4f}",
                        **{
                            k: f"{v:.4f}"
                            for k, v in self.progress_metrics(logs).items()
                        },
                        **cuda_memory_info(),
                    }
                )

                if (
                    cfg.train.empty_cache_every
                    and step % cfg.train.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

            epoch_loss = running_loss / max(1, seen)
            LOGGER.info(
                "Epoch %d/%d done | loss %.4f | %.1fs",
                epoch + 1,
                cfg.train.epochs,
                epoch_loss,
                time.time() - started,
            )
            record: Dict[str, Any] = {"epoch": epoch + 1, "train_loss": epoch_loss}
            if running_metrics:
                record["train_metrics"] = {
                    key: value / running_metric_weights[key]
                    for key, value in sorted(running_metrics.items())
                }

            is_last = epoch == cfg.train.epochs - 1
            if self.evaluator and (cfg.eval.every_epoch or is_last):
                LOGGER.info(
                    "=== Evaluating after epoch %d (split=%s) ===",
                    epoch + 1,
                    cfg.eval.split,
                )
                last_results = self.evaluator.evaluate(self.model)
                save_json(
                    last_results, self.output_dir / f"results_epoch{epoch + 1}.json"
                )
                record["eval"] = last_results["summary"]

            self.on_epoch_end(epoch, record)
            self.history.append(record)
            save_json(self.history, self.output_dir / "history.json")

        if cfg.train.save_model:
            self.save()

        report = None
        if last_results is not None:
            report = self._write_report(last_results, time.time() - run_started)
        elif self.evaluator is None:
            LOGGER.warning(
                "Evaluation is disabled (eval.enabled=false) - no results.json written."
            )

        return {"history": self.history, "results": last_results, "report": report}

    def _write_report(
        self, results: Dict[str, Any], duration_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Assemble and persist ``results.json`` / ``results.csv``, then log the table."""
        report = build_report(
            self.cfg,
            results,
            self.history,
            duration_seconds,
            method_metadata=self.method_report_metadata(),
        )
        path = write_report(report, self.output_dir)
        LOGGER.info(
            "Final results for %s on the %s split (mean over tasks):\n%s",
            self.cfg.name,
            self.cfg.eval.split,
            format_summary(report),
        )
        LOGGER.info("Wrote %s and %s", path, path.with_suffix(".csv"))
        return report

    def evaluate_only(self) -> Dict[str, Any]:
        """Evaluate the current weights and write the same report a full run does."""
        if not self.evaluator:
            raise RuntimeError(
                "evaluation is disabled in this config (eval.enabled=false)"
            )
        started = time.time()
        results = self.evaluator.evaluate(self.model)
        return self._write_report(results, time.time() - started)

    def save(self) -> None:
        target = self.output_dir / "encoder"
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        for idx, module in enumerate(self.extra_modules):
            torch.save(module.state_dict(), self.output_dir / f"loss_module_{idx}.pt")
        LOGGER.info("Saved encoder to %s", target)
