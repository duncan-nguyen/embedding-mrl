"""GSR trainer: MRL task supervision plus frozen global spectral geometry."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data import build_teacher_loader
from ..gsr_teacher import (
    SpectralTeacherCache,
    build_spectral_teacher_cache,
    encode_teacher_corpus,
)
from ..losses.gsr import GSRShellLossOutput, full_normalize, gsr_shell_loss
from ..losses.infonce import info_nce, matryoshka_info_nce
from ..pooling import pool
from ..utils import save_json
from .base import BaseTrainer

LOGGER = logging.getLogger("embedding_mrl.train.gsr")


class GSRTrainer(BaseTrainer):
    """Alternating corpus-teacher refresh and minibatch U-statistic training."""

    method = "gsr"

    def setup_modules(self) -> None:
        cfg = self.cfg.gsr
        self.teacher_loader = build_teacher_loader(
            self.tokenizer, self.cfg.data, cfg.teacher_batch_size
        )
        panel_size = min(cfg.diagnostic_samples, len(self.teacher_loader.dataset))
        generator = torch.Generator().manual_seed(self.cfg.train.seed)
        self.diagnostic_ids = torch.randperm(
            len(self.teacher_loader.dataset), generator=generator
        )[:panel_size].sort().values
        panel_dataset = Subset(
            self.teacher_loader.dataset, self.diagnostic_ids.tolist()
        )
        self.diagnostic_loader = DataLoader(
            panel_dataset,
            batch_size=min(cfg.teacher_batch_size, max(1, panel_size)),
            shuffle=False,
            collate_fn=self.teacher_loader.collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            drop_last=False,
        )
        self.teacher_cache: SpectralTeacherCache | None = None
        self.teacher_refresh_count = 0
        self.teacher_history: list[dict[str, Any]] = []
        self._pending_step_record: dict[str, Any] | None = None
        self._failure_context: dict[str, Any] = {}
        self._previous_panel_student: dict[str, torch.Tensor] = {}
        self.diagnostics_dir = self.output_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        pair_count = panel_size * (panel_size - 1) // 2
        selected_pairs = min(pair_count, cfg.diagnostic_pairs)
        self.diagnostic_pair_indices = torch.randperm(
            pair_count, generator=generator
        )[:selected_pairs].sort().values

    @property
    def geometry_dims(self) -> list[int]:
        return list(self.cfg.gsr.geometry_dims or self.cfg.matryoshka.ascending)

    def on_train_start(self) -> None:
        # A stale JSONL from an interrupted run must not be mixed with this run.
        self.steps_path = self.diagnostics_dir / "steps.jsonl"
        self.steps_path.write_text("", encoding="utf-8")
        save_json(
            {
                "geometry_dims": self.geometry_dims,
                "warmup_epochs": self.cfg.gsr.warmup_epochs,
                "refresh_every_epochs": self.cfg.gsr.refresh_every_epochs,
                "diagnostics_every_steps": self.cfg.gsr.diagnostics_every_steps,
                "diagnostic_samples": len(self.diagnostic_loader.dataset),
                "diagnostic_sample_ids": self.diagnostic_ids.tolist(),
                "diagnostic_pair_indices": self.diagnostic_pair_indices.tolist(),
            },
            self.diagnostics_dir / "manifest.json",
        )

    def on_epoch_start(self, epoch_index: int) -> None:
        cfg = self.cfg.gsr
        if cfg.weight == 0 or epoch_index < cfg.warmup_epochs:
            LOGGER.info(
                "GSR inactive at epoch %d (warmup=%d, weight=%.3g)",
                epoch_index + 1,
                cfg.warmup_epochs,
                cfg.weight,
            )
            return
        due = (
            self.teacher_cache is None
            or (epoch_index - cfg.warmup_epochs) % cfg.refresh_every_epochs == 0
        )
        if due:
            self._refresh_teacher(epoch_index)

    def _refresh_teacher(self, epoch_index: int) -> None:
        cfg = self.cfg.gsr
        LOGGER.info(
            "Building GSR teacher refresh %d from %d corpus rows",
            self.teacher_refresh_count,
            len(self.teacher_loader.dataset),
        )
        try:
            embeddings, encode_diagnostics = encode_teacher_corpus(
                self.model,
                self.teacher_loader,
                pooling=self.cfg.model.pooling,
                hidden_dim=self.cfg.model.hidden_dim,
                device=self.device,
                fp16=self.cfg.train.fp16,
            )
            new_cache = build_spectral_teacher_cache(
                embeddings,
                self.geometry_dims,
                eigengap_tolerance=cfg.eigengap_tolerance,
                eps=cfg.eps,
                refresh_index=self.teacher_refresh_count,
                source_epoch=epoch_index,
                merge_ties=cfg.merge_tied_shells,
            )
        except Exception as exc:
            self._save_failure(
                f"teacher_refresh:{type(exc).__name__}:{exc}",
                extra={"epoch": epoch_index + 1},
                stem=f"failure_teacher_epoch{epoch_index + 1}",
            )
            raise

        diagnostics = dict(new_cache.diagnostics)
        diagnostics["encoding"] = encode_diagnostics
        diagnostics["pooling"] = self.cfg.model.pooling
        diagnostics["cache_dtype"] = self.cfg.gsr.cache_dtype
        diagnostics["drift_from_previous"] = self._teacher_drift(
            self.teacher_cache, new_cache
        )
        new_cache.diagnostics = diagnostics
        self.teacher_cache = new_cache
        self.teacher_history.append(diagnostics)

        epoch_number = epoch_index + 1
        save_json(
            diagnostics, self.diagnostics_dir / f"teacher_epoch{epoch_number}.json"
        )
        if cfg.save_teacher_tensors:
            torch.save(
                new_cache.tensor_payload(),
                self.diagnostics_dir / f"teacher_epoch{epoch_number}.pt",
            )
        refresh_panel = self._panel_diagnostics_from_embeddings(
            embeddings.index_select(0, self.diagnostic_ids),
            epoch_index,
            stage="teacher_refresh",
            encoding={"source": "full_teacher_pass", **encode_diagnostics},
        )
        save_json(
            refresh_panel,
            self.diagnostics_dir / f"geometry_refresh_epoch{epoch_number}.json",
        )
        self.teacher_refresh_count += 1
        LOGGER.info(
            "GSR teacher ready | shells=%s | cT=%.6g | rank=%.2f | %.1fs",
            new_cache.shells,
            new_cache.c_teacher,
            diagnostics["effective_rank"],
            diagnostics["build_seconds"] + encode_diagnostics["encode_seconds"],
        )

    def _teacher_drift(
        self,
        old: SpectralTeacherCache | None,
        new: SpectralTeacherCache,
    ) -> dict[str, Any] | None:
        if old is None:
            return None
        eps = self.cfg.gsr.eps
        eig_relative = float(
            (new.eigenvalues - old.eigenvalues).norm()
            / old.eigenvalues.norm().clamp_min(eps)
        )
        subspace: dict[str, float] = {}
        for dim in self.geometry_dims[:-1]:
            overlap = old.eigenvectors[:, :dim].T @ new.eigenvectors[:, :dim]
            # Projector distance / sqrt(2d), invariant to sign and rotations
            # inside repeated-eigenvalue subspaces.
            residual = (dim - overlap.square().sum()).clamp_min(0) / dim
            subspace[f"dim_{dim}"] = float(residual.sqrt())
        return {
            "mean_l2": float((new.mean - old.mean).norm()),
            "eigenvalues_relative_l2": eig_relative,
            "subspace_projector_distance": subspace,
        }

    def _forward_view(
        self, batch: Dict[str, torch.Tensor], suffix: str
    ) -> torch.Tensor:
        inputs = {
            "input_ids": batch[f"input_ids{suffix}"],
            "attention_mask": batch[f"attention_mask{suffix}"],
            "output_hidden_states": True,
            "return_dict": True,
        }
        token_type_key = f"token_type_ids{suffix}"
        if token_type_key in batch:
            inputs["token_type_ids"] = batch[token_type_key]
        outputs = self.model(**inputs)
        return pool(
            outputs.hidden_states[-1],
            batch[f"attention_mask{suffix}"],
            self.cfg.model.pooling,
        )

    def progress_metrics(self, logs: Dict[str, float]) -> Dict[str, float]:
        return {
            key: logs[key]
            for key in ("loss/total", "loss/gsr", "gsr/active")
            if key in logs
        }

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        emb1 = self._forward_view(batch, "1")
        emb2 = self._forward_view(batch, "2")
        matry_loss, logits = matryoshka_info_nce(
            emb1,
            emb2,
            nested_dims=self.cfg.matryoshka.ascending,
            temperature=self.cfg.matryoshka.temperature,
        )
        task_loss, _ = info_nce(
            emb1, emb2, temperature=self.cfg.matryoshka.temperature
        )
        supervised = (
            self.cfg.mrl.w_matryoshka * matry_loss
            + self.cfg.mrl.w_task * task_loss
        )
        logs: Dict[str, float] = {
            "loss/matryoshka": float(matry_loss.detach()),
            "loss/task_full": float(task_loss.detach()),
            "loss/gsr": 0.0,
            "loss/gsr_weighted": 0.0,
            "gsr/active": 0.0,
            "gsr/effective_weight": 0.0,
        }
        labels = torch.arange(emb1.size(0), device=emb1.device)
        for key, value in logits.items():
            logs[f"task/{key}_loss"] = float(F.cross_entropy(value, labels).detach())
            logs[f"task/{key}_accuracy"] = float(
                (value.argmax(dim=1) == labels).float().mean().detach()
            )

        cache = self.teacher_cache
        geometry_loss = emb1.new_zeros((), dtype=torch.float32)
        outputs: tuple[GSRShellLossOutput, GSRShellLossOutput] | None = None
        diagnostic_step = (
            self.global_step % self.cfg.gsr.diagnostics_every_steps == 0
        )
        if cache is not None and self.cfg.gsr.weight > 0 and emb1.size(0) >= 2:
            teacher_scores = cache.lookup(batch["sample_ids"], self.device)
            normalized1 = full_normalize(emb1, eps=self.cfg.gsr.eps)
            normalized2 = full_normalize(emb2, eps=self.cfg.gsr.eps)
            gsr1 = gsr_shell_loss(
                normalized1,
                teacher_scores,
                cache.shells,
                cache.c_teacher,
                eps=self.cfg.gsr.eps,
            )
            gsr2 = gsr_shell_loss(
                normalized2,
                teacher_scores,
                cache.shells,
                cache.c_teacher,
                eps=self.cfg.gsr.eps,
            )
            outputs = (gsr1, gsr2)
            geometry_loss = 0.5 * (gsr1.total_loss + gsr2.total_loss)
            logs["gsr/active"] = 1.0
            logs["loss/gsr"] = float(geometry_loss.detach())
            logs["loss/gsr_weighted"] = float(
                (self.cfg.gsr.weight * geometry_loss).detach()
            )
            logs["gsr/effective_weight"] = self.cfg.gsr.weight
            for key in gsr1.shell_losses:
                logs[f"gsr/{key}_loss"] = float(
                    0.5
                    * (
                        gsr1.shell_losses[key].detach()
                        + gsr2.shell_losses[key].detach()
                    )
                )
            if diagnostic_step:
                logs.update(self._geometry_metrics(outputs, cache.c_teacher))
                logs.update(self._alignment_metrics(outputs))
        elif cache is not None and emb1.size(0) < 2:
            logs["gsr/skipped_singleton"] = 1.0

        loss = supervised.float() + self.cfg.gsr.weight * geometry_loss
        logs["loss/supervised"] = float(supervised.detach())
        logs["loss/total"] = float(loss.detach())

        if not torch.isfinite(loss):
            self._failure_context = {
                "sample_ids": batch["sample_ids"].detach().cpu(),
                "emb1": emb1.detach().cpu(),
                "emb2": emb2.detach().cpu(),
                "teacher_scores": teacher_scores.detach().cpu()
                if outputs is not None
                else None,
                "logs": logs,
            }
            self._save_failure("nonfinite_loss")
            raise FloatingPointError(
                f"non-finite GSR training loss at step {self.global_step}"
            )

        if self.cfg.gsr.fail_on_nonfinite:
            self._failure_context = {
                "sample_ids": batch["sample_ids"].detach(),
                "emb1": emb1.detach(),
                "emb2": emb2.detach(),
                "teacher_scores": teacher_scores.detach()
                if outputs is not None
                else None,
                "logs": logs,
            }

        if diagnostic_step:
            conflict = (
                self._gradient_conflict(supervised, geometry_loss, (emb1, emb2))
                if outputs is not None
                else {}
            )
            self._pending_step_record = {
                "epoch": self.epoch_index + 1,
                "step_in_epoch": self.step_in_epoch,
                "global_step": self.global_step,
                "sample_ids": batch["sample_ids"].detach().cpu().tolist(),
                "batch_size": emb1.size(0),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "embedding": self._embedding_metrics(emb1, emb2),
                "metrics": dict(logs),
                "gradient_conflict": conflict,
                "teacher_refresh_index": cache.refresh_index if cache else None,
            }
        return loss, logs

    def _gradient_conflict(
        self,
        task_loss: torch.Tensor,
        geometry_loss: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        task_grads = torch.autograd.grad(
            task_loss, embeddings, retain_graph=True, allow_unused=True
        )
        geometry_grads = torch.autograd.grad(
            geometry_loss, embeddings, retain_graph=True, allow_unused=True
        )
        task_parts = [g.float().reshape(-1) for g in task_grads if g is not None]
        geo_parts = [g.float().reshape(-1) for g in geometry_grads if g is not None]
        if not task_parts or not geo_parts:
            return {}
        task = torch.cat(task_parts)
        geometry = torch.cat(geo_parts)
        task_norm = task.norm()
        geometry_norm = geometry.norm()
        cosine = (task @ geometry) / (
            task_norm * geometry_norm + self.cfg.gsr.eps
        )
        return {
            "cosine": float(cosine.detach()),
            "task_norm": float(task_norm.detach()),
            "geometry_norm": float(geometry_norm.detach()),
            "norm_ratio": float(
                (geometry_norm / task_norm.clamp_min(self.cfg.gsr.eps)).detach()
            ),
        }

    def _embedding_metrics(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> dict[str, float]:
        norms = torch.cat((emb1.float().norm(dim=1), emb2.float().norm(dim=1)))
        values = torch.cat((emb1.float().reshape(-1), emb2.float().reshape(-1)))
        normalized = torch.cat(
            (
                full_normalize(emb1, eps=self.cfg.gsr.eps),
                full_normalize(emb2, eps=self.cfg.gsr.eps),
            )
        )
        coordinate_variance = normalized.var(dim=0, unbiased=False)
        result = {
            "norm_min": float(norms.min().detach()),
            "norm_mean": float(norms.mean().detach()),
            "norm_max": float(norms.max().detach()),
            "coordinate_mean": float(values.mean().detach()),
            "coordinate_std": float(values.std(unbiased=False).detach()),
            "coordinate_variance_min": float(coordinate_variance.min().detach()),
            "coordinate_variance_max": float(coordinate_variance.max().detach()),
            "dead_coordinate_fraction": float(
                (coordinate_variance <= self.cfg.gsr.eps).float().mean().detach()
            ),
        }
        previous = 0
        for dim in self.cfg.matryoshka.ascending:
            result[f"prefix_energy_dim_{dim}"] = float(
                normalized[:, :dim].square().sum(dim=1).mean().detach()
            )
            result[f"band_energy_dim_{previous}_{dim}"] = float(
                normalized[:, previous:dim].square().sum(dim=1).mean().detach()
            )
            previous = dim
        return result

    def _geometry_metrics(
        self,
        outputs: Iterable[GSRShellLossOutput],
        c_teacher: float,
        *,
        pair_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        outputs = tuple(outputs)
        metrics: dict[str, float] = {}
        cumulative_student: list[torch.Tensor] | None = None
        cumulative_teacher: list[torch.Tensor] | None = None
        for key in outputs[0].shell_losses:
            student = torch.cat(
                [out.student_distances[key].detach() for out in outputs]
            )
            teacher = torch.cat(
                [out.teacher_distances[key].detach() for out in outputs]
            )
            total_pairs = student.numel()
            if pair_indices is not None:
                indices = pair_indices.to(student.device)
                student = student.index_select(0, indices)
                teacher = teacher.index_select(0, indices)
            difference = student - teacher
            s_centered = student - student.mean()
            t_centered = teacher - teacher.mean()
            correlation = (s_centered @ t_centered) / (
                s_centered.norm() * t_centered.norm() + self.cfg.gsr.eps
            )
            prefix = f"gsr/{key}"
            metrics[f"{prefix}_loss"] = float(
                torch.stack([out.shell_losses[key].detach() for out in outputs]).mean()
            )
            metrics[f"{prefix}_energy_ratio"] = float(
                student.mean() / teacher.mean().clamp_min(self.cfg.gsr.eps)
            )
            metrics[f"{prefix}_bias"] = float(
                difference.mean() / math.sqrt(c_teacher + self.cfg.gsr.eps)
            )
            metrics[f"{prefix}_rmse"] = float(
                difference.square().mean().sqrt()
                / math.sqrt(c_teacher + self.cfg.gsr.eps)
            )
            metrics[f"{prefix}_correlation"] = float(correlation)
            metrics[f"{prefix}_pairs"] = float(total_pairs)

            if cumulative_student is None:
                cumulative_student = [
                    out.student_distances[key].detach().clone() for out in outputs
                ]
                cumulative_teacher = [
                    out.teacher_distances[key].detach().clone() for out in outputs
                ]
            else:
                for index, out in enumerate(outputs):
                    cumulative_student[index].add_(out.student_distances[key].detach())
                    cumulative_teacher[index].add_(out.teacher_distances[key].detach())
            cumulative_student_vector = torch.cat(cumulative_student)
            cumulative_teacher_vector = torch.cat(cumulative_teacher)
            if pair_indices is not None:
                indices = pair_indices.to(cumulative_student_vector.device)
                cumulative_student_vector = cumulative_student_vector.index_select(
                    0, indices
                )
                cumulative_teacher_vector = cumulative_teacher_vector.index_select(
                    0, indices
                )
            cumulative_difference = (
                cumulative_student_vector - cumulative_teacher_vector
            )
            metrics[f"{prefix}_cumulative_rmse"] = float(
                cumulative_difference.square().mean().sqrt()
                / math.sqrt(c_teacher + self.cfg.gsr.eps)
            )
        return metrics

    def _alignment_metrics(
        self,
        outputs: Iterable[GSRShellLossOutput],
        *,
        pair_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Correlation matrix from student coordinate bands to teacher shells."""
        outputs = tuple(outputs)
        keys = list(outputs[0].shell_losses)
        matrix: dict[tuple[str, str], float] = {}
        for student_key in keys:
            student = torch.cat(
                [out.student_distances[student_key].detach() for out in outputs]
            )
            if pair_indices is not None:
                indices = pair_indices.to(student.device)
                student = student.index_select(0, indices)
            student = student - student.mean()
            for teacher_key in keys:
                teacher = torch.cat(
                    [out.teacher_distances[teacher_key].detach() for out in outputs]
                )
                if pair_indices is not None:
                    teacher = teacher.index_select(0, indices)
                teacher = teacher - teacher.mean()
                value = (student @ teacher) / (
                    student.norm() * teacher.norm() + self.cfg.gsr.eps
                )
                matrix[(student_key, teacher_key)] = float(value)

        result = {
            f"gsr/alignment_student_{student}_teacher_{teacher}": value
            for (student, teacher), value in matrix.items()
        }
        diagonal = [abs(matrix[(key, key)]) for key in keys]
        off_diagonal = [
            abs(value)
            for (student, teacher), value in matrix.items()
            if student != teacher
        ]
        result["gsr/alignment_diagonal_mean"] = sum(diagonal) / len(diagonal)
        result["gsr/alignment_off_diagonal_mean"] = (
            sum(off_diagonal) / len(off_diagonal) if off_diagonal else 0.0
        )
        margins = []
        for key in keys:
            competitors = [
                abs(matrix[(key, other)]) for other in keys if other != key
            ]
            margins.append(
                abs(matrix[(key, key)]) - (max(competitors) if competitors else 0.0)
            )
        result["gsr/alignment_margin_mean"] = sum(margins) / len(margins)
        return result

    def on_after_backward(
        self,
        epoch_index: int,
        step: int,
        loss: torch.Tensor,
        logs: Dict[str, float],
    ) -> None:
        del epoch_index, step, loss, logs
        should_measure = (
            self.cfg.gsr.fail_on_nonfinite or self._pending_step_record is not None
        )
        if not should_measure:
            return

        total_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        max_abs = torch.zeros((), device=self.device, dtype=torch.float32)
        nonfinite_names: list[str] = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            if not torch.isfinite(grad).all():
                nonfinite_names.append(name)
            total_sq.add_(grad.square().sum())
            max_abs = torch.maximum(max_abs, grad.abs().max())

        gradient_metrics = {
            "total_norm": float(total_sq.sqrt()),
            "max_abs": float(max_abs),
            "nonfinite_parameter_count": len(nonfinite_names),
        }
        if nonfinite_names and self.cfg.gsr.fail_on_nonfinite:
            self._save_failure(
                "nonfinite_gradient",
                extra={"nonfinite_parameters": nonfinite_names},
            )
            raise FloatingPointError(
                "non-finite gradients at step "
                f"{self.global_step}: {nonfinite_names[:5]}"
            )

        if self._pending_step_record is not None:
            self._pending_step_record["gradient"] = gradient_metrics
            if self.device.type == "cuda":
                self._pending_step_record["cuda_memory_mib"] = {
                    "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,
                    "reserved": torch.cuda.memory_reserved(self.device) / 1024**2,
                    "max_allocated": torch.cuda.max_memory_allocated(self.device)
                    / 1024**2,
                }

    def on_after_step(
        self,
        epoch_index: int,
        step: int,
        loss: torch.Tensor,
        logs: Dict[str, float],
        step_seconds: float,
    ) -> None:
        del epoch_index, step, loss, logs
        if self._pending_step_record is not None:
            self._pending_step_record["step_seconds"] = step_seconds
            self._pending_step_record["amp_scale"] = float(self.scaler.get_scale())
            self._append_jsonl(self._pending_step_record)
            self._pending_step_record = None
        self._failure_context = {}

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def on_epoch_end(self, epoch_index: int, record: Dict[str, Any]) -> None:
        cache = self.teacher_cache
        summary: dict[str, Any] = {
            "active": cache is not None,
            "refresh_count": self.teacher_refresh_count,
        }
        if cache is not None:
            summary.update(
                {
                    "teacher_refresh_index": cache.refresh_index,
                    "teacher_source_epoch": cache.source_epoch + 1,
                    "shells": [list(shell) for shell in cache.shells],
                    "merged_boundaries": cache.merged_boundaries,
                    "c_teacher": cache.c_teacher,
                    "effective_rank": cache.diagnostics["effective_rank"],
                }
            )
            panel = self._fixed_panel_diagnostics(epoch_index)
            save_json(
                panel,
                self.diagnostics_dir / f"geometry_epoch{epoch_index + 1}.json",
            )
            summary["fixed_panel"] = panel
        record["gsr"] = summary

    @torch.no_grad()
    def _fixed_panel_diagnostics(self, epoch_index: int) -> dict[str, Any]:
        assert self.teacher_cache is not None
        embeddings, encoding = encode_teacher_corpus(
            self.model,
            self.diagnostic_loader,
            pooling=self.cfg.model.pooling,
            hidden_dim=self.cfg.model.hidden_dim,
            device=self.device,
            fp16=self.cfg.train.fp16,
            expected_sample_ids=self.diagnostic_ids,
        )
        return self._panel_diagnostics_from_embeddings(
            embeddings,
            epoch_index,
            stage="epoch_end",
            encoding=encoding,
        )

    @torch.no_grad()
    def _panel_diagnostics_from_embeddings(
        self,
        embeddings: torch.Tensor,
        epoch_index: int,
        *,
        stage: str,
        encoding: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.teacher_cache is not None
        student = full_normalize(embeddings, eps=self.cfg.gsr.eps).to(self.device)
        teacher = self.teacher_cache.lookup(self.diagnostic_ids, self.device)
        output = gsr_shell_loss(
            student,
            teacher,
            self.teacher_cache.shells,
            self.teacher_cache.c_teacher,
            eps=self.cfg.gsr.eps,
        )
        metrics = self._geometry_metrics(
            (output,),
            self.teacher_cache.c_teacher,
            pair_indices=self.diagnostic_pair_indices,
        )
        metrics.update(
            self._alignment_metrics(
                (output,), pair_indices=self.diagnostic_pair_indices
            )
        )
        student_cpu = student.detach().cpu()
        previous = self._previous_panel_student.get(stage)
        drift = None
        if previous is not None:
            representation_l2 = (student_cpu - previous).norm(dim=1)
            current_pairs = torch.pdist(student_cpu, p=2).square()
            previous_pairs = torch.pdist(previous, p=2).square()
            indices = self.diagnostic_pair_indices
            pair_delta = current_pairs.index_select(
                0, indices
            ) - previous_pairs.index_select(0, indices)
            drift = {
                "representation_l2_mean": float(representation_l2.mean()),
                "representation_l2_max": float(representation_l2.max()),
                "pair_distance_rmse": float(
                    pair_delta.square().mean().sqrt()
                    / math.sqrt(self.teacher_cache.c_teacher + self.cfg.gsr.eps)
                ),
            }
        self._previous_panel_student[stage] = student_cpu
        return {
            "epoch": epoch_index + 1,
            "stage": stage,
            "sample_count": embeddings.size(0),
            "teacher_refresh_index": self.teacher_cache.refresh_index,
            "total_loss": float(output.total_loss),
            "encoding": encoding,
            "student_drift_from_previous": drift,
            "metrics": metrics,
        }

    def _save_failure(
        self,
        reason: str,
        *,
        extra: dict[str, Any] | None = None,
        stem: str | None = None,
    ) -> Path:
        path = self.diagnostics_dir / (stem or f"failure_step{self.global_step}")
        path = path.with_suffix(".pt")
        payload: dict[str, Any] = {
            "reason": reason,
            "epoch": self.epoch_index + 1,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "torch_rng_state": torch.random.get_rng_state(),
            "context": self._failure_context,
            "teacher_diagnostics": self.teacher_cache.diagnostics
            if self.teacher_cache is not None
            else None,
        }
        if torch.cuda.is_available():
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        payload.update(extra or {})
        payload["optimizer"] = {
            "learning_rates": [
                float(group["lr"]) for group in self.optimizer.param_groups
            ],
            "weight_decay": [
                float(group.get("weight_decay", 0.0))
                for group in self.optimizer.param_groups
            ],
        }
        payload["amp_scale"] = float(self.scaler.get_scale())
        torch.save(self._cpu_payload(payload), path)
        LOGGER.error("Saved GSR failure dump to %s (%s)", path, reason)
        return path

    def method_report_metadata(self) -> Dict[str, Any]:
        cache = self.teacher_cache
        geometry: dict[str, Any] = {
            "geometry_dims": self.geometry_dims,
            "weight": self.cfg.gsr.weight,
            "warmup_epochs": self.cfg.gsr.warmup_epochs,
            "refresh_every_epochs": self.cfg.gsr.refresh_every_epochs,
            "teacher_refreshes": self.teacher_refresh_count,
            "diagnostics_dir": "diagnostics",
            "steps_path": "diagnostics/steps.jsonl",
        }
        if cache is not None:
            geometry.update(
                {
                    "final_teacher_refresh": cache.refresh_index,
                    "shells": [list(shell) for shell in cache.shells],
                    "merged_boundaries": cache.merged_boundaries,
                    "effective_rank": cache.diagnostics["effective_rank"],
                    "c_teacher": cache.c_teacher,
                    "teacher_drift": cache.diagnostics["drift_from_previous"],
                }
            )
        return {"geometry": geometry}

    @classmethod
    def _cpu_payload(cls, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cls._cpu_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._cpu_payload(item) for item in value)
        return value
