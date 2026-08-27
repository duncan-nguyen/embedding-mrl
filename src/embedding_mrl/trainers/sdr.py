"""SDR-MRL: Algorithm 1 - task loss plus rate-weighted semantic distortion.

    L_SDR = L_task + lambda_sem * sum_k pi_k D_k
                   + lambda_mono * sum_k [D_k - sg(D_{k-1})]_+    (Eq 55)

``L_task`` is the ordinary Matryoshka InfoNCE (Eq 51-52). It is not decoration:
the semantic term alone is minimised by a collapsed teacher, since a degenerate
neighborhood distribution is trivially easy for every prefix to reproduce
(Sec 4.9). It does *not*, however, rule out the subtler minimiser in which the
tail coordinates simply stop carrying anything the prefix lacks - with an
online self-teacher, ``D_k = 0`` is reached just as well by the full width
degrading to the prefix as by the prefix improving. ``norm_share`` in the
diagnostics watches for that; an EMA or frozen teacher (Sec 4.15) removes it.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..diagnostics import format_diagnostics, neighborhood_diagnostics
from ..losses.infonce import matryoshka_info_nce
from ..losses.sdr import SemanticDistortionLoss
from ..pooling import pool
from .base import BaseTrainer

LOGGER = logging.getLogger("embedding_mrl.train")


class SDRTrainer(BaseTrainer):
    method = "sdr"

    # -- setup -------------------------------------------------------------- #
    def setup_modules(self) -> None:
        cfg = self.cfg
        self.sdr_loss = SemanticDistortionLoss(
            dims=cfg.matryoshka.dims,
            full_dim=cfg.model.hidden_dim,
            teacher_temperature=cfg.sdr.teacher_temperature,
            student_temperature=cfg.sdr.student_temperature,
            divergence=cfg.sdr.divergence,
            geometry=cfg.sdr.geometry,
            candidates=cfg.sdr.candidates,
            top_m=cfg.sdr.top_m,
            rate_prior_kind=cfg.sdr.rate_prior,
            rate_weights=cfg.sdr.rate_weights,
            lambda_mono=cfg.sdr.lambda_mono,
            stochastic_rate=cfg.sdr.stochastic_rate,
            learnable_temperature=cfg.sdr.learnable_temperature,
            temperature_bounds=(cfg.sdr.temperature_min, cfg.sdr.temperature_max),
        ).to(self.device)
        if cfg.sdr.learnable_temperature:
            # The decoder temperatures tau_k (Eq 24) are the module's only
            # parameters. They belong to the training-time decoder, not to the
            # deployed representation.
            self.extra_modules.append(self.sdr_loss)

        self.teacher_model: Optional[torch.nn.Module] = self._build_teacher()
        self._global_step = 0
        self.diagnostics: list[Dict[str, Any]] = []

        # Memory queue (Sec 4.2): past full-width embeddings widen C_i beyond
        # the 15 random sentences a batch of 16 offers.
        self.queue_size = int(cfg.sdr.queue_size)
        self._queue_student: Optional[torch.Tensor] = None
        self._queue_teacher: Optional[torch.Tensor] = None
        self._queue_ptr = 0
        self._queue_count = 0

        self._log_setup()

    def extra_param_groups(self) -> List[Dict[str, Any]]:
        if not self.cfg.sdr.learnable_temperature:
            return []
        return [
            {
                "params": list(self.sdr_loss.parameters()),
                "lr": self.cfg.sdr.temperature_lr,
                "weight_decay": 0.0,  # a decayed log tau would drift toward tau = 1
            }
        ]

    def _log_setup(self) -> None:
        """State the objective the run is actually optimising, once, up front."""
        cfg = self.cfg.sdr
        LOGGER.info(
            "SDR-MRL | teacher=%s | geometry=%s | divergence=%s | candidates=%s | queue=%d",
            cfg.teacher,
            cfg.geometry,
            cfg.divergence,
            cfg.candidates,
            self.queue_size,
        )
        LOGGER.info(
            "  L = %.3g*L_task + %.3g*sum_k pi_k D_k%s",
            cfg.w_task,
            cfg.lambda_sem,
            f" + {cfg.lambda_mono:.3g}*sum_k [D_k - sg(D_(k-1))]_+"
            if cfg.lambda_mono > 0
            else "  (lambda_mono = 0, Eq 56)",
        )
        LOGGER.info(
            "  prefixes %s  pi = %s  %s",
            self.sdr_loss.prefix_dims,
            [round(w, 4) for w in self.sdr_loss.rate_prior],
            "one rate sampled per step (Eq 66-68)"
            if cfg.stochastic_rate
            else "all prefixes per step",
        )
        LOGGER.info(
            "  tau_T = %.3g, tau_k init = %.3g (%s, Eq 24), tau_K = tau_T so D_K = 0",
            cfg.teacher_temperature,
            cfg.student_temperature,
            f"learnable in [{cfg.temperature_min:g}, {cfg.temperature_max:g}]"
            if cfg.learnable_temperature
            else "fixed",
        )
        if cfg.teacher == "online":
            LOGGER.info(
                "  online self-teacher: p_T moves with the student, so D_k is a "
                "self-consistency term, not a bound on a fixed semantic variable "
                "(Sec 4.9); watch norm_share in the diagnostics."
            )

    def _build_teacher(self) -> Optional[torch.nn.Module]:
        """A3: ``online`` needs nothing; ``ema`` and ``frozen`` need a second model."""
        kind = self.cfg.sdr.teacher
        if kind == "online":
            return None

        if kind == "ema":
            teacher = copy.deepcopy(self.model)
        else:
            from transformers import AutoModel

            teacher = AutoModel.from_pretrained(
                self.cfg.sdr.teacher_model,
                trust_remote_code=self.cfg.model.trust_remote_code,
            ).to(self.device)

        teacher.requires_grad_(False)
        teacher.eval()  # a deterministic teacher: no dropout noise in p_T
        return teacher

    # -- teacher maintenance ------------------------------------------------ #
    def on_optimizer_step(self) -> None:
        """EMA teacher: ``theta_T <- m theta_T + (1 - m) theta``."""
        if self.cfg.sdr.teacher != "ema" or self.teacher_model is None:
            return

        momentum = self.cfg.sdr.teacher_momentum
        with torch.no_grad():
            for target, source in zip(
                self.teacher_model.parameters(), self.model.parameters()
            ):
                target.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)
            # Buffers (LayerNorm stats and friends) are copied, not averaged.
            for target, source in zip(
                self.teacher_model.buffers(), self.model.buffers()
            ):
                target.copy_(source)

    @torch.no_grad()
    def _teacher_embedding(
        self, batch: Dict[str, torch.Tensor], view: int
    ) -> torch.Tensor:
        """Eq 21-22's ``z_bar^(K)``, from the separate teacher encoder."""
        outputs = self.teacher_model(
            input_ids=batch[f"input_ids{view}"],
            attention_mask=batch[f"attention_mask{view}"],
            output_hidden_states=False,
            return_dict=True,
        )
        return pool(
            outputs.last_hidden_state,
            batch[f"attention_mask{view}"],
            self.cfg.model.pooling,
        )

    # -- memory queue (Sec 4.2) --------------------------------------------- #
    def queue_extras(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``(student_extra, teacher_extra)`` - the filled part of the queue, or ``(None, None)``."""
        if self.queue_size <= 0 or self._queue_count == 0:
            return None, None
        n = self._queue_count
        return self._queue_student[:n], self._queue_teacher[:n]

    @torch.no_grad()
    def _enqueue(self, student: torch.Tensor, teacher: torch.Tensor) -> None:
        """FIFO insert of one batch's full-width student and teacher embeddings.

        Student rows are stored detached: gradient only ever flows through the
        anchor (Eq 61), so a queue entry being a few steps stale is the usual
        MoCo-style approximation, not a change of objective.
        """
        if self.queue_size <= 0:
            return
        student = student.detach().float()
        teacher = teacher.detach().float()
        if self._queue_student is None:
            self._queue_student = student.new_zeros(self.queue_size, student.size(1))
            self._queue_teacher = teacher.new_zeros(self.queue_size, teacher.size(1))

        batch = student.size(0)
        for offset in range(0, batch, self.queue_size):
            rows_s = student[offset : offset + self.queue_size]
            rows_t = teacher[offset : offset + self.queue_size]
            n = rows_s.size(0)
            end = self._queue_ptr + n
            if end <= self.queue_size:
                self._queue_student[self._queue_ptr : end] = rows_s
                self._queue_teacher[self._queue_ptr : end] = rows_t
            else:  # wrap around
                head = self.queue_size - self._queue_ptr
                self._queue_student[self._queue_ptr :] = rows_s[:head]
                self._queue_teacher[self._queue_ptr :] = rows_t[:head]
                self._queue_student[: n - head] = rows_s[head:]
                self._queue_teacher[: n - head] = rows_t[head:]
            self._queue_ptr = end % self.queue_size
            self._queue_count = min(self.queue_size, self._queue_count + n)

    # -- Algorithm 1 -------------------------------------------------------- #
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        embeddings = [
            self._encode(batch, view) for view in (1, 2)
        ]  # H <- f_theta(B), line 1

        # L_task: Eq 51-52, the SimCSE positive is the other dropout view.
        task_loss, _ = matryoshka_info_nce(
            embeddings[0],
            embeddings[1],
            nested_dims=cfg.matryoshka.ascending,
            temperature=cfg.matryoshka.temperature,
            weights=cfg.sdr.task_weights,
        )

        # Line 4: one k ~ pi per step, shared by both views so the estimator
        # stays an unbiased sample of Eq 48 rather than an average of two.
        rate_index = (
            self.sdr_loss.sample_rate_index() if cfg.sdr.stochastic_rate else None
        )
        student_extra, teacher_extra = self.queue_extras()

        sem_loss = task_loss.new_zeros(())
        mono_loss = task_loss.new_zeros(())
        distortions: Dict[int, float] = {}
        teachers = []

        for view, student in enumerate(embeddings, start=1):
            teacher = self._teacher_for(batch, view, embeddings)
            teachers.append(teacher)
            outcome = self.sdr_loss(
                student,
                teacher,
                rate_index=rate_index,
                student_extra=student_extra,
                teacher_extra=teacher_extra,
            )
            sem_loss = sem_loss + outcome["sem_loss"] / len(embeddings)
            mono_loss = mono_loss + outcome["mono_loss"] / len(embeddings)
            for dim, value in outcome["distortions"].items():
                distortions[dim] = distortions.get(dim, 0.0) + value.detach().item() / len(
                    embeddings
                )

        # Line 16.
        loss = (
            cfg.sdr.w_task * task_loss
            + cfg.sdr.lambda_sem * sem_loss
            + cfg.sdr.lambda_mono * mono_loss
        )

        self._global_step += 1
        self._maybe_log_diagnostics(batch, embeddings, student_extra, teacher_extra)

        # Only after the diagnostics: this step's rows must not be their own
        # queue candidates.
        for student, teacher in zip(embeddings, teachers):
            self._enqueue(student, teacher)

        logs = {"task": task_loss.item(), "sem": sem_loss.detach().item()}
        if cfg.sdr.lambda_mono > 0:
            logs["mono"] = mono_loss.detach().item()
        smallest = min(distortions)
        logs[f"D@{smallest}"] = distortions[smallest]
        if cfg.sdr.learnable_temperature:
            logs[f"tau@{smallest}"] = self.sdr_loss.student_temperatures[smallest]
        return loss, logs

    def _encode(self, batch: Dict[str, torch.Tensor], view: int) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch[f"input_ids{view}"],
            attention_mask=batch[f"attention_mask{view}"],
            output_hidden_states=False,
            return_dict=True,
        )
        return pool(
            outputs.last_hidden_state,
            batch[f"attention_mask{view}"],
            self.cfg.model.pooling,
        )

    def _teacher_for(
        self, batch: Dict[str, torch.Tensor], view: int, embeddings
    ) -> torch.Tensor:
        """Which full-dimensional representation defines ``p_T`` for this view."""
        if self.teacher_model is not None:
            return self._teacher_embedding(batch, view)
        if self.cfg.sdr.cross_view:
            # The other SimCSE view: a noisier but augmentation-invariant teacher.
            return embeddings[view % len(embeddings)].detach()
        # Eq 21: stop-gradient on the student's own full-dimensional embedding.
        return embeddings[view - 1].detach()

    # -- mathematical diagnostics ------------------------------------------- #
    def _maybe_log_diagnostics(
        self, batch, embeddings, student_extra=None, teacher_extra=None
    ) -> None:
        """Recompute the derivation's own quantities and append them to a log.

        The training loss collapses everything into one number; these are the
        terms the propositions are stated in, so a run can be checked against
        the maths (see :mod:`embedding_mrl.diagnostics`).
        """
        every = self.cfg.sdr.diagnostics_every
        if not every or self._global_step % every != 0:
            return

        record = neighborhood_diagnostics(
            embeddings[0].detach(),
            self._teacher_for(batch, 1, embeddings),
            dims=self.cfg.matryoshka.ascending,
            teacher_temperature=self.cfg.sdr.teacher_temperature,
            student_temperature=self.sdr_loss.student_temperatures,
            divergence=self.cfg.sdr.divergence,
            candidates=self.cfg.sdr.candidates,
            top_m=self.cfg.sdr.top_m,
            knn_k=self.cfg.sdr.diagnostics_knn_k,
            student_extra=student_extra,
            teacher_extra=teacher_extra,
        )
        if not record:
            return

        record["step"] = self._global_step
        self.diagnostics.append(record)
        with (self.output_dir / "diagnostics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

        LOGGER.info(
            "step %d - semantic diagnostics\n%s",
            self._global_step,
            format_diagnostics(record, self.cfg.sdr.diagnostics_knn_k),
        )

    # -- checkpointing ------------------------------------------------------ #
    def save(self) -> None:
        super().save()
        if self.cfg.sdr.teacher == "ema" and self.teacher_model is not None:
            self.teacher_model.save_pretrained(self.output_dir / "teacher")
            LOGGER.info("Saved EMA teacher to %s", self.output_dir / "teacher")
        temperatures = {str(d): t for d, t in self.sdr_loss.student_temperatures.items()}
        (self.output_dir / "decoder_temperatures.json").write_text(
            json.dumps(temperatures, indent=2), encoding="utf-8"
        )
