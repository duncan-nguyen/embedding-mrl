"""SDR-MRL: Algorithm 1 - task loss plus rate-weighted semantic distortion.

    L_SDR = L_task + lambda_sem * sum_k pi_k D_k
                   + lambda_mono * sum_k [D_k - D_{k-1}]_+        (Eq 55)

``L_task`` is the ordinary Matryoshka InfoNCE (Eq 51-52). It is not decoration:
the semantic term alone is minimised by a collapsed teacher, since a degenerate
neighborhood distribution is trivially easy for every prefix to reproduce
(Sec 4.9).
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Optional, Tuple

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
        ).to(self.device)
        # No parameters of its own, so it stays out of ``extra_modules``: the
        # prefix cosine geometry *is* the semantic decoder (Sec 4.3).

        self.teacher_model: Optional[torch.nn.Module] = self._build_teacher()
        self._global_step = 0
        self.diagnostics: list[Dict[str, Any]] = []
        self._log_setup()

    def _log_setup(self) -> None:
        """State the objective the run is actually optimising, once, up front."""
        cfg = self.cfg.sdr
        LOGGER.info(
            "SDR-MRL | teacher=%s | geometry=%s | divergence=%s | candidates=%s",
            cfg.teacher,
            cfg.geometry,
            cfg.divergence,
            cfg.candidates,
        )
        LOGGER.info(
            "  L = %.3g*L_task + %.3g*sum_k pi_k D_k%s",
            cfg.w_task,
            cfg.lambda_sem,
            f" + {cfg.lambda_mono:.3g}*sum_k [D_k - D_(k-1)]_+"
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
        # tau_S != tau_T leaves D_K > 0: the objective then has a floor no
        # amount of training can reach, which is nearly always a config slip.
        if cfg.teacher_temperature != cfg.student_temperature:
            LOGGER.warning(
                "  tau_T=%.3g != tau_S=%.3g -> D at the full width is bounded away "
                "from 0; the semantic term cannot reach its optimum.",
                cfg.teacher_temperature,
                cfg.student_temperature,
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

        sem_loss = task_loss.new_zeros(())
        mono_loss = task_loss.new_zeros(())
        distortions: Dict[int, float] = {}

        for view, student in enumerate(embeddings, start=1):
            teacher = self._teacher_for(batch, view, embeddings)
            outcome = self.sdr_loss(student, teacher, rate_index=rate_index)
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
        self._maybe_log_diagnostics(batch, embeddings)

        logs = {"task": task_loss.item(), "sem": sem_loss.detach().item()}
        if cfg.sdr.lambda_mono > 0:
            logs["mono"] = mono_loss.detach().item()
        smallest = min(distortions)
        logs[f"D@{smallest}"] = distortions[smallest]
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
    def _maybe_log_diagnostics(self, batch, embeddings) -> None:
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
            student_temperature=self.cfg.sdr.student_temperature,
            divergence=self.cfg.sdr.divergence,
            candidates=self.cfg.sdr.candidates,
            top_m=self.cfg.sdr.top_m,
            knn_k=self.cfg.sdr.diagnostics_knn_k,
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
