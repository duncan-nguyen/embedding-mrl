"""MRL baseline: unsupervised SimCSE with InfoNCE applied at every nested dimension."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..losses.infonce import info_nce, matryoshka_info_nce
from ..pooling import pool
from .base import BaseTrainer


class MRLTrainer(BaseTrainer):
    method = "mrl"

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out1 = self.model(
            input_ids=batch["input_ids1"],
            attention_mask=batch["attention_mask1"],
            output_hidden_states=True,
            return_dict=True,
        )
        out2 = self.model(
            input_ids=batch["input_ids2"],
            attention_mask=batch["attention_mask2"],
            output_hidden_states=True,
            return_dict=True,
        )

        emb1 = pool(
            out1.hidden_states[-1], batch["attention_mask1"], self.cfg.model.pooling
        )
        emb2 = pool(
            out2.hidden_states[-1], batch["attention_mask2"], self.cfg.model.pooling
        )

        matry_loss, _ = matryoshka_info_nce(
            emb1,
            emb2,
            nested_dims=self.cfg.matryoshka.ascending,
            temperature=self.cfg.matryoshka.temperature,
        )
        task_loss, _ = info_nce(emb1, emb2, temperature=self.cfg.matryoshka.temperature)

        loss = self.cfg.mrl.w_matryoshka * matry_loss + self.cfg.mrl.w_task * task_loss
        return loss, {"matry": matry_loss.item(), "task": task_loss.item()}
