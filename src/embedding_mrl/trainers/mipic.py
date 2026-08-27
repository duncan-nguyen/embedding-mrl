"""MIPIC: Matryoshka InfoNCE plus horizontal (SIA) and vertical (PIC) alignment."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from ..losses.infonce import matryoshka_info_nce
from ..losses.mipic import TotalAlignmentLoss
from ..pooling import pool
from .base import BaseTrainer


class MIPICTrainer(BaseTrainer):
    method = "mipic"

    def setup_modules(self) -> None:
        cfg = self.cfg
        self.alignment_loss = TotalAlignmentLoss(
            d_full=cfg.model.hidden_dim,
            matryoshka_dims=cfg.matryoshka.descending,
            align_layers=cfg.mipic.align_layers,
            pipeline_pairs=cfg.mipic.parsed_pipeline_pairs(),
            alpha=cfg.mipic.alpha,
            beta=cfg.mipic.beta,
            gamma=cfg.mipic.gamma,
            k_map=cfg.mipic.k_map,
            base_k=cfg.mipic.base_k,
            attention_temperature=cfg.mipic.attention_temperature,
            pipeline_temperature=cfg.mipic.pipeline_temperature,
            d_att=cfg.mipic.d_att,
            use_attention_kl=cfg.mipic.use_attention_kl,
        ).to(self.device)

        num_hidden_states = self.model.config.num_hidden_layers + 1
        self.alignment_loss.validate_against(num_hidden_states)
        self.extra_modules.append(self.alignment_loss)

    def extra_param_groups(self) -> List[Dict[str, Any]]:
        return [
            {
                "params": list(self.alignment_loss.parameters()),
                "lr": self.cfg.train.lr * self.cfg.mipic.module_lr_scale,
            }
        ]

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

        align1 = self.alignment_loss(
            list(out1.hidden_states), mask=batch["attention_mask1"]
        )
        align2 = self.alignment_loss(
            list(out2.hidden_states), mask=batch["attention_mask2"]
        )
        align = {key: (align1[key] + align2[key]) / 2.0 for key in align1}

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

        loss = (
            self.cfg.mipic.w_align * align["total_loss"]
            + self.cfg.mipic.w_matryoshka * matry_loss
        )
        logs = {
            "align": align["total_loss"].item(),
            "att": align["att_loss"].item(),
            "cka": align["cka_loss"].item(),
            "chain": align["chain_loss"].item(),
            "matry": matry_loss.item(),
        }
        return loss, logs
