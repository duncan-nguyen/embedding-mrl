"""MIPIC: Matryoshka InfoNCE plus SIA and PIC alignment (paper Eq 18)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from ..losses.infonce import matryoshka_info_nce
from ..losses.mipic import MIPICAlignmentLoss
from ..pooling import pool
from .base import BaseTrainer


class MIPICTrainer(BaseTrainer):
    method = "mipic"

    def setup_modules(self) -> None:
        cfg = self.cfg
        self.alignment_loss = MIPICAlignmentLoss(
            d_full=cfg.model.hidden_dim,
            matryoshka_dims=cfg.matryoshka.dims,
            layers=cfg.mipic.layers,
            checkpoints=cfg.mipic.checkpoint_pairs(),
            gamma_schedule=cfg.mipic.gamma_schedule,
            k_min=cfg.mipic.k_min,
            temperature=cfg.matryoshka.temperature,
            attention_temperature=cfg.mipic.attention_temperature,
            w_att=cfg.mipic.w_att,
            w_cka=cfg.mipic.w_cka,
            w_pic=cfg.mipic.w_pic,
            aggregate=cfg.mipic.aggregate,
            pic_hidden_dim=cfg.mipic.pic_hidden_dim,
            pic_detach_target=cfg.mipic.pic_detach_target,
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

        # SIA/PIC are computed per view and averaged, since both are valid
        # self-distillation targets for the same sentence.
        align1 = self.alignment_loss(list(out1.hidden_states), mask=batch["attention_mask1"])
        align2 = self.alignment_loss(list(out2.hidden_states), mask=batch["attention_mask2"])
        align = {key: (align1[key] + align2[key]) / 2.0 for key in align1}

        # L_MRL: SimCSE InfoNCE at every nested prefix (Eq 1, Eq 19).
        emb1 = pool(out1.hidden_states[-1], batch["attention_mask1"], self.cfg.model.pooling)
        emb2 = pool(out2.hidden_states[-1], batch["attention_mask2"], self.cfg.model.pooling)
        matry_loss, _ = matryoshka_info_nce(
            emb1,
            emb2,
            nested_dims=self.cfg.matryoshka.ascending,
            temperature=self.cfg.matryoshka.temperature,
        )

        # Eq 18: L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)
        alpha = self.cfg.mipic.alpha
        loss = alpha * matry_loss + (1.0 - alpha) * align["total_loss"]

        logs = {
            "mrl": matry_loss.item(),
            "sia": align["sia_loss"].item(),
            "att": align["att_loss"].item(),
            "cka": align["cka_loss"].item(),
            "pic": align["pic_loss"].item(),
        }
        return loss, logs
