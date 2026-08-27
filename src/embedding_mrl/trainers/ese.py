"""ESE baseline: EPRESSO self-distillation across nested dimensions and layers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..losses.ese import epresso_simcse_from_hidden_states
from .base import BaseTrainer


class ESETrainer(BaseTrainer):
    method = "ese"

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

        ese_cfg = self.cfg.ese
        loss, loss_dict, acc_dict = epresso_simcse_from_hidden_states(
            out1.hidden_states,
            out2.hidden_states,
            batch["attention_mask1"],
            batch["attention_mask2"],
            matryoshka_dims=self.cfg.matryoshka.ascending,
            temperature=ese_cfg.temperature,
            n_layers_per_step=ese_cfg.n_layers_per_step,
            use_intermediate_layers=ese_cfg.use_intermediate_layers,
            use_layer_weight=ese_cfg.use_layer_weight,
        )

        # Report the smallest and largest nested dimension as a quality signal.
        dims = self.cfg.matryoshka.ascending
        logs = {
            f"acc@{dims[0]}": acc_dict.get(f"acc_dim_{dims[0]}", float("nan")),
            f"acc@{dims[-1]}": acc_dict.get(f"acc_dim_{dims[-1]}", float("nan")),
            f"loss@{dims[-1]}": loss_dict.get(f"loss_dim_{dims[-1]}", float("nan")),
        }
        return loss, logs
