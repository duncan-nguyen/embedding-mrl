"""MIPIC alignment losses, following the paper's formulation.

Equation numbers refer to *MIPIC: Matryoshka Representation Learning via
Self-Distilled Intra-Relational and Progressive Information Chaining*
(``docs/MIPIC.pdf``).

Two mechanisms sit on top of the plain Matryoshka objective:

* **SIA** - Self-Distilled Intra-Relational Alignment (Sec 3.2). For each
  truncated prefix ``d_i`` the model must reproduce the full-width model's token
  importance ordering (attention KL, Eq 4-6) and the geometry of the top-``k_i``
  most important tokens (linear CKA, Eq 7-12).
* **PIC** - Progressive Information Chaining (Sec 3.3). Each ``(dim, layer)``
  checkpoint must predict the next, deeper and wider one (InfoNCE, Eq 15-17).

Aggregation follows Eq 13/14/17: unweighted **sums** over dimensions, layers and
chain steps. Set ``aggregate="mean"`` to average instead.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cka import PerExampleCKALoss


class AttentionDistributionMatching(nn.Module):
    """Eq 4-6: the truncated prefix must rank tokens the way the full width does.

    The full-dimensional ``[CLS]`` vector is the query for *both* distributions.
    A learnable ``P_i`` (Eq 5) lifts the truncated token back to ``D`` so the two
    score sets live on the same scale.
    """

    def __init__(self, d_small: int, d_full: int):
        """
        Args:
            d_small: truncated width ``d_i``.
            d_full: full hidden width ``D``, which acts as the teacher.
        """
        super().__init__()
        self.d_full = d_full
        # P_i in R^{d_i x D}; applied as P_i^T h_j^(i), i.e. a bias-free lift.
        self.up_project = nn.Linear(d_small, d_full, bias=False)

    @staticmethod
    def _scores_to_probs(
        scores: torch.Tensor, mask: Optional[torch.Tensor], temperature: float
    ) -> torch.Tensor:
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        return F.softmax(scores / temperature, dim=-1)

    def teacher_scores(self, hidden_full: torch.Tensor) -> torch.Tensor:
        """Eq 4: ``s_j^(D) = h_CLS . h_j / sqrt(D)``. Returns ``[B, L]``."""
        cls_vector = hidden_full[:, 0, :]  # [B, D]
        return torch.einsum("bd,bld->bl", cls_vector, hidden_full) / math.sqrt(self.d_full)

    def student_scores(
        self, hidden_small: torch.Tensor, hidden_full: torch.Tensor
    ) -> torch.Tensor:
        """Eq 5: ``s_j^(i) = h_CLS . P_i^T h_j^(i) / sqrt(D)``. Returns ``[B, L]``."""
        cls_vector = hidden_full[:, 0, :]  # the query stays full-dimensional
        lifted = self.up_project(hidden_small)  # [B, L, D]
        return torch.einsum("bd,bld->bl", cls_vector, lifted) / math.sqrt(self.d_full)

    def forward(
        self,
        hidden_small: torch.Tensor,
        hidden_full: torch.Tensor,
        teacher_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """Eq 6: ``KL(a_i || a_D)``, averaged over the batch.

        Args:
            hidden_small: ``[B, L, d_i]`` truncated hidden states.
            hidden_full: ``[B, L, D]`` full-width hidden states.
            teacher_probs: ``[B, L]`` ``a_D``, computed once per layer and shared
                across every dimension.
        """
        student_probs = self._scores_to_probs(
            self.student_scores(hidden_small, hidden_full), mask, temperature
        )

        eps = 1e-8
        # F.kl_div(input=log q, target=p) == sum p * (log p - log q) == KL(p || q),
        # so `input` is the teacher and `target` the student: KL(a_i || a_D).
        return F.kl_div(
            (teacher_probs + eps).log(),
            student_probs + eps,
            reduction="batchmean",
            log_target=False,
        )


class TopKCKAAlignment(nn.Module):
    """Eq 7-12: linear CKA between the top-``k_i`` token submatrices.

    Tokens are ranked once by the teacher distribution ``a_D``, so the selected
    sets are nested (``S_k1 subset S_k2 subset ...``) as Sec 3.2.2 requires.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.cka = PerExampleCKALoss(eps=eps)

    @staticmethod
    def select_top_k(
        hidden: torch.Tensor,
        ranking_scores: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather the ``k`` highest-ranked token vectors. Returns ``[B, k, d]``."""
        _, seq_len, width = hidden.shape

        scores = ranking_scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        _, indices = torch.topk(scores, k=min(k, seq_len), dim=1)
        return torch.gather(hidden, 1, indices.unsqueeze(-1).expand(-1, -1, width))

    def forward(
        self,
        hidden_small: torch.Tensor,
        hidden_full: torch.Tensor,
        ranking_scores: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eq 12: ``1 - CKA(h_i, H_i)`` over the shared top-``k`` token subset."""
        small_sub = self.select_top_k(hidden_small, ranking_scores, k, mask)
        full_sub = self.select_top_k(hidden_full, ranking_scores, k, mask)
        return self.cka(small_sub, full_sub)


class PipelineInfoNCELoss(nn.Module):
    """Eq 15-16: a checkpoint must predict the next, deeper and wider one.

    ``phi_i`` bridges the width mismatch; the target is detached so information
    flows from the deeper checkpoint down to the shallower one, not the reverse.
    """

    def __init__(
        self,
        d_src: int,
        d_tgt: int,
        d_hidden: Optional[int] = None,
        detach_target: bool = True,
    ):
        super().__init__()
        d_hidden = d_hidden or max(d_src, d_tgt) // 2
        self.detach_target = detach_target
        self.phi = nn.Sequential(
            nn.Linear(d_src, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_tgt),
        )

    def forward(
        self,
        src_hidden: torch.Tensor,
        tgt_hidden: torch.Tensor,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """
        Args:
            src_hidden: ``[B, L, d_i]`` at the shallower checkpoint.
            tgt_hidden: ``[B, L, d_{i+1}]`` at the deeper one.
        """
        z_src = src_hidden[:, 0, :]  # the [CLS] representation, per Sec 3.3
        z_tgt = tgt_hidden[:, 0, :]
        if self.detach_target:
            z_tgt = z_tgt.detach()

        projected = F.normalize(self.phi(z_src), dim=-1)
        target = F.normalize(z_tgt, dim=-1)

        logits = torch.matmul(projected, target.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class MIPICAlignmentLoss(nn.Module):
    """``L_SIA + L_PIC`` - the ``(1 - alpha)`` half of Eq 18.

    ``L_SIA = sum_k sum_i (L_att^(i) + L_CKA^(i))`` (Eq 13, 14) and
    ``L_PIC = sum_i L_chain^(i)`` (Eq 17).
    """

    def __init__(
        self,
        d_full: int,
        matryoshka_dims: Sequence[int],
        layers: Sequence[int],
        checkpoints: Sequence[Tuple[int, int]],
        gamma_schedule: Sequence[float],
        k_min: int = 8,
        temperature: float = 0.05,
        attention_temperature: Optional[float] = None,
        w_att: float = 1.0,
        w_cka: float = 1.0,
        w_pic: float = 1.0,
        aggregate: str = "sum",
        pic_hidden_dim: Optional[int] = None,
        pic_detach_target: bool = True,
    ):
        """
        Args:
            d_full: full hidden width ``D``; the internal teacher.
            matryoshka_dims: the nested widths; ``d_full`` itself is the teacher
                and is excluded from SIA.
            layers: ``L``, the hidden-state indices SIA is applied at.
            checkpoints: ``C``, ordered ``(dim, layer)`` pairs driving PIC.
            gamma_schedule: top-k ratios, one per truncated prefix in ascending
                order of dimension (Appendix A.5).
            k_min: floor on the number of selected tokens.
            temperature: the shared ``tau``.
            attention_temperature: override for the SIA softmax; defaults to ``tau``.
            aggregate: ``"sum"`` follows Eq 13/14/17; ``"mean"`` averages instead.
        """
        super().__init__()

        if aggregate not in ("sum", "mean"):
            raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate!r}")

        self.d_full = d_full
        # Ascending: gamma_schedule[i] belongs to the i-th smallest prefix.
        self.sia_dims: List[int] = sorted(d for d in matryoshka_dims if d < d_full)
        self.layers = list(layers)
        self.checkpoints = [tuple(c) for c in checkpoints]
        self.k_min = k_min
        self.temperature = temperature
        self.attention_temperature = (
            temperature if attention_temperature is None else attention_temperature
        )
        self.w_att = w_att
        self.w_cka = w_cka
        self.w_pic = w_pic
        self.aggregate = aggregate

        if len(gamma_schedule) != len(self.sia_dims):
            raise ValueError(
                f"gamma_schedule has {len(gamma_schedule)} entries but there are "
                f"{len(self.sia_dims)} truncated prefixes {self.sia_dims}"
            )
        self.gamma_schedule = {dim: float(g) for dim, g in zip(self.sia_dims, gamma_schedule)}

        if len(self.checkpoints) < 2:
            raise ValueError("PIC needs at least two checkpoints to form a chain")

        # One P_i per (layer, dim): Eq 5's projection is layer-specific.
        self.attn_modules = nn.ModuleDict(
            {
                self._attn_key(layer, dim): AttentionDistributionMatching(
                    d_small=dim, d_full=d_full
                )
                for layer in self.layers
                for dim in self.sia_dims
            }
        )
        self.cka_module = TopKCKAAlignment()

        self.chain_modules = nn.ModuleDict(
            {
                self._chain_key(i): PipelineInfoNCELoss(
                    d_src=self.checkpoints[i][0],
                    d_tgt=self.checkpoints[i + 1][0],
                    d_hidden=pic_hidden_dim,
                    detach_target=pic_detach_target,
                )
                for i in range(len(self.checkpoints) - 1)
            }
        )

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _attn_key(layer: int, dim: int) -> str:
        return f"layer{layer}_dim{dim}"

    @staticmethod
    def _chain_key(index: int) -> str:
        return f"chain{index}"

    def top_k_for(self, dim: int, seq_len: int, min_real_tokens: Optional[int] = None) -> int:
        """Appendix A.5: ``k_i = max(k_min, ceil(gamma_i * m))``.

        ``min_real_tokens`` clamps the result so padding is never selected when a
        batch mixes short and long sequences.
        """
        k = max(self.k_min, math.ceil(self.gamma_schedule[dim] * seq_len))
        k = min(k, seq_len)
        if min_real_tokens is not None:
            k = min(k, max(1, min_real_tokens))
        return k

    def _reduce(self, total: torch.Tensor, count: int) -> torch.Tensor:
        return total / count if (self.aggregate == "mean" and count) else total

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: the encoder's per-layer states, each ``[B, L, D]``.
            mask: ``[B, L]`` attention mask.
        Returns:
            ``{"total_loss", "sia_loss", "att_loss", "cka_loss", "pic_loss"}``
        """
        reference = hidden_states[-1]
        seq_len = reference.size(1)
        min_real = int(mask.sum(dim=1).min().item()) if mask is not None else None

        att_total = reference.new_zeros(())
        cka_total = reference.new_zeros(())
        pic_total = reference.new_zeros(())
        att_count = cka_count = 0

        # ---- SIA: Eq 13 over dims, Eq 14 over layers ---------------------- #
        for layer in self.layers:
            hidden_full = hidden_states[layer]

            # a_D is the same for every dimension at this layer, so compute once.
            any_module = self.attn_modules[self._attn_key(layer, self.sia_dims[0])]
            teacher_scores = any_module.teacher_scores(hidden_full)
            teacher_probs = any_module._scores_to_probs(
                teacher_scores, mask, self.attention_temperature
            )

            for dim in self.sia_dims:
                hidden_small = hidden_full[..., :dim]
                module = self.attn_modules[self._attn_key(layer, dim)]

                att_total = att_total + module(
                    hidden_small=hidden_small,
                    hidden_full=hidden_full,
                    teacher_probs=teacher_probs,
                    mask=mask,
                    temperature=self.attention_temperature,
                )
                att_count += 1

                cka_total = cka_total + self.cka_module(
                    hidden_small=hidden_small,
                    hidden_full=hidden_full,
                    # Ranking comes from the teacher, which makes the token sets nested.
                    ranking_scores=teacher_scores,
                    k=self.top_k_for(dim, seq_len, min_real),
                    mask=mask,
                )
                cka_count += 1

        att_total = self._reduce(att_total, att_count)
        cka_total = self._reduce(cka_total, cka_count)
        sia_total = self.w_att * att_total + self.w_cka * cka_total

        # ---- PIC: Eq 17 over consecutive checkpoints ---------------------- #
        for index in range(len(self.checkpoints) - 1):
            dim_src, layer_src = self.checkpoints[index]
            dim_tgt, layer_tgt = self.checkpoints[index + 1]
            pic_total = pic_total + self.chain_modules[self._chain_key(index)](
                src_hidden=hidden_states[layer_src][..., :dim_src],
                tgt_hidden=hidden_states[layer_tgt][..., :dim_tgt],
                temperature=self.temperature,
            )
        pic_total = self._reduce(pic_total, len(self.checkpoints) - 1)

        return {
            "total_loss": sia_total + self.w_pic * pic_total,
            "sia_loss": sia_total,
            "att_loss": att_total,
            "cka_loss": cka_total,
            "pic_loss": pic_total,
        }

    def validate_against(self, num_hidden_states: int) -> None:
        """Fail fast when configured layer indices exceed what the backbone emits."""
        referenced = set(self.layers) | {layer for _, layer in self.checkpoints}
        out_of_range = sorted(i for i in referenced if i >= num_hidden_states or i < 0)
        if out_of_range:
            raise ValueError(
                f"layers/checkpoints reference hidden-state indices {out_of_range}, "
                f"but the backbone only exposes {num_hidden_states} (0 = embeddings)"
            )
