"""MIPIC alignment losses.

Three components sit on top of the plain Matryoshka InfoNCE objective:

* :class:`HorizontalAttentionAlignment` (SIA) - token-importance ordering should
  agree between a truncated prefix and the full-width representation.
* :class:`SubmatrixCKALoss` - the geometry of the top-k important tokens should
  agree, measured with per-example CKA.
* :class:`PipelineInfoNCELoss` (PIC) - a shallow/narrow stage should be able to
  predict the next deeper/wider stage.

:class:`TotalAlignmentLoss` wires them together.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .cka import PerExampleCKALoss


class HorizontalAttentionAlignment(nn.Module):
    """KL between the CLS-attention distributions of a truncated and a full prefix.

    Both views are projected into a shared ``d_att`` space first, since they have
    different widths.
    """

    def __init__(
        self, d_small: int, d_full: int, d_att: int = 64, enabled: bool = True
    ):
        """
        Args:
            d_small: truncated dimension (16, 32, ...).
            d_full: full hidden width, acting as the internal teacher.
            d_att: shared attention space width.
            enabled: when ``False`` the KL term is skipped and 0.0 is returned.
                The released notebooks hard-coded this off - see
                ``MIPICConfig.use_attention_kl``.
        """
        super().__init__()
        self.d_att = d_att
        self.enabled = enabled

        self.proj_small = nn.Linear(d_small, d_att)
        self.proj_full = nn.Linear(d_full, d_att)
        self.W_Q = nn.Linear(d_att, d_att)
        self.W_K = nn.Linear(d_att, d_att)

    def compute_attention_dist(
        self,
        hidden: torch.Tensor,
        proj: nn.Linear,
        mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention over tokens using the CLS token as the single query.

        Args:
            hidden: ``[B, L, d]``
            mask: ``[B, L]`` attention mask (padded positions are masked out).
        Returns:
            ``(probs, scores)``, both ``[B, L]``.
        """
        projected = proj(hidden)  # [B, L, d_att]

        q_cls = self.W_Q(projected[:, 0, :])  # [B, d_att]
        k_all = self.W_K(projected)  # [B, L, d_att]

        scores = torch.matmul(q_cls.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(self.d_att)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        return F.softmax(scores / temperature, dim=-1), scores

    def forward(
        self,
        h_small: torch.Tensor,
        h_full: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_small: ``[B, L, d_small]`` truncated hidden states.
            h_full: ``[B, L, d_full]`` full-width hidden states (the teacher).
        Returns:
            ``(kl_loss, small_scores)`` - ``small_scores`` drives the per-dimension
            top-k token selection used by :class:`SubmatrixCKALoss`.
        """
        small_probs, small_scores = self.compute_attention_dist(
            h_small, self.proj_small, mask, temperature
        )

        if not self.enabled:
            return h_small.new_zeros(()), small_scores

        full_probs, _ = self.compute_attention_dist(
            h_full, self.proj_full, mask, temperature
        )

        eps = 1e-8
        kl_loss = F.kl_div(
            (small_probs + eps).log(),
            full_probs + eps,
            reduction="batchmean",
            log_target=False,
        )
        return kl_loss, small_scores


class SubmatrixCKALoss(nn.Module):
    """Per-example CKA between the top-k token submatrices of two widths."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.cka_loss = PerExampleCKALoss(eps=eps)

    @staticmethod
    def select_top_k_tokens(
        hidden: torch.Tensor,
        selection_scores: torch.Tensor,
        k: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather the ``k`` highest-scoring token vectors.

        Args:
            hidden: ``[B, L, d]``
            selection_scores: ``[B, L]`` scores from *this* dimension's attention head.
        Returns:
            ``[B, min(k, L), d]``
        """
        _, seq_len, width = hidden.shape

        scores = selection_scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        _, top_k_indices = torch.topk(scores, k=min(k, seq_len), dim=1)
        gather_index = top_k_indices.unsqueeze(-1).expand(-1, -1, width)
        return torch.gather(hidden, 1, gather_index)

    def forward(
        self,
        h_small: torch.Tensor,
        h_full: torch.Tensor,
        small_scores: torch.Tensor,
        k: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``1 - CKA`` over the top-k tokens chosen by the small view's own scores."""
        small_sub = self.select_top_k_tokens(h_small, small_scores, k, mask)
        full_sub = self.select_top_k_tokens(h_full, small_scores, k, mask)
        return self.cka_loss(small_sub, full_sub)


class PipelineInfoNCELoss(nn.Module):
    """Vertical chaining: predict a deeper/wider stage from a shallower/narrower one."""

    def __init__(self, d_src: int, d_tgt: int, d_hidden: int = 256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_src, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_tgt),
        )

    def forward(
        self,
        src_hidden: torch.Tensor,
        tgt_hidden: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Args:
            src_hidden: ``[B, L, d_src]`` shallow stage.
            tgt_hidden: ``[B, L, d_tgt]`` deeper stage (gradient is stopped on it).
        """
        u_src = src_hidden[:, 0, :]
        v_tgt = tgt_hidden[:, 0, :].detach()

        u_proj = F.normalize(self.phi(u_src), dim=-1)
        v_tgt = F.normalize(v_tgt, dim=-1)

        logits = torch.matmul(u_proj, v_tgt.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class TotalAlignmentLoss(nn.Module):
    """``L_align = alpha * L_att + beta * L_CKA + gamma * L_chain``.

    Purely self-distilled: the full hidden width acts as its own teacher, so no
    second model is required.
    """

    def __init__(
        self,
        d_full: int,
        matryoshka_dims: Sequence[int],
        align_layers: Sequence[int],
        pipeline_pairs: Sequence[tuple[int, int, int, int]],
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        k_map: dict[int, int] | Callable[[int], int] | None = None,
        base_k: int = 64,
        attention_temperature: float = 1.0,
        pipeline_temperature: float = 0.07,
        d_att: int = 64,
        use_attention_kl: bool = False,
    ):
        """
        Args:
            d_full: full hidden width of the backbone (768 / 1024 here).
            matryoshka_dims: nested widths; the full width is the teacher and is skipped.
            align_layers: indices into ``hidden_states`` to align (0 = embeddings).
            pipeline_pairs: consecutive ``(layer_i, dim_i, layer_j, dim_j)`` transitions,
                as produced by :meth:`MIPICConfig.parsed_pipeline_pairs`.
            k_map: ``dim -> k`` for top-k selection; ``None`` uses
                ``max(8, int(dim / d_full * base_k))``.
            use_attention_kl: enable the SIA KL term (off in the released notebooks).
        """
        super().__init__()

        self.d_full = d_full
        self.matryoshka_dims = sorted(
            [d for d in matryoshka_dims if d <= d_full], reverse=True
        )
        self.align_layers = list(align_layers)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.attention_temperature = attention_temperature
        self.pipeline_temperature = pipeline_temperature
        self.use_attention_kl = use_attention_kl

        self.k_map = self._build_k_map(k_map, base_k, d_full)

        self.pipeline_pairs = [tuple(p) for p in pipeline_pairs]
        if not self.pipeline_pairs:
            raise ValueError("pipeline_pairs must not be empty")

        self.attn_modules = nn.ModuleDict()
        for layer_idx in self.align_layers:
            for dim in self.matryoshka_dims:
                if dim >= d_full:  # the full width is the teacher, not a student
                    continue
                self.attn_modules[self._attn_key(layer_idx, dim)] = (
                    HorizontalAttentionAlignment(
                        d_small=dim,
                        d_full=d_full,
                        d_att=d_att,
                        enabled=use_attention_kl,
                    )
                )

        self.cka_module = SubmatrixCKALoss()

        self.infonce_modules = nn.ModuleDict()
        for layer_i, dim_i, layer_j, dim_j in self.pipeline_pairs:
            self.infonce_modules[self._pipe_key(layer_i, dim_i, layer_j, dim_j)] = (
                PipelineInfoNCELoss(
                    d_src=dim_i, d_tgt=dim_j, d_hidden=max(dim_i, dim_j) // 2
                )
            )

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _build_k_map(k_map, base_k: int, d_full: int) -> Callable[[int], int]:
        default = lambda d: max(8, int((d / d_full) * base_k))
        if k_map is None:
            return default
        if callable(k_map):
            return k_map
        if isinstance(k_map, dict):
            table = {int(k): int(v) for k, v in k_map.items()}
            return lambda d: table.get(d, default(d))
        raise TypeError("k_map must be None, a dict or a callable")

    @staticmethod
    def _attn_key(layer_idx: int, dim: int) -> str:
        return f"layer_{layer_idx}_dim_{dim}"

    @staticmethod
    def _pipe_key(layer_i: int, dim_i: int, layer_j: int, dim_j: int) -> str:
        return f"pipe_{layer_i}_{dim_i}_to_{layer_j}_{dim_j}"

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: the encoder's per-layer states, each ``[B, L, d_full]``.
            mask: ``[B, L]`` attention mask.
        Returns:
            ``{"total_loss", "att_loss", "cka_loss", "chain_loss"}``
        """
        reference = hidden_states[-1]
        att_total = reference.new_zeros(())
        cka_total = reference.new_zeros(())
        chain_total = reference.new_zeros(())
        att_count = cka_count = 0

        # ---- horizontal: every prefix vs the full width, layer by layer ---- #
        for layer_idx in self.align_layers:
            full_layer = hidden_states[layer_idx]

            for dim in self.matryoshka_dims:
                if dim >= self.d_full:
                    continue

                small_layer = full_layer[..., :dim]
                attn_module = self.attn_modules[self._attn_key(layer_idx, dim)]

                att_loss, small_scores = attn_module(
                    h_small=small_layer,
                    h_full=full_layer,
                    mask=mask,
                    temperature=self.attention_temperature,
                )
                att_total = att_total + att_loss
                att_count += 1

                cka_total = cka_total + self.cka_module(
                    h_small=small_layer,
                    h_full=full_layer,
                    small_scores=small_scores,
                    k=self.k_map(dim),
                    mask=mask,
                )
                cka_count += 1

        if att_count:
            att_total = att_total / att_count
        if cka_count:
            cka_total = cka_total / cka_count

        # ---- vertical: chain shallow/narrow -> deep/wide ------------------- #
        for layer_i, dim_i, layer_j, dim_j in self.pipeline_pairs:
            module = self.infonce_modules[
                self._pipe_key(layer_i, dim_i, layer_j, dim_j)
            ]
            chain_total = chain_total + module(
                src_hidden=hidden_states[layer_i][..., :dim_i],
                tgt_hidden=hidden_states[layer_j][..., :dim_j],
                temperature=self.pipeline_temperature,
            )
        chain_total = chain_total / len(self.pipeline_pairs)

        total = (
            self.alpha * att_total + self.beta * cka_total + self.gamma * chain_total
        )
        return {
            "total_loss": total,
            "att_loss": att_total,
            "cka_loss": cka_total,
            "chain_loss": chain_total,
        }

    def validate_against(self, num_hidden_states: int) -> None:
        """Fail fast when configured layer indices exceed what the backbone emits."""
        referenced = set(self.align_layers)
        for layer_i, _, layer_j, _ in self.pipeline_pairs:
            referenced.update((layer_i, layer_j))
        out_of_range = sorted(
            i for i in referenced if i >= num_hidden_states or i < -num_hidden_states
        )
        if out_of_range:
            raise ValueError(
                f"align_layers/pipeline_pairs reference hidden-state indices {out_of_range}, "
                f"but the backbone only exposes {num_hidden_states} (0 = embeddings)"
            )
