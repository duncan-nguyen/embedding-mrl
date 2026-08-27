"""Loss functions for the four Matryoshka training methods."""

from .cka import CKALoss, MatryoshkaCKASelfDistiller, PerExampleCKALoss
from .ese import (
    epresso_simcse,
    epresso_simcse_from_hidden_states,
    epresso_simcse_with_layers,
)
from .infonce import info_nce, log_dimension_weights, matryoshka_info_nce
from .mipic import (
    AttentionDistributionMatching,
    MIPICAlignmentLoss,
    PipelineInfoNCELoss,
    TopKCKAAlignment,
)
from .sdr import (
    SemanticDistortionLoss,
    candidate_mask,
    gram_mse_distortion,
    hard_neighbor_cross_entropy,
    neighborhood_logits,
    rate_prior,
    semantic_neighborhood_distortion,
)

__all__ = [
    "CKALoss",
    "PerExampleCKALoss",
    "MatryoshkaCKASelfDistiller",
    "epresso_simcse",
    "epresso_simcse_from_hidden_states",
    "epresso_simcse_with_layers",
    "info_nce",
    "matryoshka_info_nce",
    "log_dimension_weights",
    "AttentionDistributionMatching",
    "TopKCKAAlignment",
    "PipelineInfoNCELoss",
    "MIPICAlignmentLoss",
    "SemanticDistortionLoss",
    "semantic_neighborhood_distortion",
    "neighborhood_logits",
    "candidate_mask",
    "gram_mse_distortion",
    "hard_neighbor_cross_entropy",
    "rate_prior",
]
