"""Loss functions for the Matryoshka training methods."""

from .cka import CKALoss, MatryoshkaCKASelfDistiller, PerExampleCKALoss
from .ese import (
    epresso_simcse,
    epresso_simcse_from_hidden_states,
    epresso_simcse_with_layers,
)
from .infonce import info_nce, log_dimension_weights, matryoshka_info_nce
from .gsr import (
    GSRShellLossOutput,
    build_shell_slices,
    condensed_squared_distances,
    full_normalize,
    gsr_shell_loss,
    merge_tied_shells,
)
from .mipic import (
    AttentionDistributionMatching,
    MIPICAlignmentLoss,
    PipelineInfoNCELoss,
    TopKCKAAlignment,
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
    "GSRShellLossOutput",
    "full_normalize",
    "condensed_squared_distances",
    "build_shell_slices",
    "merge_tied_shells",
    "gsr_shell_loss",
]
