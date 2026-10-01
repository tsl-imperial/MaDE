# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Training pipeline."""

from made.training.dispatch import build_trainable
from made.training.losses import (
    forward_consistency_loss,
    inequality_violation_loss,
    inverse_consistency_loss,
    inverse_residual_norm,
    minimum_norm_loss,
    phase1_loss,
    phase1_i_loss,
    phase1_t_loss,
    phase2_loss,
    phase2_i_loss,
    phase2_t_loss,
    targeted_phase_loss,
)
from made.training.sampling import sample_controls
from made.training.trainer import create_train_state, train

__all__ = [
    "build_trainable",
    "create_train_state",
    "forward_consistency_loss",
    "inequality_violation_loss",
    "inverse_consistency_loss",
    "inverse_residual_norm",
    "minimum_norm_loss",
    "phase1_loss",
    "phase1_i_loss",
    "phase1_t_loss",
    "phase2_loss",
    "phase2_i_loss",
    "phase2_t_loss",
    "sample_controls",
    "targeted_phase_loss",
    "train",
]
