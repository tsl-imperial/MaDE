# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""MaDE model components."""

from made.models.augmented_dynamics import AugmentedDynamics, ResidualNetwork, ZeroResidual
from made.models.corrector import Corrector
from made.models.encoder import MetadataEncoder
from made.models.inverse_dynamics import InverseDynamics
from made.models.made_cell import MaDECell
from made.models.made_model import MaDEModel

__all__ = [
    "AugmentedDynamics",
    "Corrector",
    "InverseDynamics",
    "MaDECell",
    "MaDEModel",
    "MetadataEncoder",
    "ResidualNetwork",
    "ZeroResidual",
]
