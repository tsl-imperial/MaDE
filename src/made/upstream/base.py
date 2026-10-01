# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Base interface for upstream predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
import jax


class UpstreamPredictor(eqx.Module, ABC):
    """Abstract upstream predictor."""

    @abstractmethod
    def __call__(self, context: jax.Array) -> jax.Array:
        """Map context features to a predicted state trajectory.

        Args:
            context: Context features.
        Returns:
            Predicted state trajectory.
        """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the output state dimension.

        Returns:
            State dimension.
        """
