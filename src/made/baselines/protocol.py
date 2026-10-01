# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Shared protocol helpers for per-step correction baselines."""

from __future__ import annotations

from typing import Protocol

import jax


class CorrectionBaseline(Protocol):
    """Per-step baseline that returns corrected consecutive states."""

    def correct_pair(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Correct a consecutive state pair.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            metadata: Optional per-sample metadata.
        Returns:
            Tuple of corrected (x_prev, x_curr).
        """


def correct_pair(
    model: CorrectionBaseline,
    x_prev: jax.Array,
    x_curr: jax.Array,
    metadata: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Call the shared correction protocol.

    Args:
        model: Baseline implementing the correction protocol.
        x_prev: Previous state.
        x_curr: Current state.
        metadata: Optional per-sample metadata.
    Returns:
        Tuple of corrected (x_prev, x_curr).
    """
    return model.correct_pair(x_prev, x_curr, metadata)
