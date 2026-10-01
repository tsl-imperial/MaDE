# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Control sampling strategies for inverse-consistency training."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from made.models import InverseDynamics
from made.physics import ConstraintSet


def sample_controls(
    strategy: str,
    constraints: ConstraintSet,
    inverse_dynamics: InverseDynamics | None,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    step: int | jax.Array,
    total_steps: int,
    key: jax.Array,
) -> jax.Array:
    """Sample a single control according to the requested strategy.

    Args:
        strategy: One of ``"prior"``, ``"predictions"``, ``"mixture"``, ``"annealed"``.
        constraints: Constraint set providing the control box.
        inverse_dynamics: Inverse-dynamics model; required for every strategy except ``"prior"``.
        x_prev: Previous state.
        x_curr: Current state.
        params: Physical parameters.
        step: Current training step.
        total_steps: Total number of training steps.
        key: PRNG key.

    Returns:
        Sampled control vector.

    Raises:
        ValueError: If a non-prior strategy is requested without an inverse-dynamics model, or the
            strategy is unsupported.
    """
    prior = jax.random.uniform(
        key,
        constraints.control_min.shape,
        minval=constraints.control_min,
        maxval=constraints.control_max,
    )
    if strategy == "prior":
        return prior

    if inverse_dynamics is None:
        raise ValueError("An inverse-dynamics model is required for non-prior sampling strategies.")

    prediction = jax.lax.stop_gradient(inverse_dynamics(x_prev, x_curr, params))
    total = float(max(total_steps, 1))
    alpha = jnp.asarray(step, dtype=prediction.dtype) / total

    if strategy == "predictions":
        return prediction
    if strategy == "mixture":
        use_prediction = jax.random.bernoulli(key, alpha)
        return jnp.where(use_prediction, prediction, prior)
    if strategy == "annealed":
        anneal = 0.5 * (1.0 - jnp.cos(jnp.pi * alpha))
        return (1.0 - anneal) * prior + anneal * prediction

    raise ValueError(f"Unsupported control sampling strategy '{strategy}'.")
