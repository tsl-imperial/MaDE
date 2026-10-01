# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from made.models import MaDECell
from made.physics import DoubleIntegrator, double_integrator_constraints
from made.utils import CorrectorConfig, ModelConfig
from made.utils.jax_setup import configure
from made.physics import ConstraintSet

configure()


@pytest.fixture
def small_key() -> jax.Array:
    """Fixed PRNG key (seed 0)."""
    return jax.random.key(0)


@pytest.fixture
def double_integrator() -> tuple[DoubleIntegrator, ConstraintSet]:
    """DoubleIntegrator physics model with its default constraint set."""
    return DoubleIntegrator(), double_integrator_constraints()


@pytest.fixture
def small_cell(
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
    small_key: jax.Array,
) -> MaDECell:
    """Small MaDECell on the double integrator with a two-step corrector."""
    physics, constraints = double_integrator
    return MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(train_steps=2, eval_max_steps=4),
        key=small_key,
    )


@pytest.fixture
def sample_batch(double_integrator: tuple[DoubleIntegrator, ConstraintSet]) -> dict[str, jax.Array]:
    """Batch of eight zero/constant double-integrator transitions."""
    physics, _ = double_integrator
    batch_size = 8
    x_prev = jnp.zeros((batch_size, physics.state_dim))
    x_curr = jnp.ones((batch_size, physics.state_dim)) * 0.1
    params = jnp.zeros((batch_size, physics.param_dim))
    u_gt = jnp.zeros((batch_size, physics.control_dim))
    return {"x_prev": x_prev, "x_curr": x_curr, "params": params, "u_gt": u_gt}
