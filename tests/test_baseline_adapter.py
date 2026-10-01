# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Baseline trajectory adapter tests."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.baselines import ClampBaseline, apply_baseline_over_trajectory
from made.physics import double_integrator_constraints
from made.physics import ConstraintSet


@pytest.fixture
def di_constraints() -> ConstraintSet:
    """Default double-integrator constraint set."""
    return double_integrator_constraints()


def test_clamp_smoke(di_constraints: ConstraintSet) -> None:
    """apply_baseline_over_trajectory with ClampBaseline returns correct shapes."""
    state_dim, control_dim = 4, 2
    T = 5
    baseline = ClampBaseline(constraints=di_constraints)
    states = jnp.ones((T, state_dim), dtype=jnp.float64) * 0.5
    controls = jnp.zeros((T - 1, control_dim), dtype=jnp.float64)

    x_corr, u_corr = apply_baseline_over_trajectory(baseline, states, controls, dt=0.1)

    assert x_corr.shape == (T, state_dim), f"Expected ({T}, {state_dim}), got {x_corr.shape}"
    assert u_corr.shape == (T - 1, control_dim), f"Expected ({T - 1}, {control_dim}), got {u_corr.shape}"
    assert jnp.all(jnp.isfinite(x_corr))
    assert jnp.all(jnp.isfinite(u_corr))


def test_clamp_clips_out_of_bound_states(di_constraints: ConstraintSet) -> None:
    """ClampBaseline correction should keep all corrected states within bounds."""
    state_dim, control_dim = 4, 2
    T = 5
    baseline = ClampBaseline(constraints=di_constraints)
    # States intentionally outside bounds (all 100)
    states = jnp.ones((T, state_dim), dtype=jnp.float64) * 100.0
    controls = jnp.zeros((T - 1, control_dim), dtype=jnp.float64)

    x_corr, _ = apply_baseline_over_trajectory(baseline, states, controls, dt=0.1)

    assert jnp.all(x_corr <= di_constraints.state_max + 1e-9)
    assert jnp.all(x_corr >= di_constraints.state_min - 1e-9)



def test_output_is_zero_filled_controls(di_constraints: ConstraintSet) -> None:
    """u_corr from baselines is zero-filled (baselines don't infer controls)."""
    state_dim, control_dim = 4, 2
    T = 6
    baseline = ClampBaseline(constraints=di_constraints)
    states = jnp.ones((T, state_dim), dtype=jnp.float64) * 0.1
    controls = jnp.ones((T - 1, control_dim), dtype=jnp.float64)

    _, u_corr = apply_baseline_over_trajectory(baseline, states, controls, dt=0.1)
    assert jnp.all(u_corr == 0.0)
