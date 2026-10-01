# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for apply_made_trajectory x0 seeding and eval_adaptive mode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from made.models import MaDECell
from made.physics import DoubleIntegrator
from made.physics.constraints import ConstraintSet
from made.upstream import apply_made_trajectory, apply_made_trajectory_with_controls

T, D = 4, 4


def _x_pred() -> jnp.ndarray:
    """Build a predicted trajectory for the diagnostics tests.

    Returns:
        Predicted trajectory of shape (T, state_dim).
    """
    return jnp.arange(T * D, dtype=jnp.float64).reshape(T, D) / 10.0


def _params(double_integrator: tuple[DoubleIntegrator, ConstraintSet]) -> jnp.ndarray:
    """Return zero physics parameters for the double integrator.

    Args:
        double_integrator: Physics and constraints fixture.

    Returns:
        Zero parameter vector.
    """
    physics, _ = double_integrator
    return jnp.zeros((physics.param_dim,))


def test_legacy_default_passes_first_row_through(
    small_cell: MaDECell,
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
) -> None:
    """Verify legacy default passes first row through."""
    x_pred = _x_pred()
    states, controls = apply_made_trajectory_with_controls(
        small_cell, x_pred, _params(double_integrator), 0.1
    )
    assert states.shape == (T, D)
    np.testing.assert_array_equal(np.asarray(states[0]), np.asarray(x_pred[0]))
    np.testing.assert_array_equal(np.asarray(controls[0]), np.zeros(controls.shape[1]))
    states_only = apply_made_trajectory(small_cell, x_pred, _params(double_integrator), 0.1)
    np.testing.assert_array_equal(np.asarray(states_only), np.asarray(states))


def test_x0_seeding_corrects_every_row(
    small_cell: MaDECell,
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
) -> None:
    """Verify x0 seeding corrects every row."""
    x_pred = _x_pred()
    x0 = jnp.zeros(D)
    params = _params(double_integrator)
    states, controls = apply_made_trajectory_with_controls(
        small_cell, x_pred, params, 0.1, x0=x0
    )
    assert states.shape == (T, D)
    assert controls.shape[0] == T
    # First output must equal a direct cell call seeded at x0 (training-mode path).
    expected_x, expected_u = small_cell(
        x0, x_pred[0], params, 0.1, training=True, correction_mode="train_fixed"
    )
    np.testing.assert_allclose(np.asarray(states[0]), np.asarray(expected_x))
    np.testing.assert_allclose(np.asarray(controls[0]), np.asarray(expected_u))


def test_eval_adaptive_matches_direct_eval_cell_call(
    small_cell: MaDECell,
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
) -> None:
    """Verify eval adaptive matches direct eval cell call."""
    x_pred = _x_pred()
    x0 = jnp.zeros(D)
    params = _params(double_integrator)
    states, controls = apply_made_trajectory_with_controls(
        small_cell, x_pred, params, 0.1, correction_mode="eval_adaptive", x0=x0
    )
    expected_x, expected_u = small_cell(x0, x_pred[0], params, 0.1, training=False)
    np.testing.assert_allclose(np.asarray(states[0]), np.asarray(expected_x))
    np.testing.assert_allclose(np.asarray(controls[0]), np.asarray(expected_u))
    assert bool(jnp.all(jnp.isfinite(states)))


def test_unknown_correction_mode_raises(
    small_cell: MaDECell,
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
) -> None:
    """Verify unknown correction mode raises."""
    with pytest.raises(ValueError, match="correction_mode"):
        apply_made_trajectory(
            small_cell, _x_pred(), _params(double_integrator), 0.1, correction_mode="bogus"
        )


def test_enabled_corrector_reduces_speed_constraint_violation(
    small_cell: MaDECell, double_integrator: tuple[DoubleIntegrator, ConstraintSet]
) -> None:
    """With the corrector enabled (small_cell's CorrectorConfig default 'enabled',
    eval_max_steps=4), correcting a trajectory that violates the DI speed-norm /
    box constraints must strictly reduce total constraint violation relative to
    the raw proposals.

    Uses a MOVING x0 (with x_prev=0 the DoubleIntegrator corrector trivially finds
    u=0 -> speed=0 -> zero violation after 2 steps, which would make the reduction
    vacuous), so the correction has to do real work against a nonzero carry state.
    """
    physics, constraints = double_integrator
    params = _params(double_integrator)
    dt = 0.1

    # Feasible moving seed: speed = sqrt(3^2 + 3^2) ~= 4.24 < v_max = 5.
    x0 = jnp.asarray([0.0, 0.0, 3.0, 3.0])
    # Proposals whose velocity gives speed ~= 8.49 > v_max = 5 (box + speed-norm violation).
    x_pred = jnp.asarray(
        [
            [1.0, 1.0, 6.0, 6.0],
            [2.0, 2.0, 6.0, 6.0],
            [3.0, 3.0, 6.0, 6.0],
            [4.0, 4.0, 6.0, 6.0],
        ]
    )

    states, controls = apply_made_trajectory_with_controls(
        small_cell, x_pred, params, dt, correction_mode="eval_adaptive", x0=x0
    )
    assert bool(jnp.all(jnp.isfinite(states)))
    assert bool(jnp.all(jnp.isfinite(controls)))

    raw_controls = jnp.zeros((x_pred.shape[0], physics.control_dim))
    raw_violation = jnp.sum(
        jax.vmap(lambda x, u: jnp.maximum(constraints.violation(x, u), 0))(x_pred, raw_controls)
    )
    corrected_violation = jnp.sum(
        jax.vmap(lambda x, u: jnp.maximum(constraints.violation(x, u), 0))(states, controls)
    )
    assert float(corrected_violation) < float(raw_violation)
