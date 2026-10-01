# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Regression tests: corrector in-loop box projection.

Pins that after each GD step in both _correct_train and _correct_eval, the
updated u is projected back inside constraint box bounds BEFORE the next
dynamics.integrate call. An initial u=(0.6, 0.0) with δ_max=0.5 is used to
verify both the post-projection box invariant and the consistency between the
returned (x_corrected, u_corrected) pair.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax

from made.models import MaDECell
from made.physics import KinematicBicycleAsDynamicState, dynamic_bicycle_constraints
from made.utils import CorrectorConfig, ModelConfig


def _build_db_cell_for_projection(key: jax.Array) -> MaDECell:
    """Small MaDECell for DB-underspecified box-projection tests.

    Args:
        key: PRNG key for initialisation.

    Returns:
        Initialised cell.
    """
    physics = KinematicBicycleAsDynamicState()
    constraints = dynamic_bicycle_constraints()
    return MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(
            step_size=0.1,
            momentum=0.0,
            train_steps=2,
            eval_max_steps=2,
            # eval_tol=-1e9: condition is max_violation > eval_tol, which is always True
            # (any finite violation > -1e9), so the loop always runs all eval_max_steps.
            eval_tol=-1e9,
            mode="enabled",
        ),
        key=key,
        dt=0.1,
    )


def test_correct_train_projects_inside_loop() -> None:
    """_correct_train clips u into the constraint box every GD step.

    Starting from u=(0.6, 0.0) with δ_max=0.5 (out of box), after train_steps=2:
    - u_corrected[0] must be in [-0.5, 0.5].
    - x_corrected must equal dynamics.integrate(x_prev, u_corrected, params, dt)
      to float64 precision, pinning that projection happens INSIDE the body
      (before the final integrate call), not outside it.
    """
    key = jax.random.key(0)
    cell = _build_db_cell_for_projection(key)

    dynamics = cell.augmented_dynamics
    constraints = cell.constraints
    corrector = cell.corrector

    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_pred = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    # δ=0.6 is outside the box [−0.5, 0.5]
    u = jnp.array([0.6, 0.0])
    params = jnp.array([2.7])
    dt = 0.1

    x_corrected, u_corrected = corrector._correct_train(
        x_pred, u, constraints, dynamics, x_prev, params, dt,
        solver=None, adjoint=None,
    )

    # Post-projection box invariant
    assert jnp.abs(u_corrected[0]) <= 0.5 + 1e-6, (
        f"u_corrected[0]={float(u_corrected[0]):.6f} violates box bound 0.5"
    )

    # Consistency: x_corrected must equal integrate(x_prev, u_corrected)
    # This is bit-exact because _correct_train literally returns integrate(x_prev, u_final).
    # A projection placed OUTSIDE the loop would break this by returning x from
    # the unprojected u iterate.
    x_expected = dynamics.integrate(x_prev, u_corrected, params, dt)
    assert jnp.allclose(x_corrected, x_expected, rtol=1e-12, atol=1e-12), (
        f"x_corrected does not match integrate(x_prev, u_corrected): "
        f"max diff = {float(jnp.max(jnp.abs(x_corrected - x_expected))):.2e}"
    )


def test_correct_eval_projects_inside_loop() -> None:
    """_correct_eval clips u into the constraint box every GD step.

    Uses eval_tol=1e9 to force exactly 2 iterations (never triggers early-exit).
    Same assertions as the train-mode test, exercising the while_loop-based path.
    """
    key = jax.random.key(1)
    cell = _build_db_cell_for_projection(key)

    dynamics = cell.augmented_dynamics
    constraints = cell.constraints
    corrector = cell.corrector

    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_pred = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    u = jnp.array([0.6, 0.0])
    params = jnp.array([2.7])
    dt = 0.1

    x_corrected, u_corrected = corrector._correct_eval(
        x_pred, u, constraints, dynamics, x_prev, params, dt,
        solver=None, adjoint=None,
    )

    # Post-projection box invariant
    assert jnp.abs(u_corrected[0]) <= 0.5 + 1e-6, (
        f"u_corrected[0]={float(u_corrected[0]):.6f} violates box bound 0.5"
    )

    # Consistency: x_corrected must equal integrate(x_prev, u_corrected)
    x_expected = dynamics.integrate(x_prev, u_corrected, params, dt)
    assert jnp.allclose(x_corrected, x_expected, rtol=1e-12, atol=1e-12), (
        f"x_corrected does not match integrate(x_prev, u_corrected): "
        f"max diff = {float(jnp.max(jnp.abs(x_corrected - x_expected))):.2e}"
    )
