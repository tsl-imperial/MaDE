"""Regression tests: corrector finiteness near the δ-singularity neighborhood.

Covers Layer 1 (kinematic_bicycle vector_field δ-clamp) and Layer 2 (corrector
in-loop box projection) working together to keep corrections finite when the
inverse-dynamics I-output δ is near the constraint box edge (0.49 rad, where
the box limit is 0.5 rad and tan(δ) starts climbing sharply).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from made.models import MaDECell
from made.physics import KinematicBicycleAsDynamicState, dynamic_bicycle_constraints
from made.utils import CorrectorConfig, ModelConfig


def _build_db_underspecified_cell(key: jax.Array) -> MaDECell:
    """Small MaDECell with KinematicBicycleAsDynamicState and DB constraints."""
    physics = KinematicBicycleAsDynamicState()
    constraints = dynamic_bicycle_constraints()
    return MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(
            step_size=0.01,
            momentum=0.0,
            train_steps=50,
            eval_max_steps=50,
            eval_tol=1e-6,
            mode="enabled",
        ),
        key=key,
        dt=0.1,
    )


def _patch_inverse_to_delta_near_edge(cell: MaDECell) -> MaDECell:
    """Monkey-patch inverse_dynamics so it always returns u=(0.49, 0.5) for any input.

    This places δ=0.49 rad right at the box edge (δ_max=0.5), exercising the
    Layer-1 clamp and Layer-2 projection without random initialization noise.
    """

    class _FixedInverse(eqx.Module):
        def __call__(self, x_prev: jax.Array, x_curr: jax.Array, params: jax.Array) -> jax.Array:
            del x_prev, x_curr, params
            return jnp.array([0.49, 0.5])

    return eqx.tree_at(lambda c: c.inverse_dynamics, cell, _FixedInverse())


def test_correct_train_finite_at_singularity_neighborhood():
    """Layer 1 + Layer 2: _correct_train stays finite with δ near the box edge.

    With 50 GD steps at step_size=0.01 starting from u=(0.49, 0.5), both the
    clamp in vector_field (Layer 1) and the per-step box projection in the
    corrector body (Layer 2) must prevent NaN/Inf from tan(δ) singularity.
    The post-projection bound asserts the tighter box limit (0.5+1e-6), NOT
    the legacy 1.4 rad ceiling — this pins a Layer-2 regression specifically.
    """
    key = jax.random.key(0)
    cell = _build_db_underspecified_cell(key)
    cell = _patch_inverse_to_delta_near_edge(cell)

    physics = cell.augmented_dynamics
    constraints = cell.constraints
    corrector = cell.corrector

    # Feasible initial state for DB-underspecified (6D: x,y,θ,v_x,v_y,ω)
    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    # x_pred: slightly displaced state, also feasible
    x_pred = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    u = jnp.array([0.49, 0.5])
    # KinematicBicycleAsDynamicState has param_dim=1 (wheelbase L)
    params = jnp.array([2.7])
    dt = 0.1

    x_final, u_final = corrector._correct_train(
        x_pred, u, constraints, physics, x_prev, params, dt,
        solver=None, adjoint=None,
    )

    assert jnp.isfinite(x_final).all(), f"x_final has non-finite values: {x_final}"
    assert jnp.isfinite(u_final).all(), f"u_final has non-finite values: {u_final}"
    # Post-projection box assertion (Layer 2) — tighter than 1.4 ceiling
    assert jnp.abs(u_final[0]) <= 0.5 + 1e-6, (
        f"u_final[0]={u_final[0]} exceeds box bound 0.5; Layer-2 projection may be missing"
    )


def test_correct_eval_finite_at_singularity_neighborhood():
    """Layer 1 + Layer 2: _correct_eval stays finite with δ near the box edge.

    Same setup as the train-mode test but uses the while_loop-based eval path.
    Also calls cell.__call__(training=False) end-to-end to exercise the full
    I-T-C pipeline in eval mode with the patched I-output.
    """
    key = jax.random.key(1)
    cell = _build_db_underspecified_cell(key)
    cell = _patch_inverse_to_delta_near_edge(cell)

    physics = cell.augmented_dynamics
    constraints = cell.constraints
    corrector = cell.corrector

    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_pred = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    u = jnp.array([0.49, 0.5])
    params = jnp.array([2.7])
    dt = 0.1

    x_final, u_final = corrector._correct_eval(
        x_pred, u, constraints, physics, x_prev, params, dt,
        solver=None, adjoint=None,
    )

    assert jnp.isfinite(x_final).all(), f"x_final has non-finite values: {x_final}"
    assert jnp.isfinite(u_final).all(), f"u_final has non-finite values: {u_final}"
    assert jnp.abs(u_final[0]) <= 0.5 + 1e-6, (
        f"u_final[0]={u_final[0]} exceeds box bound 0.5; Layer-2 projection may be missing"
    )

    # Full pipeline eval: cell.__call__(training=False)
    x_target = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    known_params = jnp.array([2.7])
    x_corrected, _u_corrected = cell(
        x_prev, x_target, known_params, dt, training=False
    )
    assert jnp.isfinite(x_corrected).all(), (
        f"cell(training=False) produced non-finite x_corrected: {x_corrected}"
    )
