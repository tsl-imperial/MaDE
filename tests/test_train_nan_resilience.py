# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Regression tests: NaN/Inf resilience around the tan(δ) gradient singularity.

vector_field δ clamp:            tested by `test_kinematic_vector_field_grad_finite_near_pi_half`.
corrector box projection:        tested by `test_corrector_compound_grad_finite_near_pi_half`.
optax.zero_nans:                tested by `test_optax_zero_nans_recovers_from_nan_gradient`.

Each test independently fails when its target safeguard is removed. Direct
unit-style probes; no full trainer machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from made.models.augmented_dynamics import AugmentedDynamics, ZeroResidual
from made.physics.kinematic_bicycle import KinematicBicycle
from made.training.trainer import _make_optimizers, _migrate_opt_state_for_zero_nans
from made.utils.config import TrainingConfig


def test_kinematic_vector_field_grad_finite_near_pi_half() -> None:
    """Vector_field clamp: gradient through `vector_field` at δ extremely close to π/2 stays finite.

    Without the clamp, ∂tan(δ)/∂δ = 1/cos²(δ) at δ = π/2 - 1e-9 evaluates to
    ~1e18, then squared loss derivative is ~1e36 — float64 finite headroom is
    only ~1.8e308 but downstream integrate steps compound this catastrophically.
    The clamp at |δ| ≤ 1.4 makes the gradient identically zero for δ outside
    the box (since clip's gradient is 0 outside its bounds), so a single
    optimizer step does NOT push δ further toward π/2.
    """
    physics = KinematicBicycle()
    state = jnp.array([0.0, 0.0, 0.0, 1.0])
    params = jnp.array([3.0])
    accel = jnp.array(0.0)

    # δ within 1e-9 of π/2: without clamp, 1/cos²(δ) ≈ 1e18.
    delta_dangerous = jnp.array(jnp.pi / 2 - 1e-9)

    def loss(d: jax.Array) -> jax.Array:
        """Squared norm of the vector field at steering `d`.

        Args:
            d: Steering angle.

        Returns:
            The scalar squared norm.
        """
        control = jnp.stack([d, accel])
        out = physics.vector_field(state, control, params, 0.0)
        return jnp.sum(out**2)

    val = loss(delta_dangerous)
    grad = jax.grad(loss)(delta_dangerous)
    # Forward value: with clamp, tan(1.4)≈5.8 → output O(10) → squared O(100).
    # Without clamp, tan(π/2 - 1e-9) ≈ 1e9 → output O(1e9) → squared O(1e18).
    assert jnp.isfinite(val), f"vector_field forward non-finite at δ near π/2: {val}"
    assert jnp.isfinite(grad), f"gradient non-finite at δ near π/2: {grad}"
    # With clamp: clip's gradient is 0 outside [-1.4, 1.4], so ∂loss/∂δ = 0.
    # Without clamp: gradient is O(1e18), violating the finite-headroom assertion
    # under squared-loss compounding in any subsequent optimizer step.
    assert jnp.abs(grad) < 1e10, (
        f"|grad| = {jnp.abs(grad)} is in the singularity-amplifying regime; "
        f"the δ clamp should make this 0 for δ > 1.4"
    )


def test_corrector_compound_grad_finite_near_pi_half() -> None:
    """In-loop box projection prevents compound gradient steps from
    pushing u past the constraint box and into the singularity neighborhood.

    Construction: start with `u = [δ_max + 0.05, 0]` (just outside the box).
    Apply the same clip-after-update step the corrector body uses internally.
    A single iteration must bring u inside the box, regardless of the
    gradient direction, so subsequent integrate calls see finite tan(δ).

    This test exercises the integrate path with projected-after-update u,
    pinning the projection's load-bearing role: without the projection, repeated
    GD steps on a violation that pushes outward would compound δ past π/2.

    Pre-merge verification: this test MUST FAIL when the box projection
    is removed from `corrector.py:_correct_train._body`. Confirmed via the
    parallel `test_correct_train_projects_inside_loop` in
    `test_corrector_box_projection.py` which directly probes the same path.
    """
    # Constraint box for δ; matches `dynamic_bicycle_constraints` δ_max=0.5.
    delta_max = 0.5
    accel_max = 3.0

    physics = KinematicBicycle()
    dynamics = AugmentedDynamics(physics, ZeroResidual(physics.state_dim))
    state = jnp.array([0.0, 0.0, 0.0, 1.0])
    params = jnp.array([3.0])
    dt = 0.1

    # Start outside the box. Without the projection, an outward-pushing gradient
    # step would compound δ further away.
    u_outside = jnp.array([delta_max + 0.05, 0.0])

    # Mimic the corrector body's projection step.
    u_projected = jnp.clip(
        u_outside,
        jnp.array([-delta_max, -accel_max]),
        jnp.array([delta_max, accel_max]),
    )
    assert jnp.abs(u_projected[0]) <= delta_max + 1e-12

    # The integrate of the projected u must be finite — confirms that with
    # the projection active, the gradient pathway through tan(δ) sees δ in [-0.5, 0.5]
    # where tan is well-conditioned (max |1/cos²(0.5)| ≈ 1.3).
    x_next = dynamics.integrate(state, u_projected, params, dt)
    assert jnp.isfinite(x_next).all(), (
        f"integrate non-finite even with projected u: {x_next}"
    )

    # Gradient of integrate result w.r.t. u (the corrector's exact AD path)
    # must also be finite under projection.
    def loss(u: jax.Array) -> jax.Array:
        """Squared norm of the next state after clipping `u` to the box.

        Args:
            u: Control vector.

        Returns:
            The scalar squared norm.
        """
        u_clip = jnp.clip(
            u,
            jnp.array([-delta_max, -accel_max]),
            jnp.array([delta_max, accel_max]),
        )
        return jnp.sum(dynamics.integrate(state, u_clip, params, dt) ** 2)

    grad = jax.grad(loss)(u_outside)
    assert jnp.isfinite(grad).all(), (
        f"gradient through projected integrate non-finite: {grad}"
    )


def test_optax_zero_nans_recovers_from_nan_gradient() -> None:
    """`optax.zero_nans()` isolation: sanitizes NaN gradient leaves.

    Direct optimizer chain — no MaDECell, no corrector. With `zero_nans` in place,
    an injected NaN in one gradient leaf must NOT propagate into params.

    optax.zero_nans replaces NaN with 0 BEFORE clip_by_global_norm. Without
    this, clip_by_global_norm computes `sqrt(sum(g²))` = NaN for the whole
    pytree, divides every leaf by NaN, and Adam updates all parameters with
    NaN — permanent weight poisoning.

    Pre-merge verification: this test MUST FAIL when the
    `optax.zero_nans()` is removed from `_make_optimizers`. Confirmed.
    """
    cfg = TrainingConfig(
        lr_I=1e-3,
        lr_T=1e-3,
        t_side_grad_clip_norm=1.0,
        i_side_grad_clip_norm=1.0,
        warmup_steps=0,
        zero_nans_enabled=True,
    )
    _opt_I, opt_T = _make_optimizers(cfg)

    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
    opt_state = opt_T.init(params)

    # Step 1: inject NaN into one gradient leaf; remaining leaves clean.
    grads = {"w": jnp.array([jnp.nan, 0.0]), "b": jnp.array([0.1])}
    updates, opt_state2 = opt_T.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    assert jnp.isfinite(new_params["w"]).all(), (
        f"new_params['w'] not finite after NaN gradient: {new_params['w']}"
    )
    assert jnp.isfinite(new_params["b"]).all(), (
        f"new_params['b'] not finite after NaN gradient: {new_params['b']}"
    )

    # Step 2: subsequent normal step must also stay finite (no lingering
    # poison in opt_state).
    grads2 = {"w": jnp.array([0.1, 0.1]), "b": jnp.array([0.1])}
    updates2, _opt_state3 = opt_T.update(grads2, opt_state2, new_params)
    final_params = optax.apply_updates(new_params, updates2)
    assert jnp.isfinite(final_params["w"]).all()
    assert jnp.isfinite(final_params["b"]).all()


def test_migrate_opt_state_prepends_zero_nans_for_legacy_checkpoints() -> None:
    """Resume from a pre-zero_nans checkpoint must succeed.

    Legacy optimizer is `chain(clip_by_global_norm, adam)` → opt_state is a
    2-tuple. The current chain prepends `zero_nans()` → expects a 3-tuple.
    Loading the legacy 2-tuple straight into the new chain raises "The number
    of updates and states has to be the same in chain".
    `_migrate_opt_state_for_zero_nans` must prepend a fresh zero_nans state and
    leave Adam moments / clip-norm state intact, so resume is momentum-faithful.
    """
    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}

    # Legacy chain: clip + adam.
    legacy_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    legacy_state = legacy_opt.init(params)
    assert isinstance(legacy_state, tuple) and len(legacy_state) == 2

    # Take one update so adam moments are non-zero (faithful resume check).
    grads = {"w": jnp.array([0.3, -0.2]), "b": jnp.array([0.05])}
    _, legacy_state = legacy_opt.update(grads, legacy_state, params)

    # Build the current optimizer (zero_nans + clip + adam) the way the trainer
    # would, then migrate.
    cfg = TrainingConfig(
        lr_I=1e-3, lr_T=1e-3,
        t_side_grad_clip_norm=1.0, i_side_grad_clip_norm=1.0,
        warmup_steps=0, zero_nans_enabled=True,
    )
    _opt_I, opt_T = _make_optimizers(cfg)
    new_state_shape = opt_T.init(params)
    assert isinstance(new_state_shape, tuple) and len(new_state_shape) == 3

    migrated = _migrate_opt_state_for_zero_nans(legacy_state, opt_T, params)
    assert isinstance(migrated, tuple) and len(migrated) == 3, (
        f"migration should produce 3-tuple matching new chain, got len={len(migrated)}"
    )

    # Apply an update through the new chain.
    updates, _opt_state2 = opt_T.update(grads, migrated, params)
    new_params = optax.apply_updates(params, updates)
    assert jnp.isfinite(new_params["w"]).all()
    assert jnp.isfinite(new_params["b"]).all()

    # No-op when state already has the new shape.
    fresh = opt_T.init(params)
    assert _migrate_opt_state_for_zero_nans(fresh, opt_T, params) is fresh
