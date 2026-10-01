# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Production-dt round-trip tests for known_control_prior on KB and DynBike.

These tests pin the Newton-on-Heun inverse priors at the production dt=0.1 used
in the simulated experiments.  Heun integration of (delta, a) under ZOH constant control is the
forward map the trainer actually uses; the inverse priors are constructed to be
exact (KB, closed-form) or machine-zero accurate (DynBike, Newton-on-Heun)
against that same forward map.

Distinct from ``test_known_inverse_prior.py`` which uses dt=0.001 to mask the
old O(dt) finite-difference bias floor.  These tests would have failed loudly
under the previous FD-based priors (~5e-6 KB inverse_consistency, ~1e-5 DynBike
delta_i_norm at dt=0.1).
"""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.physics import (
    DynamicBicycle,
    KinematicBicycle,
    KinematicBicycleAsDynamicState,
    KinematicBicycleFieldData,
    resolve_params,
)
from made.physics import PhysicsModel


_DT = 0.1


def _heun_step(
    physics: PhysicsModel, x0: jax.Array, control: jax.Array, params: jax.Array, dt: float
) -> jax.Array:
    """Single diffrax Heun step at constant control over [0, dt].

    Args:
        physics: Physics model.
        x0: Initial state.
        control: Constant control.
        params: Physics parameters.
        dt: Step size.

    Returns:
        State after one step.
    """
    term = diffrax.ODETerm(lambda t, y, args: physics.vector_field(y, control, params, t))
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Heun(),
        t0=0.0,
        t1=dt,
        dt0=dt,
        y0=x0,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=4,
    )
    return sol.ys[-1]


# KinematicBicycle: closed-form Heun inverse must be exact (machine zero).


def test_kinematic_bicycle_inverse_exact_at_production_dt() -> None:
    """KB Heun inverse with v_avg = (v_prev + v_curr)/2 must recover (delta, a)
    to machine precision at production dt=0.1.

    Heun integration of theta_dot = v*tan(delta)/L is exact under ZOH constant
    control because v is linear in time and tan(delta) is constant — the Heun
    quadrature reproduces the analytic integral.
    """
    physics = KinematicBicycle()
    params = jnp.array([2.7], dtype=jnp.float64)
    delta_true = jnp.asarray(0.20, dtype=jnp.float64)   # ~11.5 deg
    accel_true = jnp.asarray(1.5, dtype=jnp.float64)
    control = jnp.array([delta_true, accel_true], dtype=jnp.float64)
    x0 = jnp.array([0.0, 0.0, 0.0, 5.0], dtype=jnp.float64)

    x1 = _heun_step(physics, x0, control, params, _DT)
    u_pred = physics.known_control_prior(x0, x1, params, _DT)

    abs_err_delta = jnp.abs(u_pred[0] - delta_true)
    abs_err_accel = jnp.abs(u_pred[1] - accel_true)
    assert abs_err_delta < 1e-10, (
        f"KB delta absolute error {float(abs_err_delta):.3e} exceeds 1e-10 "
        f"at production dt={_DT}; Heun inverse should be analytic-exact."
    )
    assert abs_err_accel < 1e-10, (
        f"KB accel absolute error {float(abs_err_accel):.3e} exceeds 1e-10 "
        f"at production dt={_DT}; ZOH inverse on v should be exact."
    )


def test_kinematic_bicycle_inverse_stationary_delta_zero_at_production_dt() -> None:
    """Option D: |v_avg| < 0.5 m/s → δ = 0 by fiat; acceleration is ZOH-exact.

    Stationary fallback is a primitive-level convention: the arctan singularity
    near v_avg=0 produces ±π/2 spikes driven by sensor-jitter Δθ. We define δ=0
    in that regime to avoid the spurious controls, while keeping accel = (v_curr
    - v_prev)/dt which is well-defined regardless of v_avg.

    The convention applies to FIELD DATA ONLY, so this asserts it on
    ``KinematicBicycleFieldData``. The class it used to assert on,
    ``KinematicBicycle``, is the analytic inverse the solver appendix describes, and
    ``test_kinematic_bicycle_base_keeps_the_analytic_arctan`` below pins that it does NOT
    zero δ — which is the behaviour the simulated ladder needs and had silently lost.
    """
    physics = KinematicBicycleFieldData()
    params = jnp.array([2.7], dtype=jnp.float64)
    dt = _DT

    # Case 1: both endpoints at rest with mild heading jitter (sensor noise).
    x_prev = jnp.array([0.0, 0.0, 0.00, 0.0], dtype=jnp.float64)
    x_curr = jnp.array([0.0, 0.0, 0.05, 0.0], dtype=jnp.float64)
    u_pred = physics.known_control_prior(x_prev, x_curr, params, dt)
    assert jnp.all(jnp.isfinite(u_pred))
    assert float(u_pred[0]) == 0.0, (
        f"expected δ=0 on stationary pair, got {float(u_pred[0])}"
    )
    assert float(u_pred[1]) == 0.0, (
        f"expected a=0 on no-velocity-change pair, got {float(u_pred[1])}"
    )

    # Case 2: low-speed pair (v_avg = 0.3 m/s, below 0.5 threshold) with heading change.
    x_prev = jnp.array([0.0, 0.0, 0.00, 0.2], dtype=jnp.float64)
    x_curr = jnp.array([0.02, 0.0, 0.10, 0.4], dtype=jnp.float64)
    u_pred = physics.known_control_prior(x_prev, x_curr, params, dt)
    assert float(u_pred[0]) == 0.0, (
        f"low-speed pair should yield δ=0, got {float(u_pred[0])}"
    )
    # accel = (0.4 - 0.2) / 0.1 = 2.0 m/s² — ZOH-exact even in stationary branch.
    assert jnp.abs(u_pred[1] - 2.0) < 1e-12, (
        f"low-speed accel should be ZOH-exact, got {float(u_pred[1])}"
    )

    # Case 3: boundary — v_avg = 0.5 exactly. Threshold is strict `<`, so 0.5
    # takes the non-stationary branch (existing Heun-exact behavior).
    x_prev = jnp.array([0.0, 0.0, 0.0, 0.5], dtype=jnp.float64)
    x_curr = jnp.array([0.05, 0.0, 0.0, 0.5], dtype=jnp.float64)  # no heading change
    u_pred = physics.known_control_prior(x_prev, x_curr, params, dt)
    # arctan(L * 0 / (0.5 * 0.1)) = 0 — same answer either branch when dtheta=0.
    assert jnp.abs(u_pred[0]) < 1e-12
    assert jnp.abs(u_pred[1]) < 1e-12


def test_kinematic_bicycle_inverse_exact_multi_regime() -> None:
    """KB inverse exact across mild/strong/aggressive (delta, a) at dt=0.1."""
    physics = KinematicBicycle()
    params = jnp.array([2.7], dtype=jnp.float64)
    x0 = jnp.array([0.0, 0.0, 0.0, 6.0], dtype=jnp.float64)
    cases = [
        jnp.array([0.05, 0.5], dtype=jnp.float64),   # mild
        jnp.array([0.20, 1.5], dtype=jnp.float64),   # strong
        jnp.array([0.35, -2.0], dtype=jnp.float64),  # aggressive
    ]
    for control in cases:
        x1 = _heun_step(physics, x0, control, params, _DT)
        u_pred = physics.known_control_prior(x0, x1, params, _DT)
        assert jnp.all(jnp.abs(u_pred - control) < 1e-10), (
            f"KB inverse failed at control {control}: pred {u_pred}"
        )


# DynamicBicycle: Newton-on-Heun inverse should match Heun forward to ~1e-8.


def test_dynamic_bicycle_inverse_machine_zero_at_production_dt() -> None:
    """DynBike Newton-on-Heun inverse recovers (delta, a) to ~1e-8 at dt=0.1.

    With ``_NEWTON_HEUN_STEPS = 3`` outer iterations, the residual converges
    quadratically once the warm start is in the basin of attraction; absolute
    errors are typically machine-zero (~1e-12) for moderate manoeuvres.
    """
    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    delta_true = jnp.asarray(0.15, dtype=jnp.float64)
    accel_true = jnp.asarray(1.5, dtype=jnp.float64)
    control = jnp.array([delta_true, accel_true], dtype=jnp.float64)
    x0 = jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=jnp.float64)

    x1 = _heun_step(physics, x0, control, params, _DT)
    u_pred = physics.known_control_prior(x0, x1, params, _DT)

    abs_err_delta = jnp.abs(u_pred[0] - delta_true)
    abs_err_accel = jnp.abs(u_pred[1] - accel_true)
    assert abs_err_delta < 1e-8, (
        f"DynBike delta absolute error {float(abs_err_delta):.3e} exceeds 1e-8 "
        f"at production dt={_DT}; Newton-on-Heun should converge to machine zero."
    )
    assert abs_err_accel < 1e-8, (
        f"DynBike accel absolute error {float(abs_err_accel):.3e} exceeds 1e-8 "
        f"at production dt={_DT}; Newton-on-Heun should converge to machine zero."
    )


def test_dynamic_bicycle_inverse_multi_regime_at_production_dt() -> None:
    """DynBike inverse stays accurate across mild/strong/aggressive at dt=0.1."""
    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    cases = [
        # (x0, control)
        (jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=jnp.float64),
         jnp.array([0.05, 0.5], dtype=jnp.float64)),
        (jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=jnp.float64),
         jnp.array([0.15, 1.5], dtype=jnp.float64)),
        (jnp.array([0.0, 0.0, 0.0, 12.0, 0.5, 0.1], dtype=jnp.float64),
         jnp.array([0.25, -2.0], dtype=jnp.float64)),
    ]
    for x0, control in cases:
        x1 = _heun_step(physics, x0, control, params, _DT)
        u_pred = physics.known_control_prior(x0, x1, params, _DT)
        abs_err = jnp.abs(u_pred - control)
        assert jnp.all(abs_err < 1e-7), (
            f"DynBike inverse error {abs_err} exceeds 1e-7 at control {control}"
        )


def test_dynamic_bicycle_inverse_vmap_at_production_dt() -> None:
    """jax.vmap over a batch of Heun-generated pairs returns finite, accurate u."""
    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    delta_true = jnp.asarray(0.15, dtype=jnp.float64)
    accel_true = jnp.asarray(1.5, dtype=jnp.float64)
    control = jnp.array([delta_true, accel_true], dtype=jnp.float64)
    x0 = jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=jnp.float64)
    x1 = _heun_step(physics, x0, control, params, _DT)

    B = 8
    x0_batch = jnp.tile(x0[None], (B, 1))
    x1_batch = jnp.tile(x1[None], (B, 1))
    p_batch = jnp.tile(params[None], (B, 1))

    def prior_single(xp: jax.Array, xc: jax.Array, p: jax.Array) -> jax.Array:
        """Known control prior for a single transition.

        Args:
            xp: Previous state.
            xc: Current state.
            p: Parameters.

        Returns:
            Prior control.
        """
        return physics.known_control_prior(xp, xc, p, _DT)

    out = jax.vmap(prior_single)(x0_batch, x1_batch, p_batch)
    assert out.shape == (B, 2)
    assert jnp.all(jnp.isfinite(out))
    # All rows must be the exact same recovered control.
    assert jnp.all(jnp.abs(out - control[None]) < 1e-8)


# KinematicBicycleAsDynamicState delegation regression: bytewise equality.


def test_kinematic_bicycle_as_dynamic_state_delegation_is_identical() -> None:
    """KBAsDyn.known_control_prior == KB.known_control_prior on sliced state.

    The underspecified-condition wrapper must continue to delegate exactly to
    the underlying KinematicBicycle on the (x, y, theta, v) slice.  Anything
    else would break the kinematic-by-design contract for the underspecified
    condition (CLAUDE.md: 'KinematicBicycleAsDynamicState's prior delegation
    stay kinematic by design').
    """
    kb = KinematicBicycle()
    kb_as_dyn = KinematicBicycleAsDynamicState()
    params = jnp.array([2.7], dtype=jnp.float64)
    dt = _DT

    # Several distinct (prev, curr) pairs to ensure the equality is structural.
    pairs = [
        (jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0], dtype=jnp.float64),
         jnp.array([0.5, 0.0, 0.05, 5.1, 0.0, 0.0], dtype=jnp.float64)),
        (jnp.array([1.0, 2.0, 0.1, 5.0, 0.3, 0.05], dtype=jnp.float64),
         jnp.array([1.5, 2.5, 0.15, 5.1, 0.4, 0.05], dtype=jnp.float64)),
        (jnp.array([-3.0, 4.0, -0.2, 7.0, -0.1, 0.02], dtype=jnp.float64),
         jnp.array([-2.5, 4.3, -0.15, 7.2, -0.05, 0.02], dtype=jnp.float64)),
    ]
    for x_prev_6, x_curr_6 in pairs:
        u_dyn = kb_as_dyn.known_control_prior(x_prev_6, x_curr_6, params, dt)
        u_kin = kb.known_control_prior(x_prev_6[:4], x_curr_6[:4], params, dt)
        assert jnp.array_equal(u_dyn, u_kin), (
            f"KBAsDyn delegation diverged from KB.\n"
            f"  prev={x_prev_6}\n  curr={x_curr_6}\n"
            f"  u_dyn={u_dyn}\n  u_kin={u_kin}"
        )


def test_kinematic_bicycle_base_keeps_the_analytic_arctan() -> None:
    """The base class must NOT zero δ, and the value it returns is why the split exists.

    On a jittery at-rest pair the analytic inverse returns ≈ ±π/2 — the spurious steering
    the low-speed fallback was introduced to suppress. That is correct to suppress on
    recorded data and wrong to suppress on simulated data, which has no jitter. The split
    rests on that premise alone. The fallback IS what shifted the published simulated numbers
    on the seeds that ran under it.
    """
    physics = KinematicBicycle()
    params = jnp.array([2.7], dtype=jnp.float64)
    x_prev = jnp.array([0.0, 0.0, 0.00, 0.0], dtype=jnp.float64)
    x_curr = jnp.array([0.0, 0.0, 0.05, 0.0], dtype=jnp.float64)
    u = physics.known_control_prior(x_prev, x_curr, params, _DT)
    assert jnp.all(jnp.isfinite(u))
    assert float(u[0]) != 0.0, "the analytic base must not apply the stationary fallback"
    assert abs(abs(float(u[0])) - jnp.pi / 2) < 1e-3, (
        f"expected the near-±π/2 arctan spike the fallback exists to suppress, "
        f"got {float(u[0])}"
    )
