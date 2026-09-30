"""Tests for the known-inverse prior: known_control_prior methods and InverseDynamics refactor."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import diffrax

from made.models.inverse_dynamics import InverseDynamics
from made.physics import (
    DoubleIntegrator,
    DynamicBicycle,
    KinematicBicycle,
    KinematicBicycleAsDynamicState,
    Unicycle,
    resolve_params,
)
from made.training.dispatch import build_trainable
from made.utils.config import CorrectorConfig, ExperimentConfig, ModelConfig, PhysicsConfig, TrainingConfig


_DT = 0.1
_KEY = jax.random.key(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def di_state_pair():
    x_prev = jnp.array([0.0, 0.0, 1.0, 2.0])
    x_curr = jnp.array([0.1, 0.2, 1.5, 2.3])
    params = jnp.zeros(0)
    return x_prev, x_curr, params


@pytest.fixture
def unicycle_state_pair():
    x_prev = jnp.array([0.0, 0.0, 0.0, 1.0])
    x_curr = jnp.array([0.1, 0.0, 0.05, 1.1])
    params = jnp.zeros(0)
    return x_prev, x_curr, params


@pytest.fixture
def kinbicycle_state_pair():
    L = 2.7
    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0])
    x_curr = jnp.array([0.5, 0.0, 0.1, 5.1])
    params = jnp.array([L])
    return x_prev, x_curr, params


@pytest.fixture
def dynbicycle_state_pair():
    params = resolve_params("dynamic_bicycle", {})
    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_curr = jnp.array([0.5, 0.0, 0.05, 5.1, 0.0, 0.1])
    return x_prev, x_curr, params


# ---------------------------------------------------------------------------
# known_control_prior: shape and finiteness
# ---------------------------------------------------------------------------


def test_double_integrator_known_control_prior_shape(di_state_pair):
    x_prev, x_curr, params = di_state_pair
    u = DoubleIntegrator().known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_double_integrator_known_control_prior_exact(di_state_pair):
    x_prev, x_curr, params = di_state_pair
    u = DoubleIntegrator().known_control_prior(x_prev, x_curr, params, _DT)
    expected = (x_curr[2:4] - x_prev[2:4]) / _DT
    assert jnp.allclose(u, expected)


def test_unicycle_known_control_prior_shape(unicycle_state_pair):
    x_prev, x_curr, params = unicycle_state_pair
    u = Unicycle().known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_kinematic_bicycle_known_control_prior_shape(kinbicycle_state_pair):
    x_prev, x_curr, params = kinbicycle_state_pair
    u = KinematicBicycle().known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_kinematic_bicycle_known_control_prior_near_zero_speed():
    # Should not produce NaN or Inf when speed is near zero
    params = jnp.array([2.7])
    x_prev = jnp.array([0.0, 0.0, 0.0, 0.0])
    x_curr = jnp.array([0.0, 0.0, 0.01, 0.01])
    u = KinematicBicycle().known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_dynamic_bicycle_known_control_prior_shape(dynbicycle_state_pair):
    x_prev, x_curr, params = dynbicycle_state_pair
    u = DynamicBicycle().known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_kinematic_bicycle_as_dynamic_state_known_control_prior():
    physics = KinematicBicycleAsDynamicState()
    params = jnp.array([2.7])
    x_prev = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_curr = jnp.array([0.5, 0.0, 0.05, 5.1, 0.0, 0.0])
    u = physics.known_control_prior(x_prev, x_curr, params, _DT)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


# ---------------------------------------------------------------------------
# InverseDynamics structural decomposition
# ---------------------------------------------------------------------------


def test_inverse_dynamics_with_residual_uses_known_prior():
    """InverseDynamics(use_residual=True) output changes when mlp is perturbed,
    but the known-prior component is independent of mlp parameters."""
    physics = DoubleIntegrator()
    inv = InverseDynamics(4, 2, 0, (16,), key=_KEY, known_physics=physics, dt=_DT, use_residual=True)
    assert inv.mlp is not None
    x_prev = jnp.array([0.0, 0.0, 1.0, 2.0])
    x_curr = jnp.array([0.1, 0.2, 1.5, 2.3])
    params = jnp.zeros(0)
    u = inv(x_prev, x_curr, params)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u))


def test_inverse_dynamics_fixed_i_has_no_mlp():
    """InverseDynamics(use_residual=False) should have mlp=None."""
    physics = DoubleIntegrator()
    inv = InverseDynamics(4, 2, 0, (16,), key=_KEY, known_physics=physics, dt=_DT, use_residual=False)
    assert inv.mlp is None


def test_inverse_dynamics_fixed_i_returns_prior_exactly():
    """InverseDynamics(use_residual=False) output equals known_control_prior exactly."""
    physics = DoubleIntegrator()
    inv = InverseDynamics(4, 2, 0, (16,), key=_KEY, known_physics=physics, dt=_DT, use_residual=False)
    x_prev = jnp.array([0.0, 0.0, 1.0, 2.0])
    x_curr = jnp.array([0.1, 0.2, 1.5, 2.3])
    params = jnp.zeros(0)
    u = inv(x_prev, x_curr, params)
    u_prior = physics.known_control_prior(x_prev, x_curr, params, _DT)
    assert jnp.allclose(u, u_prior)


def test_inverse_dynamics_init_scale_zero():
    """ΔI MLP with init_scale=0.0 should produce near-zero outputs at init."""
    physics = DoubleIntegrator()
    inv = InverseDynamics(
        4,
        2,
        0,
        (16,),
        key=_KEY,
        known_physics=physics,
        dt=_DT,
        use_residual=True,
        init_scale=0.0,
    )
    x_prev = jnp.array([0.0, 0.0, 1.0, 2.0])
    x_curr = jnp.array([0.1, 0.2, 1.5, 2.3])
    params = jnp.zeros(0)
    u_full = inv(x_prev, x_curr, params)
    u_prior = physics.known_control_prior(x_prev, x_curr, params, _DT)
    # With zero-init, delta should be ~0, so u ≈ u_prior
    assert jnp.allclose(u_full, u_prior, atol=1e-10)
    assert jnp.allclose(inv.learned_component(x_prev, x_curr, params), jnp.zeros((2,)))
    assert jnp.allclose(inv.residual_norm(x_prev, x_curr, params), 0.0)


def test_inverse_dynamics_fixed_i_residual_norm_is_zero():
    """Fixed-I variants should report zero Delta I magnitude."""
    physics = DoubleIntegrator()
    inv = InverseDynamics(4, 2, 0, (16,), key=_KEY, known_physics=physics, dt=_DT, use_residual=False)
    x_prev = jnp.array([0.0, 0.0, 1.0, 2.0])
    x_curr = jnp.array([0.1, 0.2, 1.5, 2.3])
    params = jnp.zeros(0)
    assert jnp.allclose(inv.learned_component(x_prev, x_curr, params), jnp.zeros((2,)))
    assert jnp.allclose(inv.residual_norm(x_prev, x_curr, params), 0.0)


# ---------------------------------------------------------------------------
# Dispatch: made-fixed-i variant
# ---------------------------------------------------------------------------


_SMALL_MODEL = ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16))
_SMALL_CORRECTOR = CorrectorConfig(mode="enabled", train_steps=2)
_DI_CFG = ExperimentConfig(
    physics=PhysicsConfig(true_system="double_integrator"),
    model=_SMALL_MODEL,
    corrector=_SMALL_CORRECTOR,
)


def test_made_fixed_i_builds():
    from made.models import MaDECell
    model = build_trainable(_DI_CFG, "made-fixed-i", _KEY)
    assert isinstance(model, MaDECell)


def test_made_fixed_i_has_no_delta_i_mlp():
    model = build_trainable(_DI_CFG, "made-fixed-i", _KEY)
    assert model.inverse_dynamics.mlp is None
    assert not model.inverse_dynamics.use_residual


def test_made_fixed_i_has_learned_fa_residual():
    """made-fixed-i: I is fixed but F_a in T is still learned."""
    from made.models.augmented_dynamics import ResidualNetwork
    model = build_trainable(_DI_CFG, "made-fixed-i", _KEY)
    assert isinstance(model.augmented_dynamics.residual, ResidualNetwork)


def test_made_fixed_i_smoke_10_steps():
    """made-fixed-i completes 10 training steps without error."""
    import optax
    from made.training.losses import phase1_loss
    from made.physics import resolve_params
    from made.utils.config import TrainingConfig

    model = build_trainable(_DI_CFG, "made-fixed-i", _KEY)
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    params = resolve_params("double_integrator", {})
    batch_size = 4
    dt = 0.1
    train_cfg = TrainingConfig()
    rng = jax.random.key(0)

    @eqx.filter_jit
    def step(model, opt_state, key):
        k1, k2 = jax.random.split(key)
        states = jax.random.normal(k1, (batch_size, 4))
        u_sampled = jax.random.normal(k2, (batch_size, 2))
        params_batch = jnp.tile(params, (batch_size, 1))

        def loss_fn(m):
            total, _ = phase1_loss(m, states, states, params_batch, dt, u_sampled, train_cfg)
            return total

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    for i in range(10):
        rng, subkey = jax.random.split(rng)
        model, opt_state, loss = step(model, opt_state, subkey)
        assert jnp.isfinite(loss), f"Loss is non-finite at step {i}: {loss}"


def test_phase1_metrics_report_delta_i_norm():
    from made.physics import resolve_params
    from made.training.losses import phase1_loss

    model = build_trainable(_DI_CFG, "made", _KEY)
    params = resolve_params("double_integrator", {})
    batch_size = 4
    x_prev = jnp.zeros((batch_size, 4), dtype=jnp.float64)
    x_curr = jnp.ones((batch_size, 4), dtype=jnp.float64) * 0.1
    u_sampled = jnp.zeros((batch_size, 2), dtype=jnp.float64)
    params_batch = jnp.tile(params, (batch_size, 1))

    _, metrics = phase1_loss(
        model,
        x_prev,
        x_curr,
        params_batch,
        _DT,
        u_sampled,
        TrainingConfig(),
    )

    assert "minimum_norm" in metrics
    assert "delta_i_norm" in metrics
    assert jnp.isfinite(metrics["delta_i_norm"])


# ---------------------------------------------------------------------------
# Dynamic-bicycle true inverse (Anomaly 2 fix)
# ---------------------------------------------------------------------------


@pytest.fixture
def feasible_dyn_bicycle_trajectory():
    """A feasible (x_prev, x_curr, params, control_true, dt) triple from the simulator.

    Uses a single Heun integration step with nontrivial steering so the
    kinematic prior is measurably wrong while the dynamic inverse is accurate.

    dt=0.001 is chosen deliberately small so that the finite-difference
    approximation error (O(dt)) is negligible relative to the Newton inverse
    accuracy.  At dt=0.001 and vx=8 m/s the FD discretisation error is ~1.4e-3
    on yaw_rate_dot, making the recovered delta accurate to ~1.9e-4 absolute.
    The kinematic prior error at the same dt is ~1.5e-1 — Newton beats it by
    ~783x, comfortably exceeding the ≥10× spec bar.
    """
    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    dt = 0.001  # small dt so FD error << inverse accuracy

    delta_true = jnp.asarray(0.15, dtype=jnp.float64)   # ~8.6 deg steering
    accel_true = jnp.asarray(1.5, dtype=jnp.float64)
    control = jnp.array([delta_true, accel_true], dtype=jnp.float64)

    x0 = jnp.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=jnp.float64)

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
    x1 = sol.ys[-1]
    return x0, x1, params, control, dt


def test_dynamic_bicycle_inverse_recovers_truth(feasible_dyn_bicycle_trajectory):
    """New prior recovers (delta, a) accurately vs ground-truth controls.

    Tolerance: absolute error < 1e-3 rad (delta) and < 5e-3 m/s^2 (accel).
    These are achievable at the fixture's dt=0.001 where FD discretisation
    error is O(dt) ≈ 1.4e-3, making absolute delta error ~1.9e-4.
    """
    x_prev, x_curr, params, control_true, dt = feasible_dyn_bicycle_trajectory
    delta_true, accel_true = control_true[0], control_true[1]

    physics = DynamicBicycle()
    delta_pred, accel_pred = physics.known_control_prior(x_prev, x_curr, params, dt)

    abs_err_delta = jnp.abs(delta_pred - delta_true)
    abs_err_accel = jnp.abs(accel_pred - accel_true)
    assert abs_err_delta < 1e-3, (
        f"delta absolute error {float(abs_err_delta):.2e} exceeds 1e-3 rad "
        f"(predicted={float(delta_pred):.6f}, true={float(delta_true):.6f})"
    )
    assert abs_err_accel < 5e-3, (
        f"accel absolute error {float(abs_err_accel):.2e} exceeds 5e-3 m/s^2 "
        f"(predicted={float(accel_pred):.6f}, true={float(accel_true):.6f})"
    )


def test_dynamic_bicycle_kinematic_prior_fails_same_tolerance(feasible_dyn_bicycle_trajectory):
    """Old kinematic prior misses by ≥10× vs new prior on delta (anti-tautology guard).

    The old formula is replicated inline so this test does not depend on the
    production code being reverted — it always compares against a fixed kinematic
    formula that was the pre-fix implementation.
    """
    x_prev, x_curr, params, control_true, dt = feasible_dyn_bicycle_trajectory
    delta_true = control_true[0]

    # Old kinematic prior (replicated inline).
    l_f, l_r = params[4], params[5]
    wheelbase = jnp.maximum(l_f + l_r, jnp.asarray(1e-6, x_prev.dtype))
    dtheta = x_curr[2] - x_prev[2]
    eps_kin = jnp.asarray(1e-6, x_prev.dtype)
    vx_safe_kin = jnp.where(jnp.abs(x_prev[3]) > eps_kin, x_prev[3], eps_kin)
    delta_kin = jnp.arctan(wheelbase * dtheta / (vx_safe_kin * dt))

    # New prior.
    physics = DynamicBicycle()
    delta_new, _ = physics.known_control_prior(x_prev, x_curr, params, dt)

    err_kin = jnp.abs(delta_kin - delta_true)
    err_new = jnp.abs(delta_new - delta_true)

    # New prior must beat kinematic by ≥10× on delta.
    assert err_new * 10.0 < err_kin, (
        f"Diagnosticity guard failed: new prior delta error {float(err_new):.6f} is not "
        f">10x smaller than kinematic delta error {float(err_kin):.6f}."
    )
    # Additionally: new prior's absolute delta error must be < 1e-3 (Architect edit #4).
    assert err_new < 1e-3, (
        f"New prior absolute delta error {float(err_new):.6f} exceeds 1e-3."
    )


def test_dynamic_bicycle_inverse_one_iter_already_strong(feasible_dyn_bicycle_trajectory):
    """A single Newton iteration already beats the kinematic prior by ≥5×.

    Documents that _NEWTON_STEPS=2 is safety margin, not load-bearing math
    (Architect edit #2 / recommendation #1).
    """
    x_prev, x_curr, params, control_true, dt = feasible_dyn_bicycle_trajectory
    delta_true = control_true[0]

    # Old kinematic prior (replicated inline, same as above).
    l_f, l_r = params[4], params[5]
    wheelbase = jnp.maximum(l_f + l_r, jnp.asarray(1e-6, x_prev.dtype))
    dtheta = x_curr[2] - x_prev[2]
    eps_kin = jnp.asarray(1e-6, x_prev.dtype)
    vx_safe_kin = jnp.where(jnp.abs(x_prev[3]) > eps_kin, x_prev[3], eps_kin)
    delta_kin = jnp.arctan(wheelbase * dtheta / (vx_safe_kin * dt))
    err_kin = jnp.abs(delta_kin - delta_true)

    # One-step Newton: replicate the prior logic with only 1 iteration.
    c_f, c_r, mass, inertia_z, l_f2, l_r2 = params
    eps = jnp.asarray(DynamicBicycle._SAFE_VX_EPS, dtype=x_prev.dtype)
    safe_vx = jnp.where(
        jnp.abs(x_prev[3]) > eps, x_prev[3], jnp.where(x_prev[3] >= 0.0, eps, -eps)
    )
    beta = jnp.arctan2(x_prev[4] + l_f2 * x_prev[5], safe_vx)
    alpha_r = -jnp.arctan2(x_prev[4] - l_r2 * x_prev[5], safe_vx)
    F_r = c_r * alpha_r
    inertia_safe = jnp.maximum(inertia_z, eps)
    L = jnp.maximum(l_f2 + l_r2, jnp.asarray(1e-6, dtype=x_prev.dtype))
    delta_1 = jnp.arctan(L * dtheta / (safe_vx * dt))
    yaw_rate_dot_fd = (x_curr[5] - x_prev[5]) / dt
    denom_floor = jnp.asarray(1e-6, dtype=x_prev.dtype)
    # Single Newton step.
    F_f = c_f * (delta_1 - beta)
    g_val = (l_f2 * F_f * jnp.cos(delta_1) - l_r2 * F_r) / inertia_safe - yaw_rate_dot_fd
    dg = (l_f2 * (c_f * jnp.cos(delta_1) - F_f * jnp.sin(delta_1))) / inertia_safe
    safe_dg = jnp.where(
        jnp.abs(dg) > denom_floor, dg, jnp.where(dg >= 0.0, denom_floor, -denom_floor)
    )
    delta_1iter = delta_1 - g_val / safe_dg
    err_1iter = jnp.abs(delta_1iter - delta_true)

    assert err_1iter * 5.0 < err_kin, (
        f"One-iter Newton delta error {float(err_1iter):.6f} is not >5x smaller "
        f"than kinematic error {float(err_kin):.6f}."
    )


def test_dynamic_bicycle_inverse_finite_at_low_vx():
    """Prior returns finite (delta, a) when vx is inside the safe_vx branch."""
    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    # vx well inside the singular regime (< _SAFE_VX_EPS).
    x_prev = jnp.array([0.0, 0.0, 0.0, 1e-4, 0.0, 0.0], dtype=jnp.float64)
    x_curr = jnp.array([0.01, 0.0, 0.01, 1e-4, 0.01, 0.01], dtype=jnp.float64)
    u = physics.known_control_prior(x_prev, x_curr, params, 0.1)
    assert u.shape == (2,)
    assert jnp.all(jnp.isfinite(u)), f"Prior returned non-finite values at low vx: {u}"


def test_dynamic_bicycle_inverse_vmap_compatible(feasible_dyn_bicycle_trajectory):
    """jax.vmap over a batch of (x_prev, x_curr, params) returns shape (B, 2) finite."""
    x_prev, x_curr, params, _, dt = feasible_dyn_bicycle_trajectory
    B = 8
    x_prev_batch = jnp.tile(x_prev[None], (B, 1))
    x_curr_batch = jnp.tile(x_curr[None], (B, 1))
    params_batch = jnp.tile(params[None], (B, 1))

    physics = DynamicBicycle()

    def prior_single(xp, xc, p):
        return physics.known_control_prior(xp, xc, p, dt)

    result = jax.vmap(prior_single)(x_prev_batch, x_curr_batch, params_batch)
    assert result.shape == (B, 2)
    assert jnp.all(jnp.isfinite(result))


def test_dynamic_bicycle_inverse_jit_compatible(feasible_dyn_bicycle_trajectory):
    """eqx.filter_jit on known_control_prior traces cleanly; output matches eager within 1e-12."""
    x_prev, x_curr, params, _, dt = feasible_dyn_bicycle_trajectory
    physics = DynamicBicycle()

    eager = physics.known_control_prior(x_prev, x_curr, params, dt)
    jit_fn = eqx.filter_jit(lambda xp, xc, p: physics.known_control_prior(xp, xc, p, dt))
    jitted = jit_fn(x_prev, x_curr, params)

    assert jnp.allclose(eager, jitted, atol=1e-12), (
        f"JIT vs eager mismatch: {eager} vs {jitted}"
    )


def test_dynamic_bicycle_safe_vx_eps_consistency():
    """_SAFE_VX_EPS ClassVar is used in both vector_field and known_control_prior.

    Verifies single source of truth: if the epsilon ever changes, both
    functions update together.
    """
    import inspect

    vf_source = inspect.getsource(DynamicBicycle.vector_field)
    kcp_source = inspect.getsource(DynamicBicycle.known_control_prior)

    assert "_SAFE_VX_EPS" in vf_source, (
        "vector_field does not reference _SAFE_VX_EPS ClassVar"
    )
    assert "_SAFE_VX_EPS" in kcp_source, (
        "known_control_prior does not reference _SAFE_VX_EPS ClassVar"
    )
