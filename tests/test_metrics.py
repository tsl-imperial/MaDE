import math

import diffrax
import jax
import jax.numpy as jnp

from made.evaluation import (
    EmpiricalEnvelope,
    METRIC_VERSION,
    add_observation_noise,
    ade,
    empirical_envelope_violation_magnitude,
    empirical_envelope_violation_rate,
    estimate_empirical_envelope,
    estimate_gt_reference_residual,
    fde,
    fidelity,
    gt_normalised_dynamics_residual,
    heading_jerk,
    inequality_violation_magnitude,
    inequality_violation_rate,
    jerk,
    kinematic_bicycle_inverse_controls,
    perturb_trajectories,
)
from made.evaluation.metrics import (
    dynamics_violation_known,
    dynamics_violation_true,
    position_ade,
    position_fde,
)
from made.physics import DynamicBicycle, KinematicBicycle, double_integrator_constraints
from made.physics import KinematicBicycleAsDynamicState
from made.utils import DataConfig


def test_inequality_metrics_feasible():
    constraints = double_integrator_constraints()
    x = jnp.zeros((10, 4))
    u = jnp.zeros((10, 2))
    assert inequality_violation_rate(x, u, constraints) == 0.0
    assert inequality_violation_magnitude(x, u, constraints) == 0.0


def test_fidelity_identity():
    x = jnp.zeros((10, 4))
    assert fidelity(x, x) == 0.0


def test_perturbation_deterministic():
    states = jnp.zeros((2, 4, 4))
    config = DataConfig(perturbation_scale=0.1)
    key = jax.random.key(0)
    first = perturb_trajectories(states, config, key)
    second = perturb_trajectories(states, config, key)
    assert jnp.allclose(first, second)
    noisy = add_observation_noise(states, 0.1, key)
    assert noisy.shape == states.shape


def test_dynamics_violation_known_finite_under_extreme_steering():
    """Heun+ConstantStep must return a finite value even for delta near pi/2.

    Previously Tsit5+PIDController would contract dt indefinitely on this
    stiff input (tan(delta) -> inf) and exhaust max_steps, raising an error.
    """
    physics = KinematicBicycleAsDynamicState()
    params = jnp.array([2.7])
    dt = 0.1
    # 6D state: (x, y, theta, v, vy, yaw_rate) — vy/yaw_rate ignored by kinematic vf
    x0 = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    delta_extreme = jnp.pi / 2 - 1e-3
    u = jnp.array([[delta_extreme, 0.0]])  # shape (1, 2): one control for one step
    # Integrate x1 directly with Heun so x has shape (2, 6)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(
            lambda t, y, args: physics.vector_field(y, args["control"], params, t)
        ),
        diffrax.Heun(),
        t0=0.0,
        t1=dt,
        dt0=dt,
        y0=x0,
        args={"control": u[0]},
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=16,
    )
    x1 = sol.ys[0]
    x = jnp.stack([x0, x1])  # (2, 6)
    result = dynamics_violation_known(x, u, physics, params, dt)
    assert jnp.isfinite(result), f"Expected finite result, got {result}"


def _integrate_kinbicycle_trajectory(n_steps: int, dt: float) -> tuple[jax.Array, jax.Array]:
    """Build a short kinematic-bicycle trajectory using Heun+ConstantStep."""
    physics = KinematicBicycle()
    params = jnp.array([2.7])
    # Moderate in-distribution conditions: moderate speed, small steering, small accel
    x0 = jnp.array([0.0, 0.0, 0.0, 8.0])
    delta = 0.1  # rad
    accel = 0.5  # m/s^2
    u_step = jnp.array([delta, accel])

    states = [x0]
    x = x0
    for _ in range(n_steps):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(
                lambda t, y, args: physics.vector_field(y, args["control"], params, t)
            ),
            diffrax.Heun(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=x,
            args={"control": u_step},
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=16,
        )
        x = sol.ys[0]
        states.append(x)

    x_traj = jnp.stack(states)  # (n_steps+1, 4)
    u_traj = jnp.tile(u_step, (n_steps, 1))  # (n_steps, 2)
    return x_traj, u_traj, params


def test_dynamics_violation_known_near_zero_on_simulator_trajectory():
    """When stencil matches stencil, the known-physics metric must be near machine zero."""
    dt = 0.1
    x_traj, u_traj, params = _integrate_kinbicycle_trajectory(n_steps=5, dt=dt)
    physics = KinematicBicycle()
    result = dynamics_violation_known(x_traj, u_traj, physics, params, dt)
    assert float(result) < 1e-10, f"Expected near-zero dynamics violation, got {result}"


def test_dynamics_violation_true_uses_heun_stencil():
    """dynamics_violation_true inherits the Heun fix — near zero on Heun-generated DynBicycle traj."""
    physics = DynamicBicycle()
    params = jnp.array([19000.0, 20000.0, 1500.0, 3000.0, 1.2, 1.5])
    dt = 0.1
    # Moderate in-distribution state: 10 m/s forward, small lateral motion
    x0 = jnp.array([0.0, 0.0, 0.0, 10.0, 0.1, 0.05])
    delta = 0.05
    accel = 0.2
    u_step = jnp.array([delta, accel])

    states = [x0]
    x = x0
    for _ in range(4):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(
                lambda t, y, args: physics.vector_field(y, args["control"], params, t)
            ),
            diffrax.Heun(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=x,
            args={"control": u_step},
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=16,
        )
        x = sol.ys[0]
        states.append(x)

    x_traj = jnp.stack(states)  # (5, 6)
    u_traj = jnp.tile(u_step, (4, 1))  # (4, 2)
    result = dynamics_violation_true(x_traj, u_traj, physics, params, dt)
    assert float(result) < 1e-10, f"Expected near-zero dynamics violation, got {result}"


# ---------------------------------------------------------------------------
# Real-data (E2 / E3) metric helpers — ralplan-real-data-metrics-v1.
# ---------------------------------------------------------------------------


def test_metric_version_is_real_data_v3_gaussian():
    assert METRIC_VERSION == "real-data-v3-gaussian"


def test_estimate_empirical_envelope_quantiles():
    """1st/99th percentiles of a uniform [0, 1] distribution lie in [0.005, 0.995]."""
    rng = jax.random.key(0)
    states = jax.random.uniform(rng, shape=(500, 1, 4), dtype=jnp.float64)
    rng2 = jax.random.key(1)
    controls = jax.random.uniform(rng2, shape=(500, 0, 2), dtype=jnp.float64)
    # Add at least one transition per trajectory by giving 2 timesteps of states + 1 of controls
    rng3 = jax.random.key(2)
    states = jax.random.uniform(rng3, shape=(200, 5, 4), dtype=jnp.float64)
    rng4 = jax.random.key(3)
    controls = jax.random.uniform(rng4, shape=(200, 4, 2), dtype=jnp.float64)
    env = estimate_empirical_envelope(states, controls)
    for d in range(4):
        assert env.state_min[d] >= 0.0
        assert env.state_min[d] <= 0.05
        assert env.state_max[d] >= 0.95
        assert env.state_max[d] <= 1.0
    for d in range(2):
        assert 0.0 <= env.control_min[d] <= 0.05
        assert 0.95 <= env.control_max[d] <= 1.0


def test_estimate_empirical_envelope_rounding():
    """Lower bound rounds DOWN; upper bound rounds UP per increment."""
    states = jnp.zeros((10, 3, 4), dtype=jnp.float64)
    states = states.at[:, :, 0].set(jnp.linspace(-0.37, 0.42, 30).reshape(10, 3))
    states = states.at[:, :, 3].set(jnp.linspace(0.13, 1.27, 30).reshape(10, 3))
    controls = jnp.zeros((10, 2, 2), dtype=jnp.float64)
    env = estimate_empirical_envelope(
        states,
        controls,
        state_rounding=(0.1, 0.1, 0.01, 0.1),
        control_rounding=(0.01, 0.01),
    )
    # The 1st-quantile of states[..., 0] is negative; floor(neg/0.1)*0.1 stays
    # at or below the original quantile, e.g. -0.37 → floor(-3.7)=-4 → -0.4.
    assert float(env.state_min[0]) <= -0.36
    # 99th quantile of states[..., 0] is near 0.4; ceil(0.4/0.1)*0.1 = 0.4.
    assert float(env.state_max[0]) >= 0.39
    # state_min[3] should be a multiple of 0.1 and ≤ original 1st quantile.
    assert math.isclose(
        float(env.state_min[3]) / 0.1,
        round(float(env.state_min[3]) / 0.1),
        abs_tol=1e-9,
    )
    assert math.isclose(
        float(env.state_max[3]) / 0.1,
        round(float(env.state_max[3]) / 0.1),
        abs_tol=1e-9,
    )


def test_estimate_empirical_envelope_respects_lengths():
    """Padded entries past lengths must be excluded from the envelope."""
    real = jnp.array([
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]],
    ])
    pad_value = 999.0
    padded = jnp.concatenate(
        [real, jnp.full((2, 3, 4), pad_value, dtype=real.dtype)], axis=1
    )  # (2, 5, 4) with last 3 timesteps padding
    real_u = jnp.zeros((2, 1, 2), dtype=real.dtype)
    padded_u = jnp.concatenate(
        [real_u, jnp.full((2, 3, 2), pad_value, dtype=real.dtype)], axis=1
    )  # (2, 4, 2)
    lengths = jnp.array([2, 2], dtype=jnp.int32)
    env = estimate_empirical_envelope(padded, padded_u, lengths=lengths)
    for d in range(4):
        assert float(env.state_max[d]) <= 1.001, (
            f"padding leaked into envelope: state_max[{d}] = {env.state_max[d]}"
        )
    for d in range(2):
        assert float(env.control_max[d]) <= 0.001


def _simple_envelope() -> EmpiricalEnvelope:
    return EmpiricalEnvelope(
        state_min=jnp.array([-1.0, -1.0, -1.0, -1.0]),
        state_max=jnp.array([1.0, 1.0, 1.0, 1.0]),
        control_min=jnp.array([-0.5, -0.5]),
        control_max=jnp.array([0.5, 0.5]),
    )


def test_envelope_violation_rate_zero_inside():
    env = _simple_envelope()
    x = jnp.zeros((5, 4))
    u = jnp.zeros((5, 2))
    assert float(empirical_envelope_violation_rate(x, u, env)) == 0.0


def test_envelope_violation_rate_one_outside():
    env = _simple_envelope()
    x = jnp.full((5, 4), 5.0)  # every step violates state_max in dim 0
    u = jnp.zeros((5, 2))
    assert float(empirical_envelope_violation_rate(x, u, env)) == 1.0


def test_envelope_violation_rate_u_none_skips_control_box():
    """When u is None, only state-side bounds are evaluated."""
    env = _simple_envelope()
    x = jnp.zeros((5, 4))  # state-feasible
    rate = float(empirical_envelope_violation_rate(x, None, env))
    assert rate == 0.0
    # If u were checked, the implicit zero-u would be inside [-0.5, 0.5] anyway,
    # so this test validates the call signature accepts u=None and returns
    # finite/feasible rates without raising.


def test_envelope_violation_magnitude_l2():
    env = _simple_envelope()
    # Single-trajectory: x[t] = [0, 0, 2.0, 0]. state_max[2]=1.0 → violation 1.0
    # in dim 2 (non-positional) at every step. x,y (dims 0,1) are stripped to ±inf
    # by envelope_constraint, so only dim-2 contributes.
    x = jnp.tile(jnp.array([0.0, 0.0, 2.0, 0.0]), (4, 1))
    u = jnp.zeros((4, 2))
    mag = float(empirical_envelope_violation_magnitude(x, u, env))
    assert math.isclose(mag, 1.0, abs_tol=1e-6)


def test_ade_alias_matches_fidelity():
    x = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    y = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert float(ade(x, y)) == float(fidelity(x, y))


def test_fde_final_step_only():
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]])  # final L2 distance = 5
    y = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    assert math.isclose(float(fde(x, y)), 5.0, abs_tol=1e-9)


def _non_heun_circular_trajectory(dt: float, n: int = 8) -> jax.Array:
    """Build a circular trajectory whose KB residual is non-trivially positive.

    The points lie on a perfect circle of radius 5 with angular velocity ω,
    so headings/speeds are analytically consistent. KB+L=2.7 Heun stencil
    does NOT reproduce this trajectory (the curvature implies a different
    wheelbase), so ``dynamics_violation_known`` reports a positive residual.
    """
    t = jnp.arange(n, dtype=jnp.float64)
    radius = 5.0
    omega = 0.5
    x = radius * jnp.cos(omega * dt * t)
    y = radius * jnp.sin(omega * dt * t)
    heading = omega * dt * t + jnp.pi / 2
    speed = jnp.full_like(t, radius * omega)
    return jnp.stack([x, y, heading, speed], axis=-1)


def test_gt_normalised_residual_zero_when_equal():
    """When method residual == GT reference residual, output is 0."""
    physics = KinematicBicycle()
    params = jnp.array([2.7])
    dt = 0.2
    states = _non_heun_circular_trajectory(dt)
    controls, _ = kinematic_bicycle_inverse_controls(states, dt, wheelbase=2.7)
    method = float(dynamics_violation_known(states, controls, physics, params, dt))
    assert method > 1e-6, "needs a non-trivial method residual for this test"
    out = float(
        gt_normalised_dynamics_residual(
            states, controls, physics, params, dt,
            gt_reference_residual=method,
        )
    )
    assert abs(out) < 1e-9, f"expected 0, got {out}"


def test_gt_normalised_residual_eps_floor():
    """GT reference residual = 0 must NOT produce inf/nan."""
    physics = KinematicBicycle()
    params = jnp.array([2.7])
    dt = 0.1
    x_traj, u_traj, _ = _integrate_kinbicycle_trajectory(n_steps=3, dt=dt)
    out = float(
        gt_normalised_dynamics_residual(
            x_traj, u_traj, physics, params, dt,
            gt_reference_residual=0.0,
        )
    )
    assert math.isfinite(out), f"expected finite, got {out}"


def test_gt_normalised_residual_negative_overprojection():
    """method = 0.5 × gt → output = 0.5 / 1 - 1 = -0.5."""
    physics = KinematicBicycle()
    params = jnp.array([2.7])
    dt = 0.2
    states = _non_heun_circular_trajectory(dt)
    controls, _ = kinematic_bicycle_inverse_controls(states, dt, wheelbase=2.7)
    method = float(dynamics_violation_known(states, controls, physics, params, dt))
    assert method > 1e-6
    out = float(
        gt_normalised_dynamics_residual(
            states, controls, physics, params, dt,
            gt_reference_residual=2.0 * method,
        )
    )
    assert math.isclose(out, -0.5, abs_tol=1e-3), out


def test_gt_normalised_residual_nontrivial_on_non_heun_trajectory():
    """Non-Heun-stencil trajectory must yield a non-trivial ratio."""
    # Construct a trajectory that does NOT obey the KB Heun stencil exactly:
    # constant acceleration with non-zero δ but states picked from a circular
    # arc that has a different curvature than KB+L=2.7 would prescribe.
    dt = 0.2
    t = jnp.arange(8, dtype=jnp.float64)
    radius = 5.0
    omega = 0.5
    x = radius * jnp.cos(omega * dt * t)
    y = radius * jnp.sin(omega * dt * t)
    heading = omega * dt * t + jnp.pi / 2
    speed = jnp.full_like(t, radius * omega)
    states = jnp.stack([x, y, heading, speed], axis=-1)

    # Recover controls under wheelbase = 2.7 (the canonical fixed L).
    controls, _ = kinematic_bicycle_inverse_controls(states, dt, wheelbase=2.7)
    physics = KinematicBicycle()
    method = float(
        dynamics_violation_known(states, controls, physics, jnp.array([2.7]), dt)
    )
    # GT reference is what the same recovery + dynamics_violation_known would
    # report on the same trajectory. The numerator and denominator are both
    # tiny here, so the ratio + eps floor must remain finite.
    ref = max(method * 0.9, 1e-12)
    out = float(
        gt_normalised_dynamics_residual(
            states, controls, physics, jnp.array([2.7]), dt,
            gt_reference_residual=ref,
        )
    )
    assert math.isfinite(out)
    # ratio is method / ref = 1/0.9 ≈ 1.111 → out ≈ 0.111 > 1e-3.
    assert abs(out) > 1e-3, f"expected non-trivial residual ratio, got {out}"


def test_estimate_gt_reference_residual_heun_perfect_near_zero():
    """Heun-generated trajectories give a near-zero reference residual."""
    dt = 0.1
    x_traj, u_traj, params = _integrate_kinbicycle_trajectory(n_steps=5, dt=dt)
    physics = KinematicBicycle()
    ref = estimate_gt_reference_residual(
        x_traj[None],
        u_traj[None],
        physics,
        params,
        dt,
        lengths=jnp.array([x_traj.shape[0]], dtype=jnp.int32),
    )
    assert ref < 1e-10, f"expected near-zero reference residual, got {ref}"


def test_jerk_zero_for_constant_velocity():
    """Constant-velocity straight line has third-difference zero."""
    dt = 0.1
    n = 6
    t = jnp.arange(n, dtype=jnp.float64)
    states = jnp.stack(
        [t * 1.0, jnp.zeros_like(t), jnp.zeros_like(t), jnp.ones_like(t)],
        axis=-1,
    )
    j = float(jerk(states, dt))
    assert math.isclose(j, 0.0, abs_tol=1e-10), j


def test_jerk_short_trajectory():
    """Length-3 trajectory returns 0.0 (third diff requires 4 states)."""
    dt = 0.1
    states = jnp.zeros((3, 4))
    assert float(jerk(states, dt)) == 0.0


def test_heading_jerk_unwraps():
    """A heading sequence with a 2π wrap returns a small jerk after unwrap."""
    dt = 0.1
    # Smooth ramp through ±π that wraps in the principal-value representation.
    # Linear ramp Δθ = 0.2 per step, sampled at modulo 2π.
    n = 8
    underlying = 2.5 + 0.2 * jnp.arange(n, dtype=jnp.float64)
    raw = jnp.mod(underlying + jnp.pi, 2 * jnp.pi) - jnp.pi  # principal value
    states = jnp.stack(
        [jnp.zeros_like(raw), jnp.zeros_like(raw), raw, jnp.ones_like(raw)],
        axis=-1,
    )
    hj = float(heading_jerk(states, dt))
    assert math.isfinite(hj)
    # Without unwrap the third difference at the wrap is O(2π / dt^3) ≈ 6.3e3.
    # With unwrap, the underlying function is linear so higher-order diffs
    # collapse to machine zero up to fp roundoff.
    assert hj < 1.0, f"unwrap appears broken: heading_jerk={hj}"


def test_heading_jerk_short_trajectory():
    dt = 0.1
    states = jnp.zeros((3, 4))
    assert float(heading_jerk(states, dt)) == 0.0


def test_kb_inverse_controls_round_trip():
    """Inverse controls recover the ground-truth (δ, a) under Heun."""
    dt = 0.1
    x_traj, u_traj, params = _integrate_kinbicycle_trajectory(n_steps=5, dt=dt)
    controls, aux = kinematic_bicycle_inverse_controls(
        x_traj, dt, wheelbase=float(params[0])
    )
    err = float(jnp.max(jnp.abs(controls - u_traj)))
    assert err < 1e-9, f"round-trip error {err} exceeds 1e-9"
    assert aux["stationary_frame_count"] == 0


def test_ind_physical_and_envelope_strip_xy():
    """Regression guard for 2026-05-14 fix: x,y must be ±inf on inD inequality path."""
    from made.evaluation.metrics import envelope_constraint, EmpiricalEnvelope
    from made.physics import inD_physical_constraints

    phys = inD_physical_constraints()
    xy = jnp.array([0, 1])
    assert bool(jnp.all(jnp.isinf(phys.state_min[xy])))
    assert bool(jnp.all(jnp.isinf(phys.state_max[xy])))
    assert bool(jnp.all(phys.state_min[xy] < 0))
    assert bool(jnp.all(phys.state_max[xy] > 0))

    env = EmpiricalEnvelope(
        state_min=jnp.array([0., 0., -3.14, 0.]),
        state_max=jnp.array([100., 100., 3.14, 22.]),
        control_min=jnp.array([-0.5, -8.]),
        control_max=jnp.array([0.5, 4.]),
    )
    env_box = envelope_constraint(env)
    assert bool(jnp.all(jnp.isinf(env_box.state_min[xy])))
    assert bool(jnp.all(jnp.isinf(env_box.state_max[xy])))


def test_kb_inverse_controls_stationary_no_pi_over_two_spike():
    """A stationary segment must NOT produce ±π/2 deltas."""
    dt = 0.1
    # Build a trajectory that is stationary (v=0) for several frames with
    # tiny heading noise — would otherwise yield arctan(±large)→±π/2.
    n = 6
    states = jnp.zeros((n, 4), dtype=jnp.float64)
    headings = jnp.array([0.0, 0.01, -0.01, 0.005, -0.005, 0.0])
    states = states.at[:, 2].set(headings)
    # speed stays at 0 → v_avg = 0 → stationary branch should kick in.
    controls, aux = kinematic_bicycle_inverse_controls(states, dt, wheelbase=2.7)
    deltas = controls[:, 0]
    assert float(jnp.max(jnp.abs(deltas))) < 1.0, (
        f"stationary segment produced large δ: {deltas}"
    )
    assert aux["stationary_frame_count"] == n - 1

def test_position_ade_equals_norm_over_xy():
    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (3, 7, 4))
    y = jax.random.normal(ky, (3, 7, 4))
    expected = jnp.mean(jnp.linalg.norm(x[..., :2] - y[..., :2], axis=-1))
    assert math.isclose(float(position_ade(x, y)), float(expected), rel_tol=1e-12)

    expected_single = jnp.mean(jnp.linalg.norm(x[0][..., :2] - y[0][..., :2], axis=-1))
    assert math.isclose(float(position_ade(x[0], y[0])), float(expected_single), rel_tol=1e-12)


def test_position_fde_final_step_xy_only():
    x = jnp.array([[0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 100.0, -50.0]])
    y = jnp.zeros((2, 4))
    assert math.isclose(float(position_fde(x, y)), 5.0, abs_tol=1e-9)
    assert float(fde(x, y)) > 100.0


def test_position_metrics_ignore_heading_and_speed():
    key = jax.random.key(1)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (3, 7, 4))
    y = jax.random.normal(ky, (3, 7, 4))
    x_offset = x.at[..., 2:].add(100.0)
    assert position_ade(x, y) == position_ade(x_offset, y)
    assert position_fde(x, y) == position_fde(x_offset, y)


def test_full_state_ade_position_ade_distinction():
    key = jax.random.key(2)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (3, 7, 4))
    y = jax.random.normal(ky, (3, 7, 4))
    assert ade(x, y) == fidelity(x, y)
    assert ade(x, y) > position_ade(x, y)
