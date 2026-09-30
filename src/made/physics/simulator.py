"""Trajectory generation utilities for feasible-data simulation."""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from made.physics.base import ConstraintSet, PhysicsModel


def _sampling_bounds(
    constraints: ConstraintSet,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        constraints.state_min,
        constraints.state_max,
        constraints.control_min,
        constraints.control_max,
    )


def _uses_dynamic_bicycle_layout(physics: PhysicsModel) -> bool:
    return physics.state_dim == 6 and physics.control_dim == 2 and physics.param_dim == 6


def _sample_initial_state(
    physics: PhysicsModel,
    constraints: ConstraintSet,
    key: jax.Array,
    control_profile: str = "iid_uniform",
    min_speed: float | None = None,
) -> jax.Array:
    state_min, state_max, _, _ = _sampling_bounds(constraints)
    if _uses_dynamic_bicycle_layout(physics):
        # v0 floor: the pre-existing 2.0 clamp, raised to `min_speed` when a
        # feasibility floor is configured. Otherwise the over-generate-and-reject
        # loop in generate_trajectories wastes draws on trajectories that start
        # below the floor and get rejected outright. min_speed=None reproduces
        # the 2.0 clamp byte-for-byte.
        v0_floor = 2.0 if min_speed is None else jnp.maximum(2.0, min_speed)
        state_min = state_min.at[3].set(jnp.maximum(state_min[3], v0_floor))
        state_min = state_min.at[4].set(jnp.maximum(state_min[4], -1.0))
        state_max = state_max.at[4].set(jnp.minimum(state_max[4], 1.0))
        state_min = state_min.at[5].set(jnp.maximum(state_min[5], -0.3))
        state_max = state_max.at[5].set(jnp.minimum(state_max[5], 0.3))
        if control_profile == "smooth_ou":
            # Calmer entry speed, and near-zero lateral motion/yaw-rate — like a
            # vehicle entering the scene driving straight, matching inD-like
            # calm driving. Only narrows bounds (no extra random draws), so the
            # default "iid_uniform" path's key consumption is unaffected.
            state_max = state_max.at[3].set(jnp.minimum(state_max[3], 15.0))
            vy_bound = state_max[4]
            yaw_bound = state_max[5]
            state_min = state_min.at[4].set(-0.1 * vy_bound)
            state_max = state_max.at[4].set(0.1 * vy_bound)
            state_min = state_min.at[5].set(-0.1 * yaw_bound)
            state_max = state_max.at[5].set(0.1 * yaw_bound)
    return jax.random.uniform(key, state_min.shape, minval=state_min, maxval=state_max)



def _control_sampling_bounds(
    constraints: ConstraintSet,
    control_sample_min: tuple[float, ...] | None,
    control_sample_max: tuple[float, ...] | None,
) -> tuple[jax.Array, jax.Array]:
    """The box the control sampler draws from.

    **The dynamic-bicycle special case is DELETED.** Sampling uses the constraint set's own
    bounds, which for the dynamic bicycle are steering +-0.5 and acceleration +-3.0. The
    narrowing that used to be applied here -- steering to +-0.2 and acceleration to +-1.5 inside
    that same set -- is what made the generated data sit well inside its own feasible region by
    construction, which this settles.

    An earlier reading narrowed the SCORED box to
    match the sampler; that reading is withdrawn: the scorer stays at +-0.5 and +-3.0 and the
    sampler widens to meet it. Nothing here touches `dynamic_bicycle_constraints()`.

    `control_sample_min`/`max`, when given, still ARE the box -- that is how an arm samples from
    something other than the constraint set, and it is what the margin control uses to reproduce
    the old narrowing for comparison.

    CONSEQUENCE, stated because it is not recoverable by reading the code later: a dataset
    generated before this change is no longer reproducible from its own config, because its
    config never recorded the narrowing -- the narrowing was hardcoded here. The datasets on
    disk are untouched; only regeneration from config would differ.

    Written once and called from BOTH samplers. The narrowing was duplicated in
    `_sample_control_sequence` and `_sample_control_sequence_ou`, so removing it here once
    removes it from both.
    """
    _, _, control_min, control_max = _sampling_bounds(constraints)
    if control_sample_min is not None and control_sample_max is not None:
        lo = jnp.asarray(control_sample_min, dtype=control_min.dtype)
        hi = jnp.asarray(control_sample_max, dtype=control_max.dtype)
        if lo.shape != control_min.shape:
            raise ValueError(
                f"control_sample_min has shape {lo.shape} but this system has "
                f"{control_min.shape[0]} control dimensions"
            )
        return lo, hi
    return control_min, control_max


def _sample_control_sequence(
    constraints: ConstraintSet,
    length: int,
    key: jax.Array,
    control_sample_min: tuple[float, ...] | None = None,
    control_sample_max: tuple[float, ...] | None = None,
) -> jax.Array:
    control_min, control_max = _control_sampling_bounds(
        constraints, control_sample_min, control_sample_max)
    shape = (length, control_min.shape[0])
    return jax.random.uniform(key, shape, minval=control_min, maxval=control_max)


def _sample_control_sequence_ou(
    constraints: ConstraintSet,
    length: int,
    dt: float,
    tau: float,
    key: jax.Array,
    control_sample_min: tuple[float, ...] | None = None,
    control_sample_max: tuple[float, ...] | None = None,
) -> jax.Array:
    """Smooth, temporally-correlated control profile via a discretised OU process.

    Per control dim: x_{t+1} = x_t + (dt/tau)(mu - x_t) + sigma*sqrt(dt)*eps_t,
    mu = 0, eps_t ~ N(0, 1), clipped to the sensible-control box after every
    step. The stationary std is set to half the box half-width so that ~2 std
    spans the box (few samples need clipping), which pins sigma via the
    continuous-time OU relation stationary_var = sigma^2 * tau / 2.
    """
    control_min, control_max = _control_sampling_bounds(
        constraints, control_sample_min, control_sample_max)

    control_dim = control_min.shape[0]
    if length <= 0:
        return jnp.zeros((0, control_dim), dtype=control_min.dtype)

    box_half_width = jnp.minimum(-control_min, control_max)
    stationary_std = box_half_width / 2.0
    sigma = stationary_std * jnp.sqrt(2.0 / tau)

    init_key, noise_key = jax.random.split(key)
    initial_control = jax.random.normal(init_key, (control_dim,)) * (stationary_std / 2.0)
    initial_control = jnp.clip(initial_control, control_min, control_max)
    noise = jax.random.normal(noise_key, (length - 1, control_dim))

    def _ou_step(x_t: jax.Array, eps_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_next = x_t + (dt / tau) * (0.0 - x_t) + sigma * jnp.sqrt(dt) * eps_t
        x_next = jnp.clip(x_next, control_min, control_max)
        return x_next, x_next

    _, remaining_controls = jax.lax.scan(_ou_step, initial_control, noise)
    return jnp.concatenate([initial_control[None, :], remaining_controls], axis=0)


def _sample_controls(
    physics: PhysicsModel,
    constraints: ConstraintSet,
    length: int,
    dt: float,
    key: jax.Array,
    control_profile: str,
    control_tau: float,
    control_sample_min: tuple[float, ...] | None = None,
    control_sample_max: tuple[float, ...] | None = None,
) -> jax.Array:
    """Dispatch to the configured control sampler.

    The "iid_uniform" branch calls `_sample_control_sequence` with `key`
    unmodified — same call, same key, as the pre-existing code path — so the
    default profile's key consumption (and therefore its generated data) is
    byte-identical to before this dispatcher existed. The "smooth_ou" branch
    splits its own keys internally and never touches the default branch.
    """
    if control_profile == "iid_uniform":
        return _sample_control_sequence(
            constraints, length, key, control_sample_min, control_sample_max)
    if control_profile == "smooth_ou":
        return _sample_control_sequence_ou(
            constraints, length, dt, control_tau, key,
            control_sample_min, control_sample_max)
    raise ValueError(f"Unsupported control_profile '{control_profile}'.")


def _integrate_step(
    physics: PhysicsModel,
    x_prev: jax.Array,
    control: jax.Array,
    params: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    term = diffrax.ODETerm(
        lambda t, y, args: physics.vector_field(y, args["control"], args["params"], t)
    )
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=0.0,
        t1=dt,
        dt0=dt,
        y0=x_prev,
        args={"control": control, "params": params},
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        saveat=diffrax.SaveAt(t1=True),
        throw=False,
    )
    return solution.ys[0], solution.result == diffrax.RESULTS.successful


def _generate_single_trajectory(
    physics: PhysicsModel,
    constraints: ConstraintSet,
    trajectory_length: int,
    dt: float,
    key: jax.Array,
    true_params: jax.Array,
    control_profile: str = "iid_uniform",
    control_tau: float = 1.5,
    min_speed: float | None = None,
    control_sample_min: tuple[float, ...] | None = None,
    control_sample_max: tuple[float, ...] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    init_key, control_key = jax.random.split(key)
    initial_state = _sample_initial_state(
        physics, constraints, init_key, control_profile, min_speed
    )
    controls = _sample_controls(
        physics,
        constraints,
        trajectory_length - 1,
        dt,
        control_key,
        control_profile,
        control_tau,
        control_sample_min,
        control_sample_max,
    )

    def _scan_step(carry: tuple[jax.Array, jax.Array], control: jax.Array):
        state, previous_ok = carry
        next_state, step_ok = _integrate_step(physics, state, control, true_params, dt)
        current_ok = previous_ok & step_ok & jnp.all(jnp.isfinite(next_state))
        return (next_state, current_ok), (next_state, current_ok)

    (_, solve_ok), (future_states, _) = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(True)),
        controls,
    )
    states = jnp.concatenate([initial_state[None, :], future_states], axis=0)
    return states, controls, solve_ok


def _trajectory_feasible(
    constraints: ConstraintSet,
    states: jax.Array,
    controls: jax.Array,
) -> jax.Array:
    aligned_controls = jnp.concatenate([controls, controls[-1:]], axis=0)
    violations = jax.vmap(constraints.violation)(states, aligned_controls)
    return jnp.all(violations <= 0.0)


def generate_trajectories(
    physics: PhysicsModel,
    constraints: ConstraintSet,
    num_trajectories: int,
    trajectory_length: int,
    dt: float,
    key: jax.Array,
    true_params: jax.Array,
    control_profile: str = "iid_uniform",
    control_tau: float = 1.5,
    min_speed: float | None = None,
    control_sample_min: tuple[float, ...] | None = None,
    control_sample_max: tuple[float, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generate feasible trajectories by over-generating and filtering.

    The over-generate-and-filter behaviour is DELIBERATE and is kept
    exactly as it is. `control_sample_min`/`max` change only the box controls are drawn from,
    never the filtering.

    `min_speed`, when set, adds a feasibility criterion: a trajectory is kept
    only if its longitudinal speed (state index 3) never drops below the
    floor, at every timestep including the initial state. Index 3 is only
    meaningful as speed for the dynamic-bicycle 6D layout (the tire-slip terms
    are singular at standstill there), so `min_speed` is rejected outright for
    any other system rather than silently ignored. `min_speed=None` (default)
    disables the criterion and reproduces legacy generation byte-for-byte.
    """
    if min_speed is not None and not _uses_dynamic_bicycle_layout(physics):
        raise ValueError(
            "min_speed is only meaningful for the dynamic-bicycle 6D layout "
            "(state index 3 is longitudinal speed only there); got "
            f"min_speed={min_speed} for a physics model without that layout."
        )
    collected_states: list[jax.Array] = []
    collected_controls: list[jax.Array] = []
    remaining = num_trajectories
    current_key = key

    for _ in range(10):
        if remaining <= 0:
            break
        current_key, subkey = jax.random.split(current_key)
        num_candidates = max(remaining * 3, remaining)
        candidate_keys = jax.random.split(subkey, num_candidates)
        states, controls, solve_ok = jax.vmap(
            lambda single_key: _generate_single_trajectory(
                physics,
                constraints,
                trajectory_length,
                dt,
                single_key,
                true_params,
                control_profile,
                control_tau,
                min_speed,
                control_sample_min,
                control_sample_max,
            )
        )(candidate_keys)
        feasible = jax.vmap(
            lambda x, u: _trajectory_feasible(constraints, x, u)
        )(states, controls)
        feasible = feasible & solve_ok & jnp.all(jnp.isfinite(states), axis=(1, 2))
        if min_speed is not None:
            feasible = feasible & jnp.all(states[:, :, 3] >= min_speed, axis=1)
        feasible_indices = np.flatnonzero(np.asarray(feasible))
        if feasible_indices.size == 0:
            continue
        take_count = min(int(feasible_indices.size), remaining)
        selected = feasible_indices[:take_count]
        collected_states.append(states[selected])
        collected_controls.append(controls[selected])
        remaining -= take_count

    if remaining > 0:
        raise RuntimeError(
            "Unable to generate enough feasible trajectories under the current constraints."
        )

    return jnp.concatenate(collected_states, axis=0), jnp.concatenate(collected_controls, axis=0)
