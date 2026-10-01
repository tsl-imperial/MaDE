# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

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
    """Return the constraint set's sampling bounds.

    Args:
        constraints: Constraint set exposing state and control bounds.

    Returns:
        Tuple ``(state_min, state_max, control_min, control_max)``.
    """
    return (
        constraints.state_min,
        constraints.state_max,
        constraints.control_min,
        constraints.control_max,
    )


def _uses_dynamic_bicycle_layout(physics: PhysicsModel) -> bool:
    """Whether ``physics`` has the dynamic-bicycle 6D state, 2D control, 6 parameter layout.

    Args:
        physics: Physics model to inspect.

    Returns:
        True when the layout matches.
    """
    return physics.state_dim == 6 and physics.control_dim == 2 and physics.param_dim == 6


def _sample_initial_state(
    physics: PhysicsModel,
    constraints: ConstraintSet,
    key: jax.Array,
    control_profile: str = "iid_uniform",
    min_speed: float | None = None,
) -> jax.Array:
    """Draw an initial state uniformly from the constraint box.

    For the dynamic-bicycle layout the box is narrowed so trajectories start in a calm regime.

    Args:
        physics: Physics model.
        constraints: Constraint set providing the sampling box.
        key: PRNG key.
        control_profile: ``"iid_uniform"`` or ``"smooth_ou"``.
        min_speed: Optional speed floor for the initial longitudinal speed.

    Returns:
        Initial state vector.
    """
    state_min, state_max, _, _ = _sampling_bounds(constraints)
    if _uses_dynamic_bicycle_layout(physics):
        # v0 floor: pre-existing 2.0 clamp, raised to `min_speed` when set, else the
        # over-generate-and-reject loop wastes draws on trajectories that start below the
        # floor and get rejected. min_speed=None reproduces the 2.0 clamp byte-for-byte.
        v0_floor = 2.0 if min_speed is None else jnp.maximum(2.0, min_speed)
        state_min = state_min.at[3].set(jnp.maximum(state_min[3], v0_floor))
        state_min = state_min.at[4].set(jnp.maximum(state_min[4], -1.0))
        state_max = state_max.at[4].set(jnp.minimum(state_max[4], 1.0))
        state_min = state_min.at[5].set(jnp.maximum(state_min[5], -0.3))
        state_max = state_max.at[5].set(jnp.minimum(state_max[5], 0.3))
        if control_profile == "smooth_ou":
            # Calmer entry speed, near-zero lateral motion/yaw-rate: like a vehicle entering
            # driving straight, matching inD-like calm driving. Only narrows bounds (no extra
            # draws), so "iid_uniform"'s key consumption is unaffected.
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

    Sampling uses the constraint set's own bounds (dynamic bicycle: steering +-0.5,
    acceleration +-3.0) -- no dynamic-bicycle narrowing to +-0.2/+-1.5 here.
    `dynamic_bicycle_constraints()` (the scorer) is untouched.

    `control_sample_min`/`max`, when given, ARE the box -- used to sample from something
    other than the constraint set, e.g. to reproduce the old narrowing for comparison.

    Consequence: a dataset generated before this change is not reproducible from its own
    config, since the config never recorded the narrowing (it was hardcoded here). Datasets
    on disk are untouched; only regeneration from config would differ.

    Called from both `_sample_control_sequence` and `_sample_control_sequence_ou`.

    Args:
        constraints: Constraint set providing the default box.
        control_sample_min: Optional lower bound overriding the constraint set.
        control_sample_max: Optional upper bound overriding the constraint set.

    Returns:
        Tuple ``(lower, upper)``.

    Raises:
        ValueError: If ``control_sample_min`` has the wrong number of dimensions.
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
    """Sample i.i.d. uniform controls from the sampling box.

    Args:
        constraints: Constraint set providing the default box.
        length: Number of control steps.
        key: PRNG key.
        control_sample_min: Optional lower bound overriding the constraint set.
        control_sample_max: Optional upper bound overriding the constraint set.

    Returns:
        Controls of shape ``(length, control_dim)``.
    """
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

    Per control dim: x_{t+1} = x_t + (dt/tau)(mu - x_t) + sigma*sqrt(dt)*eps_t, mu=0,
    eps_t ~ N(0, 1), clipped to the box after every step. Stationary std is set to half the
    box half-width so ~2 std spans the box (few clips), pinning sigma via the continuous-time
    OU relation stationary_var = sigma^2 * tau / 2.

    Args:
        constraints: Constraint set providing the default box.
        length: Number of control steps.
        dt: Step length.
        tau: OU time constant.
        key: PRNG key.
        control_sample_min: Optional lower bound overriding the constraint set.
        control_sample_max: Optional upper bound overriding the constraint set.

    Returns:
        Controls of shape ``(length, control_dim)``.
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
        """One clipped OU step.

        Args:
            x_t: Current control.
            eps_t: Standard-normal noise for this step.

        Returns:
            The next control twice (scan carry and output).
        """
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

    "iid_uniform" calls `_sample_control_sequence` with `key` unmodified, so its key
    consumption (and generated data) stays byte-identical to before this dispatcher existed.
    "smooth_ou" splits its own keys internally.

    Args:
        physics: Physics model (unused by the samplers; kept for a uniform signature).
        constraints: Constraint set providing the default box.
        length: Number of control steps.
        dt: Step length.
        key: PRNG key.
        control_profile: ``"iid_uniform"`` or ``"smooth_ou"``.
        control_tau: OU time constant for ``"smooth_ou"``.
        control_sample_min: Optional lower bound overriding the constraint set.
        control_sample_max: Optional upper bound overriding the constraint set.

    Returns:
        Controls of shape ``(length, control_dim)``.

    Raises:
        ValueError: If ``control_profile`` is unsupported.
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
    """Integrate the physics over one step with Tsit5 and an adaptive controller.

    Args:
        physics: Physics model.
        x_prev: State at the start of the step.
        control: Control vector.
        params: Physical parameters.
        dt: Step length.

    Returns:
        Tuple ``(next_state, solve_ok)``.
    """
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
    """Simulate one trajectory under sampled controls.

    Args:
        physics: Physics model.
        constraints: Constraint set providing the sampling box.
        trajectory_length: Number of states.
        dt: Step length.
        key: PRNG key.
        true_params: Physical parameters of the simulated system.
        control_profile: ``"iid_uniform"`` or ``"smooth_ou"``.
        control_tau: OU time constant for ``"smooth_ou"``.
        min_speed: Optional speed floor for the initial state.
        control_sample_min: Optional lower control-sampling bound.
        control_sample_max: Optional upper control-sampling bound.

    Returns:
        Tuple ``(states, controls, solve_ok)``.
    """
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

    def _scan_step(
        carry: tuple[jax.Array, jax.Array], control: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Advance one step and track solver success.

        Args:
            carry: ``(state, previous_ok)``.
            control: Control for this step.

        Returns:
            New carry and the per-step ``(next_state, ok)`` output.
        """
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
    """Whether every state-control pair of a trajectory satisfies the constraints.

    Args:
        constraints: Constraint set.
        states: States of shape ``(T, state_dim)``.
        controls: Controls of shape ``(T - 1, control_dim)``.

    Returns:
        Boolean scalar.
    """
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

    `control_sample_min`/`max` change only the box controls are drawn from, never the
    filtering.

    `min_speed`, when set, keeps a trajectory only if its longitudinal speed (state index 3)
    never drops below the floor at any timestep. Meaningful only for the dynamic-bicycle 6D
    layout (tire-slip terms are singular at standstill there); rejected outright for any
    other system. `min_speed=None` (default) reproduces legacy generation byte-for-byte.

    Args:
        physics: Physics model.
        constraints: Constraint set providing the sampling box and feasibility check.
        num_trajectories: Number of trajectories to return.
        trajectory_length: Number of states per trajectory.
        dt: Step length.
        key: PRNG key.
        true_params: Physical parameters of the simulated system.
        control_profile: ``"iid_uniform"`` or ``"smooth_ou"``.
        control_tau: OU time constant for ``"smooth_ou"``.
        min_speed: Optional minimum longitudinal speed (dynamic-bicycle layout only).
        control_sample_min: Optional lower control-sampling bound.
        control_sample_max: Optional upper control-sampling bound.

    Returns:
        Tuple ``(states, controls)`` with leading dimension ``num_trajectories``.

    Raises:
        ValueError: If ``min_speed`` is set for a non dynamic-bicycle layout.
        RuntimeError: If too few feasible trajectories can be generated.
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
