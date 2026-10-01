# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Evaluation metrics for feasibility and fidelity.

Also exposes a real-data metric suite (inD experiments):
:class:`EmpiricalEnvelope`, :func:`estimate_empirical_envelope`,
:func:`envelope_constraint`, :func:`empirical_envelope_violation_rate`,
:func:`empirical_envelope_violation_magnitude`, :func:`ade`, :func:`fde`,
:func:`gt_normalised_dynamics_residual`, :func:`estimate_gt_reference_residual`,
:func:`jerk`, :func:`heading_jerk`, :func:`kinematic_bicycle_inverse_controls`,
:data:`METRIC_VERSION`.

``fidelity``, ``dynamics_violation_*``, ``compute_metrics`` are pinned by
the simulated-experiment paper tables and regression tests; do not change their behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp

from made.models import AugmentedDynamics
from made.physics import BoxConstraints, ConstraintSet, KinematicBicycle, PhysicsModel

# Locked; do not change without sign-off.
METRIC_VERSION: str = "real-data-v3-gaussian"


def _is_batched(array: jax.Array) -> bool:
    """Whether the array carries a leading batch dimension.

    Args:
        array: Array of states or controls.
    Returns:
        True when the array has at least three dimensions.
    """
    return array.ndim >= 3


def _align_controls_to_states(x: jax.Array, u: jax.Array) -> jax.Array:
    """Return one control per state for inequality metrics.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
    Returns:
        Controls with one row per state.
    """
    if u.shape[0] == x.shape[0]:
        return u
    return jnp.concatenate([u, u[-1:]], axis=0)


def _violation_channel_slice(constraints: ConstraintSet, channel: str) -> slice | None:
    """Column slice of a `BoxConstraints` violation vector for one channel.

    `BoxConstraints.__call__` concatenates, in this order::

        [state - state_max, state_min - state, control - control_max, control_min - control]

    First `2 * state_dim` columns are STATE-bound, remainder are CONTROL-bound. Returns `None`
    when the split can't be taken (e.g. `SpeedNormConstraint` has no `state_min`) -- caller
    must then report the combined figure, never guess a split.

    Args:
        constraints: Constraint set to score against.
        channel: Which bounds to score: "all", "state" or "control".
    Returns:
        Column slice, or None for the combined figure.
    Raises:
        ValueError: If `channel` is not "all", "state" or "control".
    """
    if channel == "all":
        return None
    try:
        state_dim = int(constraints.state_min.shape[-1])
    except (AttributeError, TypeError, IndexError):
        return None
    if channel == "state":
        return slice(0, 2 * state_dim)
    if channel == "control":
        return slice(2 * state_dim, None)
    raise ValueError(f"channel must be 'all', 'state' or 'control', got {channel!r}")


def _trajectory_inequality_violation_rate(
    x: jax.Array,
    u: jax.Array,
    constraints: ConstraintSet,
    channel: str = "all",
) -> jax.Array:
    """`channel` selects state-bound, control-bound or both.

    Shares one implementation across channels so components can't drift from the total.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        constraints: Constraint set to score against.
        channel: Which bounds to score: "all", "state" or "control".
    Returns:
        Fraction of violating steps.
    """
    if x.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x.dtype)
    if u.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x.dtype)
    aligned_u = _align_controls_to_states(x, u)
    violations = jax.vmap(lambda x_i, u_i: constraints(x_i, u_i))(x, aligned_u)
    sl = _violation_channel_slice(constraints, channel)
    if sl is not None:
        violations = violations[:, sl]
    return jnp.mean(jnp.any(violations > 0, axis=-1))


def _trajectory_inequality_violation_magnitude(
    x: jax.Array,
    u: jax.Array,
    constraints: ConstraintSet,
    channel: str = "all",
) -> jax.Array:
    """`channel` selects state-bound, control-bound or both. See the rate function above.

    Total is not the sum of the components: magnitude is the L2 norm over the whole violation
    vector (e.g. state=4, control=6 gives sqrt(4^2+6^2)=7.2111, not 10), and rate is an "any"
    over the union (`rate_all <= rate_state + rate_control`). Reconstructing one channel by
    subtracting the other from the total gives a wrong but plausible-looking number.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        constraints: Constraint set to score against.
        channel: Which bounds to score: "all", "state" or "control".
    Returns:
        Mean violation magnitude over violating steps.
    """
    if x.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x.dtype)
    if u.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x.dtype)
    aligned_u = _align_controls_to_states(x, u)
    violation = jax.vmap(lambda x_i, u_i: constraints.violation(x_i, u_i))(x, aligned_u)
    sl = _violation_channel_slice(constraints, channel)
    if sl is not None:
        violation = violation[:, sl]
    per_step = jnp.linalg.norm(violation, axis=-1)
    mask = per_step > 0
    count = jnp.sum(mask)
    mean_violation = jnp.where(count > 0, jnp.sum(per_step * mask) / count, 0.0)
    return mean_violation


def _trajectory_dynamics_violation_known(
    x: jax.Array,
    u: jax.Array,
    physics: PhysicsModel,
    params: jax.Array,
    dt: float,
) -> jax.Array:
    """Mean one-step violation of one trajectory under the known physics model.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        physics: Known physics model.
        params: Physics parameters.
        dt: Timestep in seconds.
    Returns:
        Scalar metric value.
    """
    if x.shape[0] <= 1:
        return jnp.asarray(0.0, dtype=x.dtype)
    predicted = jax.vmap(
        lambda x_prev, control: diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: physics.vector_field(y, args["control"], params, t)),
            diffrax.Heun(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=x_prev,
            args={"control": control},
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=16,
        ).ys[0]
    )(x[:-1], u)
    return jnp.mean(jnp.linalg.norm(x[1:] - predicted, axis=-1))


def _trajectory_dynamics_violation_learned(
    x: jax.Array,
    u: jax.Array,
    augmented_dynamics: AugmentedDynamics,
    params: jax.Array,
    dt: float,
) -> jax.Array:
    """Mean one-step violation of one trajectory under the learned dynamics.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        augmented_dynamics: Learned augmented dynamics.
        params: Physics parameters.
        dt: Timestep in seconds.
    Returns:
        Scalar metric value.
    """
    if x.shape[0] <= 1:
        return jnp.asarray(0.0, dtype=x.dtype)
    predicted = jax.vmap(
        lambda x_prev, control: augmented_dynamics.integrate(x_prev, control, params, dt)
    )(x[:-1], u)
    return jnp.mean(jnp.linalg.norm(x[1:] - predicted, axis=-1))


def _trajectory_fidelity(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Mean per-step distance of one trajectory to ground truth.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Scalar metric value.
    """
    if x_corrected.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x_corrected.dtype)
    return jnp.mean(jnp.linalg.norm(x_corrected - x_gt, axis=-1))


def inequality_violation_rate(
    x: jax.Array,
    u: jax.Array,
    constraints: ConstraintSet,
    channel: str = "all",
) -> jax.Array:
    """Fraction of trajectory steps that violate at least one constraint.

    Scoring convention for controls, applied everywhere: a row that emits controls (MaDE,
    EKF/RTS smoother) is scored on its own emitted controls; a row that emits none (raw, clamp)
    is scored on controls recovered from its state trajectory via the inverse known model. No
    row is scored on states alone. This is the INEQUALITY convention only -- dynamics residual
    and control recovery still use a row's emitted controls where it has them.

    Consequence: clamp fails control bounds it has no mechanism to satisfy (it projects states,
    no dynamics step), so it is not exactly 0.0000 on the simulated panel.

    Returns FLOAT32 (`jnp.mean` of a bool array is float32 even under x64), unlike every other
    metric here (float64).

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        constraints: Constraint set to score against.
        channel: Which bounds to score: "all", "state" or "control".
    Returns:
        Fraction of violating steps, as float32.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(
                lambda x_i, u_i: _trajectory_inequality_violation_rate(
                    x_i, u_i, constraints, channel
                )
            )(x, u)
        )
    return _trajectory_inequality_violation_rate(x, u, constraints, channel)


def inequality_violation_magnitude(
    x: jax.Array,
    u: jax.Array,
    constraints: ConstraintSet,
    channel: str = "all",
) -> jax.Array:
    """Mean violation magnitude over violating steps.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        constraints: Constraint set to score against.
        channel: Which bounds to score: "all", "state" or "control".
    Returns:
        Scalar metric value.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(
                lambda x_i, u_i: _trajectory_inequality_violation_magnitude(
                    x_i, u_i, constraints, channel
                )
            )(x, u)
        )
    return _trajectory_inequality_violation_magnitude(x, u, constraints, channel)


def dynamics_violation_known(
    x: jax.Array,
    u: jax.Array,
    physics: PhysicsModel,
    params: jax.Array,
    dt: float,
) -> jax.Array:
    """Mean one-step violation under the known physics model.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        physics: Known physics model.
        params: Physics parameters.
        dt: Timestep in seconds.
    Returns:
        Scalar metric value.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(
                lambda x_i, u_i: _trajectory_dynamics_violation_known(
                    x_i,
                    u_i,
                    physics,
                    params,
                    dt,
                )
            )(x, u)
        )
    return _trajectory_dynamics_violation_known(x, u, physics, params, dt)


def dynamics_violation_learned(
    x: jax.Array,
    u: jax.Array,
    augmented_dynamics: AugmentedDynamics,
    params: jax.Array,
    dt: float,
) -> jax.Array:
    """Mean one-step violation under the learned augmented dynamics.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        augmented_dynamics: Learned augmented dynamics.
        params: Physics parameters.
        dt: Timestep in seconds.
    Returns:
        Scalar metric value.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(
                lambda x_i, u_i: _trajectory_dynamics_violation_learned(
                    x_i, u_i, augmented_dynamics, params, dt
                )
            )(x, u)
        )
    return _trajectory_dynamics_violation_learned(x, u, augmented_dynamics, params, dt)


def dynamics_violation_true(
    x: jax.Array,
    u: jax.Array,
    true_dynamics: PhysicsModel,
    true_params: jax.Array,
    dt: float,
) -> jax.Array:
    """Mean one-step violation under the ground-truth physics model.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        u: Controls aligned with the states.
        true_dynamics: Ground-truth physics model.
        true_params: Ground-truth physics parameters.
        dt: Timestep in seconds.
    Returns:
        Scalar metric value.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(
                lambda x_i, u_i: _trajectory_dynamics_violation_known(
                    x_i, u_i, true_dynamics, true_params, dt
                )
            )(x, u)
        )
    return _trajectory_dynamics_violation_known(x, u, true_dynamics, true_params, dt)


def fidelity(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Mean per-step distance to ground-truth states.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Scalar metric value.
    """
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_fidelity)(x_corrected, x_gt))
    return _trajectory_fidelity(x_corrected, x_gt)


def compute_metrics(
    x_corrected: jax.Array,
    u_corrected: jax.Array,
    x_gt: jax.Array,
    u_gt: jax.Array,
    constraints: ConstraintSet,
    physics_known: PhysicsModel,
    dynamics_learned: AugmentedDynamics,
    params: jax.Array,
    dt: float,
    dynamics_true: PhysicsModel | None = None,
    true_params: jax.Array | None = None,
    u_for_dynamics: jax.Array | None = None,
    u_for_inequality: jax.Array | None = None,
    u_for_dyn_known: jax.Array | None = None,
    u_for_dyn_true: jax.Array | None = None,
    u_for_dyn_learned: jax.Array | None = None,
    dyn_learned_available: bool = True,
) -> dict[str, float]:
    """Compute the standard evaluation metrics for a batch or one trajectory.

    Most metrics use ``u_corrected``, the control the method actually produced. The two
    dynamics-violation metrics ask whether ``x_corrected`` is consistent with some control
    driven through the physics model. For MaDE variants ``u_corrected`` is the model's own
    predicted control. Non-MaDE baselines (Clamp, MLP, FAB) don't infer controls, so
    ``u_corrected`` is a zero placeholder; scoring dynamics violation against a zero-control
    free coast is meaningless (Clamp trivially reads ~0.2 despite enforcing nothing).

    ``u_for_dynamics`` overrides the control used only for the two dynamics-violation calls.
    Pass the ground-truth control trajectory for non-MaDE baselines so the metric measures
    consistency with the actual rollout, not a free coast. Leave it ``None`` for MaDE variants.

    ``dynamics_violation_learned`` always uses ``u_corrected`` (ignoring ``u_for_dynamics``)
    since it's a cycle-consistency check tied to the model's own output. ``u_for_dyn_learned``
    is for rows with no learned model of their own, borrowing the canonical MaDE model's.

    Args:
        x_corrected: Corrected states.
        u_corrected: Controls produced by the method.
        x_gt: Ground-truth states.
        u_gt: Ground-truth controls.
        constraints: Constraint set for the inequality metrics.
        physics_known: Known physics model.
        dynamics_learned: Learned augmented dynamics.
        params: Physics parameters.
        dt: Timestep in seconds.
        dynamics_true: Ground-truth physics model, if available.
        true_params: Ground-truth parameters, if available.
        u_for_dynamics: Control override for the dynamics-violation metrics.
        u_for_inequality: Control override for the inequality metrics.
        u_for_dyn_known: Control override for the known-model dynamics violation.
        u_for_dyn_true: Control override for the ground-truth dynamics violation.
        u_for_dyn_learned: Control override for the learned-model dynamics violation.
        dyn_learned_available: Whether the learned-dynamics metric applies to this row.
    Returns:
        Dict of metric name to value.
    """
    del u_gt
    u_dyn = u_corrected if u_for_dynamics is None else u_for_dynamics
    # Per-metric control set:
    #   inequality -- own emitted controls, else recovered through the known model.
    #   Dyn.-K -- recovered through the known model, every row.
    #   Dyn.-T -- for the dynamic bicycle (underspecified), own emitted controls where
    #             emitted, else recovered through the known (kinematic-bicycle) model's
    #             inverse -- never the true model's inverse (off-manifold blow-up). For the
    #             three fully specified systems this equals Dyn.-K. Caller supplies the
    #             choice via `u_for_dyn_true`; applied in `scripts/evaluate.py`.
    #   Dyn.-L -- own emitted control for a row with a learned model; a row with none borrows
    #             the canonical MaDE model's controls/residual via `u_for_dyn_learned` and
    #             `dynamics_learned`. `dyn_learned_available=False` omits the metric.
    u_ineq = u_corrected if u_for_inequality is None else u_for_inequality
    u_dyn_k = u_dyn if u_for_dyn_known is None else u_for_dyn_known
    u_dyn_t = u_dyn if u_for_dyn_true is None else u_for_dyn_true
    metrics = {
        "inequality_violation_rate": float(
            inequality_violation_rate(x_corrected, u_ineq, constraints)
        ),
        "inequality_violation_magnitude": float(
            inequality_violation_magnitude(x_corrected, u_ineq, constraints)
        ),
        "dynamics_violation_known": float(
            dynamics_violation_known(x_corrected, u_dyn_k, physics_known, params, dt)
        ),
        **({"dynamics_violation_learned": float(
            dynamics_violation_learned(
                x_corrected,
                u_corrected if u_for_dyn_learned is None else u_for_dyn_learned,
                dynamics_learned, params, dt)
        )} if dyn_learned_available else {}),
        "fidelity": float(fidelity(x_corrected, x_gt)),
    }
    if dynamics_true is not None:
        if true_params is None:
            raise ValueError("true_params must be provided when dynamics_true is set.")
        metrics["dynamics_violation_true"] = float(
            dynamics_violation_true(x_corrected, u_dyn_t, dynamics_true, true_params, dt)
        )
    return metrics


# Real-data (inD experiments) metric helpers, additive siblings of the simulated-experiment
# helpers above.


@dataclass(frozen=True)
class EmpiricalEnvelope:
    """Per-dimension empirical state and control bounds.

    Estimated from a training-split ground-truth distribution (states + inferred controls) by
    :func:`estimate_empirical_envelope`. Dataset-support proxy for inequality metrics on
    real-data trajectories where physical actuator limits are unknown. Not physical limits.
    """

    state_min: jax.Array  # (state_dim,)
    state_max: jax.Array  # (state_dim,)
    control_min: jax.Array  # (control_dim,)
    control_max: jax.Array  # (control_dim,)


def _flatten_with_lengths(
    array: jax.Array,
    lengths: jax.Array | None,
) -> jax.Array:
    """Flatten ``(N, T, D)`` over ``(N, T)`` honouring per-trajectory ``lengths``.

    Returns ``(M, D)``, ``M`` the total valid timesteps. ``lengths=None`` treats every entry
    as valid. Mirrors the inD pipeline, which pads ragged trajectories to a fixed length.

    Args:
        array: Array of shape (N, T, D).
        lengths: Valid length per trajectory; None treats every step as valid.
    Returns:
        Valid timesteps, shape (M, D).
    """
    if array.ndim == 2:
        return array
    if lengths is None:
        return array.reshape(-1, array.shape[-1])
    # mask (N, T): t < lengths[i]
    t = array.shape[1]
    t_idx = jnp.arange(t)
    mask = t_idx[None, :] < lengths[:, None]
    flat = array.reshape(-1, array.shape[-1])
    flat_mask = mask.reshape(-1)
    return flat[flat_mask]


def _round_outward(
    bounds_low: jax.Array,
    bounds_high: jax.Array,
    rounding: tuple[float, ...] | None,
) -> tuple[jax.Array, jax.Array]:
    """Round lower bounds DOWN and upper bounds UP to multiples of ``rounding``.

    Args:
        bounds_low: Lower bounds.
        bounds_high: Upper bounds.
        rounding: Per-dimension rounding increments; None disables rounding.
    Returns:
        Tuple (rounded lower bounds, rounded upper bounds).
    """
    if rounding is None:
        return bounds_low, bounds_high
    inc = jnp.asarray(rounding, dtype=bounds_low.dtype)
    low = jnp.floor(bounds_low / inc) * inc
    high = jnp.ceil(bounds_high / inc) * inc
    return low, high


def estimate_empirical_envelope(
    states: jax.Array,
    controls: jax.Array,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    state_rounding: tuple[float, ...] | None = None,
    control_rounding: tuple[float, ...] | None = None,
    lengths: jax.Array | None = None,
) -> EmpiricalEnvelope:
    """Estimate per-dimension state/control bounds from training data.

    ``states`` may be ``(N, T, state_dim)`` or ``(T, state_dim)``; ``controls`` may be
    ``(N, T-1, control_dim)`` or ``(T-1, control_dim)``. With ``lengths`` (inD pipeline),
    padding past ``lengths[i]`` is masked before quantile estimation.

    Bounds are the 1st/99th percentile (configurable), rounded outward when increments are
    given. No IO; callable from JIT contexts.

    Args:
        states: Training states.
        controls: Training controls.
        lower_quantile: Lower quantile for the bounds.
        upper_quantile: Upper quantile for the bounds.
        state_rounding: Per-dimension rounding increments for state bounds.
        control_rounding: Per-dimension rounding increments for control bounds.
        lengths: Valid length per trajectory; masks padding.
    Returns:
        The estimated envelope.
    """
    state_lengths = lengths
    flat_states = _flatten_with_lengths(states, state_lengths)
    if flat_states.shape[0] == 0:
        raise ValueError("estimate_empirical_envelope: no valid states after masking.")
    s_low = jnp.quantile(flat_states, lower_quantile, axis=0)
    s_high = jnp.quantile(flat_states, upper_quantile, axis=0)
    s_low, s_high = _round_outward(s_low, s_high, state_rounding)

    # Controls have one fewer step per trajectory; subtract 1 from lengths to align the mask.
    if state_lengths is not None and controls.ndim == 3:
        ctrl_lengths = jnp.maximum(state_lengths - 1, 0)
    else:
        ctrl_lengths = None
    flat_controls = _flatten_with_lengths(controls, ctrl_lengths)
    if flat_controls.shape[0] == 0:
        raise ValueError(
            "estimate_empirical_envelope: no valid controls after masking."
        )
    c_low = jnp.quantile(flat_controls, lower_quantile, axis=0)
    c_high = jnp.quantile(flat_controls, upper_quantile, axis=0)
    c_low, c_high = _round_outward(c_low, c_high, control_rounding)

    return EmpiricalEnvelope(
        state_min=s_low,
        state_max=s_high,
        control_min=c_low,
        control_max=c_high,
    )


def _envelope_state_only_constraint(envelope: EmpiricalEnvelope) -> ConstraintSet:
    """Return a state-only BoxConstraints proxy with permissive control bounds.

    Used when the caller passes ``u=None`` to the empirical-envelope helpers: control box is
    ``[-inf, +inf]`` so control-side terms contribute zero violation.

    Args:
        envelope: Empirical envelope of the training data.
    Returns:
        Constraint set with permissive control bounds.
    """
    state_dtype = envelope.state_min.dtype
    ctrl_dim = envelope.control_min.shape[-1]
    inf = jnp.asarray(jnp.inf, dtype=state_dtype)
    return BoxConstraints(
        state_min=envelope.state_min,
        state_max=envelope.state_max,
        control_min=jnp.full((ctrl_dim,), -inf, dtype=state_dtype),
        control_max=jnp.full((ctrl_dim,), inf, dtype=state_dtype),
    )


def envelope_constraint(envelope: EmpiricalEnvelope) -> ConstraintSet:
    """Returns x,y-stripped bounds for use as an inequality constraint.

    For sampling/perturbation finite x,y, use _envelope_sampling_constraint
    or EmpiricalEnvelope directly.

    Args:
        envelope: Empirical envelope of the training data.
    Returns:
        Constraint set with x and y bounds removed.
    """
    xy_idx = jnp.asarray([0, 1])
    state_min = envelope.state_min.at[xy_idx].set(-jnp.inf)
    state_max = envelope.state_max.at[xy_idx].set(jnp.inf)
    return BoxConstraints(
        state_min=state_min,
        state_max=state_max,
        control_min=envelope.control_min,
        control_max=envelope.control_max,
    )


def _envelope_sampling_constraint(envelope: EmpiricalEnvelope) -> "BoxConstraints":
    """Return the unwrapped finite box from the envelope (including x,y bounds).

    Use this for sampling or perturbation where finite x,y limits are needed.

    Args:
        envelope: Empirical envelope of the training data.
    Returns:
        Finite box including x and y bounds.
    """
    return BoxConstraints(
        state_min=envelope.state_min,
        state_max=envelope.state_max,
        control_min=envelope.control_min,
        control_max=envelope.control_max,
    )


def _envelope_constraint(envelope: EmpiricalEnvelope) -> ConstraintSet:
    """Constraint set derived from the envelope for inequality scoring.

    Args:
        envelope: Empirical envelope of the training data.
    Returns:
        Constraint set.
    """
    return envelope_constraint(envelope)


def empirical_envelope_violation_rate(
    x: jax.Array,
    u: jax.Array | None,
    envelope: EmpiricalEnvelope,
) -> jax.Array:
    """Fraction of timesteps that violate the empirical envelope.

    When ``u`` is ``None``, only the state-side bounds are checked
    (used by non-MaDE variants which do not produce internal controls).
    Batched/unbatched dispatch matches :func:`inequality_violation_rate`.

    Args:
        x: States.
        u: Controls, or None to check state bounds only.
        envelope: Empirical envelope.
    Returns:
        Fraction of violating steps.
    """
    if u is None:
        constraints = _envelope_state_only_constraint(envelope)
        # Zero-control placeholder; control bounds are ±inf so it contributes zero violation.
        if x.ndim == 3:
            zero_u = jnp.zeros(
                (x.shape[0], x.shape[1], envelope.control_min.shape[-1]),
                dtype=x.dtype,
            )
        else:
            zero_u = jnp.zeros(
                (x.shape[0], envelope.control_min.shape[-1]), dtype=x.dtype
            )
        return inequality_violation_rate(x, zero_u, constraints)
    return inequality_violation_rate(x, u, _envelope_constraint(envelope))


def empirical_envelope_violation_magnitude(
    x: jax.Array,
    u: jax.Array | None,
    envelope: EmpiricalEnvelope,
) -> jax.Array:
    """Mean violation magnitude (L2) over violating timesteps.

    Mirrors the ``u=None`` semantics of
    :func:`empirical_envelope_violation_rate`.

    Args:
        x: States.
        u: Controls, or None to check state bounds only.
        envelope: Empirical envelope.
    Returns:
        Mean violation magnitude.
    """
    if u is None:
        constraints = _envelope_state_only_constraint(envelope)
        if x.ndim == 3:
            zero_u = jnp.zeros(
                (x.shape[0], x.shape[1], envelope.control_min.shape[-1]),
                dtype=x.dtype,
            )
        else:
            zero_u = jnp.zeros(
                (x.shape[0], envelope.control_min.shape[-1]), dtype=x.dtype
            )
        return inequality_violation_magnitude(x, zero_u, constraints)
    return inequality_violation_magnitude(x, u, _envelope_constraint(envelope))


def ade(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Average Displacement Error.

    Alias of :func:`fidelity` (kept for existing imports of that name). Full-state norm
    (all state dims), used by the simulated experiments and the inD results JSON.
    Printed inD tables use :func:`position_ade` / :func:`position_fde` instead, restricted to
    (x, y).

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Average displacement error.
    """
    return fidelity(x_corrected, x_gt)


def _trajectory_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Final-step L2 distance of one trajectory to ground truth.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Scalar distance.
    """
    if x_corrected.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x_corrected.dtype)
    return jnp.linalg.norm(x_corrected[-1] - x_gt[-1])


def fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Final Displacement Error — L2 distance between final timesteps.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Final displacement error.
    """
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_fde)(x_corrected, x_gt))
    return _trajectory_fde(x_corrected, x_gt)


def _trajectory_position_ade(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Mean over steps of the Euclidean (x, y) distance -- state indices 0 and 1.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Mean position error.
    """
    return _trajectory_fidelity(x_corrected[..., :2], x_gt[..., :2])


def _trajectory_position_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Euclidean (x, y) distance at the final step.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Final position error.
    """
    return _trajectory_fde(x_corrected[..., :2], x_gt[..., :2])


def position_ade(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """inD ADE, position only, metres. Batched ([K,F,D]) or single ([F,D]).

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Position ADE in metres.
    """
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_position_ade)(x_corrected, x_gt))
    return _trajectory_position_ade(x_corrected, x_gt)


def position_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """inD FDE, position only, metres. Batched or single.

    Args:
        x_corrected: Corrected states.
        x_gt: Ground-truth states.
    Returns:
        Position FDE in metres.
    """
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_position_fde)(x_corrected, x_gt))
    return _trajectory_position_fde(x_corrected, x_gt)


def gt_normalised_dynamics_residual(
    x: jax.Array,
    u: jax.Array,
    physics: PhysicsModel,
    params: jax.Array,
    dt: float,
    gt_reference_residual: float,
    *,
    eps: float = 1e-12,
) -> jax.Array:
    """Method known-physics residual normalised by the GT reference residual.

    Returns ``method_residual / max(gt_reference_residual, eps) - 1.0``. Near ``0.0`` means the
    method matches the dataset's intrinsic known-physics residual; positive is drift, negative
    is over-projection toward the simplified known model.

    .. warning:: On a trajectory generated by the same KB+L Heun stencil that
        ``dynamics_violation_known`` evaluates against, the residual is bounded by Heun
        discretisation error and the ratio is uninformative. Use only on trajectories NOT
        produced by that stencil (real inD data, or rollouts off the KB+L manifold).

    Args:
        x: Trajectory states.
        u: Controls.
        physics: Known physics model.
        params: Physics parameters.
        dt: Timestep in seconds.
        gt_reference_residual: Reference residual on ground-truth trajectories.
        eps: Floor on the reference residual.
    Returns:
        Normalised residual.
    """
    method = dynamics_violation_known(x, u, physics, params, dt)
    denom = jnp.maximum(jnp.asarray(gt_reference_residual, x.dtype), eps)
    return method / denom - 1.0


def estimate_gt_reference_residual(
    gt_states: jax.Array,
    gt_controls: jax.Array,
    physics: PhysicsModel,
    params: jax.Array,
    dt: float,
    *,
    lengths: jax.Array | None = None,
) -> float:
    """Mean known-model transition residual on training-split GT trajectories.

    Per-trajectory residual via ``_trajectory_dynamics_violation_known``, meaned across
    trajectories (``lengths`` masks out padding). Returns a Python ``float`` for embedding in
    JSON metadata.

    Args:
        gt_states: Ground-truth training states.
        gt_controls: Ground-truth training controls.
        physics: Known physics model.
        params: Physics parameters.
        dt: Timestep in seconds.
        lengths: Valid length per trajectory; masks padding.
    Returns:
        Mean residual as a Python float.
    """
    if gt_states.ndim == 2:
        residual = _trajectory_dynamics_violation_known(
            gt_states, gt_controls, physics, params, dt
        )
        return float(residual)

    n = gt_states.shape[0]
    per_traj_residuals = []
    if lengths is None:
        lengths_local = jnp.full((n,), gt_states.shape[1], dtype=jnp.int32)
    else:
        lengths_local = lengths

    for i in range(n):
        length = int(lengths_local[i])
        if length < 2:
            continue
        x_i = gt_states[i, :length]
        u_i = gt_controls[i, : max(length - 1, 0)]
        per_traj_residuals.append(
            float(_trajectory_dynamics_violation_known(x_i, u_i, physics, params, dt))
        )

    if not per_traj_residuals:
        return 0.0
    return float(jnp.mean(jnp.asarray(per_traj_residuals, dtype=gt_states.dtype)))


def _trajectory_jerk(
    x: jax.Array,
    dt: float,
    position_indices: tuple[int, int],
) -> jax.Array:
    """Mean L2 norm of the third finite difference of position for one trajectory.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        dt: Timestep in seconds.
        position_indices: State indices of the (x, y) position.
    Returns:
        Scalar metric value.
    """
    if x.shape[0] < 4:
        return jnp.asarray(0.0, dtype=x.dtype)
    pos = x[..., jnp.asarray(position_indices)]
    # Third finite difference: j_t = (p_{t+3} - 3 p_{t+2} + 3 p_{t+1} - p_t) / dt^3
    j = (pos[3:] - 3.0 * pos[2:-1] + 3.0 * pos[1:-2] - pos[:-3]) / (dt ** 3)
    return jnp.mean(jnp.linalg.norm(j, axis=-1))


def jerk(
    x: jax.Array,
    dt: float,
    *,
    position_indices: tuple[int, int] = (0, 1),
) -> jax.Array:
    """Mean L2 norm of the third finite difference of position.

    Returns ``0.0`` when the trajectory has fewer than four states.
    Defaults assume inD state ordering ``[x, y, heading, speed]``.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        dt: Timestep in seconds.
        position_indices: State indices of the (x, y) position.
    Returns:
        Mean position jerk.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(lambda x_i: _trajectory_jerk(x_i, dt, position_indices))(x)
        )
    return _trajectory_jerk(x, dt, position_indices)


def _trajectory_heading_jerk(
    x: jax.Array,
    dt: float,
    heading_index: int,
) -> jax.Array:
    """Mean absolute heading jerk of one trajectory after unwrapping.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        dt: Timestep in seconds.
        heading_index: State index of the heading.
    Returns:
        Scalar metric value.
    """
    if x.shape[0] < 4:
        return jnp.asarray(0.0, dtype=x.dtype)
    h = jnp.unwrap(x[..., heading_index])
    j = (h[3:] - 3.0 * h[2:-1] + 3.0 * h[1:-2] - h[:-3]) / (dt ** 3)
    return jnp.mean(jnp.abs(j))


def heading_jerk(
    x: jax.Array,
    dt: float,
    *,
    heading_index: int = 2,
) -> jax.Array:
    """Mean absolute heading jerk after unwrapping.

    Returns ``0.0`` when the trajectory has fewer than four states.

    Args:
        x: States, shape (T, state_dim) or (N, T, state_dim).
        dt: Timestep in seconds.
        heading_index: State index of the heading.
    Returns:
        Mean absolute heading jerk.
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(lambda x_i: _trajectory_heading_jerk(x_i, dt, heading_index))(x)
        )
    return _trajectory_heading_jerk(x, dt, heading_index)


def _perturb_base_vector(envelope: EmpiricalEnvelope) -> jax.Array:
    """Base perturbation vector for the inD multiplier sweep.

    State dim 4 (KB ordering x, y, theta, v). Heading dim (idx 2) is ABSOLUTE
    radians (0.1), NOT envelope-relative.

    Args:
        envelope: Empirical envelope of the training data.
    Returns:
        Base perturbation per state dimension.
    """
    state_min = envelope.state_min
    state_max = envelope.state_max
    range_ = state_max - state_min
    safe_range = jnp.where(jnp.isfinite(range_), range_, 0.0)
    base = 0.1 * safe_range
    return base.at[2].set(jnp.asarray(0.1, dtype=base.dtype))


def _perturb_pair(
    state_curr: jax.Array,
    multiplier: jax.Array,
    base_vec: jax.Array,
    sign: jax.Array,
) -> jax.Array:
    """Single-pair perturbation: x_t + m * base * sign.

    Args:
        state_curr: Current state.
        multiplier: Perturbation multiplier.
        base_vec: Base perturbation vector.
        sign: Random sign per dimension.
    Returns:
        Perturbed state.
    """
    return state_curr + multiplier * base_vec * sign


def _stationary_infeasible_mask(v_avg: jax.Array, threshold: float = 0.5) -> jax.Array:
    """True iff |v_avg| < threshold (KB inverse degenerates near v=0).

    Args:
        v_avg: Average speed over the step.
        threshold: Speed below which a step is stationary.
    Returns:
        Boolean mask.
    """
    return jnp.abs(v_avg) < threshold


def _i_known(
    x_prev: jax.Array,
    x_curr: jax.Array,
    dt: float,
    *,
    wheelbase: float = 2.7,
    stationary_speed_threshold: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """Pair-wise known-physics inverse on a single (x_prev, x_curr).

    Returns (u, is_stationary_carry): u has shape (2,), is_stationary_carry is a scalar bool.
    Distinct from `kinematic_bicycle_inverse_controls` (trajectory-major).

    Args:
        x_prev: Previous state.
        x_curr: Current state.
        dt: Timestep in seconds.
        wheelbase: Vehicle wheelbase in metres.
        stationary_speed_threshold: Speed below which a step counts as stationary.

    Returns:
        Tuple (u, is_stationary_carry).
    """
    # FieldData deliberately: used on inD, where the simulated-data fallback does not apply. The
    # analytic base would change a published real-data feasibility metric; fallback stays scoped to
    # simulated data.
    from made.physics.kinematic_bicycle import KinematicBicycleFieldData
    physics = KinematicBicycleFieldData()
    params = jnp.asarray([wheelbase], dtype=x_prev.dtype)
    u = physics.known_control_prior(x_prev, x_curr, params, dt)
    v_avg = 0.5 * (x_prev[3] + x_curr[3])
    is_stationary = _stationary_infeasible_mask(v_avg, stationary_speed_threshold)
    return u, is_stationary


def evaluate_stepwise_feasible_split(
    x_perturbed: jax.Array,
    x_prev_gt: jax.Array,
    x_corrected: jax.Array,
    x_curr_gt: jax.Array,
    dt: float,
    constraints_phys: "ConstraintSet",
    *,
    wheelbase: float = 2.7,
) -> dict[str, jax.Array]:
    """Per-step Fid feas/inf split + fraction_infeasible + fraction_stationary_carry.

    Inputs are flat (M, state_dim). Feasibility predicate uses _i_known (NOT
    method's I), so bucket assignment is method-independent.

    Returns dict with scalar jax arrays:
        fidelity_feasible, fidelity_infeasible, fidelity_stationary,
        fraction_feasible, fraction_infeasible, fraction_stationary_carry.

    ``fraction_infeasible`` and ``fraction_stationary_carry`` are DISJOINT —
    stationary steps are removed from the infeasible bucket because
    ``_i_known``'s arctan branch is unreliable at ``|v_avg| < 0.5 m/s``.

    Args:
        x_perturbed: Perturbed current states, shape (M, state_dim).
        x_prev_gt: Ground-truth previous states.
        x_corrected: Corrected current states.
        x_curr_gt: Ground-truth current states.
        dt: Timestep in seconds.
        constraints_phys: Physical constraint set.
        wheelbase: Vehicle wheelbase in metres.
    Returns:
        Dict of scalar arrays with the keys listed above.
    """
    i_known_v = jax.vmap(lambda xp, xc: _i_known(xp, xc, dt, wheelbase=wheelbase))
    u_known, is_stationary = i_known_v(x_prev_gt, x_perturbed)
    state_min = constraints_phys.state_min
    state_max = constraints_phys.state_max
    ctrl_min = constraints_phys.control_min
    ctrl_max = constraints_phys.control_max
    state_viol = (jnp.maximum(0.0, state_min - x_perturbed).sum(axis=-1)
                  + jnp.maximum(0.0, x_perturbed - state_max).sum(axis=-1))
    ctrl_viol = (jnp.maximum(0.0, ctrl_min - u_known).sum(axis=-1)
                 + jnp.maximum(0.0, u_known - ctrl_max).sum(axis=-1))
    per_pair_viol = state_viol + ctrl_viol  # (M,)
    is_infeasible_raw = per_pair_viol > 0.0
    is_infeasible = is_infeasible_raw & (~is_stationary)
    feas_mask = (~is_infeasible) & (~is_stationary)
    fid = jnp.linalg.norm(x_corrected - x_curr_gt, axis=-1)  # (M,)
    fid_feas = jnp.where(feas_mask, fid, 0.0).sum() / jnp.maximum(jnp.sum(feas_mask), 1)
    fid_inf = jnp.where(is_infeasible, fid, 0.0).sum() / jnp.maximum(jnp.sum(is_infeasible), 1)
    fid_stat = jnp.where(is_stationary, fid, 0.0).sum() / jnp.maximum(jnp.sum(is_stationary), 1)
    frac_inf = jnp.mean(is_infeasible.astype(jnp.float64))
    frac_stat = jnp.mean(is_stationary.astype(jnp.float64))
    frac_feas = jnp.mean(feas_mask.astype(jnp.float64))
    return {
        "fidelity_feasible": fid_feas,
        "fidelity_infeasible": fid_inf,
        "fidelity_stationary": fid_stat,
        "fraction_feasible": frac_feas,
        "fraction_infeasible": frac_inf,
        "fraction_stationary_carry": frac_stat,
    }


def _physical_state_only_constraint(physical: "ConstraintSet") -> ConstraintSet:
    """Return a state-only proxy of *physical* with permissive control bounds.

    Used when :func:`compute_inequality_dual` is called with ``u=None`` (methods
    that do not produce their own controls, e.g. raw upstream-predictor output).
    Mirrors :func:`_envelope_state_only_constraint`.

    Args:
        physical: Physical constraint set.
    Returns:
        Constraint set with permissive control bounds.
    """
    state_dtype = physical.state_min.dtype
    ctrl_dim = physical.control_min.shape[-1]
    inf = jnp.asarray(jnp.inf, dtype=state_dtype)
    return BoxConstraints(
        state_min=physical.state_min,
        state_max=physical.state_max,
        control_min=jnp.full((ctrl_dim,), -inf, dtype=state_dtype),
        control_max=jnp.full((ctrl_dim,), inf, dtype=state_dtype),
    )


def compute_inequality_dual(
    x: jax.Array,
    u: jax.Array | None,
    envelope: EmpiricalEnvelope,
    physical: "ConstraintSet",
    lengths: jax.Array | None = None,
) -> dict[str, float]:
    """Dual yardstick: physical (headline) + envelope (data-coverage).

    When ``u`` is ``None`` (methods that do not produce their own controls,
    e.g. raw upstream-predictor output or the clamp baseline), only the
    state-side bounds are checked on both constraint sets — mirrors the
    ``u=None`` semantics of :func:`empirical_envelope_violation_rate`.

    Args:
        x: Corrected states.
        u: Controls, or None to check state bounds only.
        envelope: Empirical envelope.
        physical: Physical constraint set.
        lengths: Valid length per trajectory; masks padding.
    Returns:
        Dict of inequality metrics on both constraint sets.
    """
    env_box = envelope_constraint(envelope)
    if u is None:
        phys_box = _physical_state_only_constraint(physical)
        return {
            "inequality_violation_rate_physical": float(
                inequality_violation_rate(x, jnp.zeros(x.shape[:-1] + phys_box.control_min.shape, dtype=x.dtype), phys_box)
            ),
            "inequality_violation_magnitude_physical": float(
                inequality_violation_magnitude(x, jnp.zeros(x.shape[:-1] + phys_box.control_min.shape, dtype=x.dtype), phys_box)
            ),
            "inequality_violation_rate_envelope": float(empirical_envelope_violation_rate(x, None, envelope)),
            "inequality_violation_magnitude_envelope": float(
                empirical_envelope_violation_magnitude(x, None, envelope)
            ),
        }
    out = {
        "inequality_violation_rate_physical": float(inequality_violation_rate(x, u, physical)),
        "inequality_violation_magnitude_physical": float(inequality_violation_magnitude(x, u, physical)),
        "inequality_violation_rate_envelope": float(inequality_violation_rate(x, u, env_box)),
        "inequality_violation_magnitude_envelope": float(inequality_violation_magnitude(x, u, env_box)),
    }
    # Four-way split on the PHYSICAL set, recorded as separate fields (primary record); the
    # combined values above are derived. Total is not the sum of components: magnitude
    # combines in quadrature, rate is an "any" over the union. Needed for clamp to read
    # honestly: it has zero state-bound violation by construction (no dynamics step) while
    # its control-bound violation is whatever its implied controls score.
    for channel in ("state", "control"):
        out[f"inequality_violation_rate_physical_{channel}"] = float(
            inequality_violation_rate(x, u, physical, channel)
        )
        out[f"inequality_violation_magnitude_physical_{channel}"] = float(
            inequality_violation_magnitude(x, u, physical, channel)
        )
    return out


def kinematic_bicycle_inverse_controls(
    states: jax.Array,
    dt: float,
    *,
    wheelbase: float = 2.7,
    eps: float = 1e-6,
    stationary_speed_threshold: float = 0.5,
) -> tuple[jax.Array, dict]:
    """Recover ``(δ, a)`` from consecutive KB states, exact-against-Heun.

    Wraps ``KinematicBicycle.known_control_prior`` (`src/made/physics/kinematic_bicycle.py:59`),
    the Heun-exact inverse for ZOH-constant ``(δ, a)`` with fixed wheelbase ``L``.

    Stationary-frame filter: when ``|v_avg| < stationary_speed_threshold`` (default 0.5 m/s)
    the closed-form inverse ``δ = arctan(L * dθ / (v_avg * dt))`` is near a singularity. To
    avoid spurious ±π/2 spikes from sensor jitter on parked vehicles, carry forward the
    previous frame's ``δ`` (``δ = 0`` at ``t == 0``). ``aux`` records ``stationary_frame_count``.

    .. warning:: This is the exact-against-Heun inverse for the KB stencil with wheelbase
        ``L``. Composed with :func:`gt_normalised_dynamics_residual` on a trajectory generated
        by the same KB+L Heun stencil, the residual is bounded by Heun discretisation error
        plus ``v_avg``-regularisation and is uninformative. Informative only on trajectories
        NOT generated by that stencil (real inD data; not Heun-perfect synthetic ones).

    Returns:
        Tuple ``(controls, aux)``. ``controls`` has shape ``(N, T-1, 2)`` or ``(T-1, 2)`` and holds
        the recovered ``[δ, a]`` per consecutive state pair. ``aux`` is
        ``{"stationary_frame_count": int}``, the transitions in the stationary-frame branch.

    Args:
        states: States, shape (N, T, 4) or (T, 4).
        dt: Timestep in seconds.
        wheelbase: Vehicle wheelbase in metres.
        eps: Unused; regularisation lives in the physics model.
        stationary_speed_threshold: Speed below which the previous steering angle is carried
            forward.
    """
    del eps  # v_avg regularisation is inside known_control_prior
    physics = KinematicBicycle()
    params = jnp.asarray([wheelbase], dtype=jnp.float64)

    def _per_traj(traj: jax.Array) -> tuple[jax.Array, jax.Array]:
        # traj: (T, 4) -> controls (T-1, 2) plus a (T-1,) stationary mask.
        """Recover controls for one trajectory.

        Args:
            traj: States, shape (T, 4).
        Returns:
            Tuple (controls (T-1, 2), stationary mask (T-1,)).
        """
        prev = traj[:-1]
        curr = traj[1:]
        v_avg = 0.5 * (prev[:, 3] + curr[:, 3])
        stationary = jnp.abs(v_avg) < stationary_speed_threshold

        raw = jax.vmap(
            lambda x_p, x_c: physics.known_control_prior(x_p, x_c, params, dt)
        )(prev, curr)

        # Carry forward δ when stationary: walk left-to-right, replacing
        # δ_t with δ_{t-1} (or 0.0 at t=0). a_t (raw[..., 1]) is exact under
        # ZOH and stays as-is.
        def step(
            prev_delta: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            """Carry the previous steering angle forward on stationary steps.

            Args:
                prev_delta: Previous steering angle.
                inputs: Tuple (raw controls, stationary flag).
            Returns:
                Tuple (new steering angle, controls row).
            """
            r, s = inputs
            new_delta = jnp.where(s, prev_delta, r[0])
            out = jnp.stack([new_delta, r[1]])
            return new_delta, out

        init_delta = jnp.asarray(0.0, dtype=traj.dtype)
        _, controls_out = jax.lax.scan(step, init_delta, (raw, stationary))
        return controls_out, stationary.astype(jnp.int32)

    if states.ndim == 2:
        controls, stationary_mask = _per_traj(states)
        aux = {"stationary_frame_count": int(jnp.sum(stationary_mask))}
        return controls, aux

    controls, stationary_mask = jax.vmap(_per_traj)(states)
    aux = {"stationary_frame_count": int(jnp.sum(stationary_mask))}
    return controls, aux
