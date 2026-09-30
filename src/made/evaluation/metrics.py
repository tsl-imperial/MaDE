"""Evaluation metrics for feasibility and fidelity.

This module also exposes a real-data metric suite (Experiment 2 / Experiment 3
on the inD dataset) added per ``ralplan-real-data-metrics-v1``:

- :class:`EmpiricalEnvelope` plus :func:`estimate_empirical_envelope`
- :func:`envelope_constraint` — convert an :class:`EmpiricalEnvelope` to a :class:`ConstraintSet`
- :func:`empirical_envelope_violation_rate`
- :func:`empirical_envelope_violation_magnitude`
- :func:`ade`, :func:`fde`
- :func:`gt_normalised_dynamics_residual`
- :func:`estimate_gt_reference_residual`
- :func:`jerk`, :func:`heading_jerk`
- :func:`kinematic_bicycle_inverse_controls`
- :data:`METRIC_VERSION`

Existing helpers (``fidelity``, ``dynamics_violation_*``, ``compute_metrics``)
are intentionally untouched — Experiment 1 paper tables and several
regression tests pin them.
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp

from made.models import AugmentedDynamics
from made.physics import BoxConstraints, ConstraintSet, KinematicBicycle, PhysicsModel

# D1-locked; do not change without user sign-off.
METRIC_VERSION: str = "real-data-v3-gaussian"


def _is_batched(array: jax.Array) -> bool:
    return array.ndim >= 3


def _align_controls_to_states(x: jax.Array, u: jax.Array) -> jax.Array:
    """Return one control per state for inequality metrics."""
    if u.shape[0] == x.shape[0]:
        return u
    return jnp.concatenate([u, u[-1:]], axis=0)


def _violation_channel_slice(constraints, channel: str):
    """Column slice of a `BoxConstraints` violation vector for one channel.

    `BoxConstraints.__call__` concatenates, in this order::

        [state - state_max, state_min - state, control - control_max, control_min - control]

    so the first `2 * state_dim` columns are the STATE-bound components and the remainder are
    the CONTROL-bound ones. Returns `None` when the split cannot be taken -- a constraint set
    that does not expose `state_min` (SpeedNormConstraint raises on it) or a composite whose
    layout is not this one. **A caller that gets `None` must report the combined figure and say
    the components were unavailable, never guess a split.**
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

    The three share ONE implementation deliberately: the components are the
    primary record and the total a derived quantity, and a component computed by a second code
    path is a component that can drift from the total it is supposed to decompose.
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

    **THE TOTAL IS NOT THE SUM OF THE COMPONENTS, on either instrument.** Measured, not
    asserted: with a state violation of 4 and a control violation of 6 on the same step, the
    combined magnitude is 7.2111 -- the L2 norm across the whole violation vector, sqrt(4^2 +
    6^2) -- and not 10. The rate is not additive either, being an "any" over the union of both
    channels, so `rate_all <= rate_state + rate_control` with equality only when the two never
    co-occur on a step.

    This is exactly why the components are recorded as the primary record: a reader who
    reconstructs one channel by subtracting the other from the total gets a wrong number, and
    it will look plausible.
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
    if x.shape[0] <= 1:
        return jnp.asarray(0.0, dtype=x.dtype)
    predicted = jax.vmap(
        lambda x_prev, control: augmented_dynamics.integrate(x_prev, control, params, dt)
    )(x[:-1], u)
    return jnp.mean(jnp.linalg.norm(x[1:] - predicted, axis=-1))


def _trajectory_fidelity(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
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

    **THE SCORING CONVENTION. One convention, everywhere, both panels.** Written
    here at the definition of the metric because it was previously reconstructible only from a
    harness flag, and two panels came to disagree as a result. In short:

        "Clamp only clamps states and does not re-integrate."
        "Scoring for inequality violations considers both states and controls."
        "Controls are always recovered as implied by the states through the inverse known
         model."

    **This amends the control clause above.** The rule is:

    - a row that **emits** controls is scored on **its own emitted controls**. MaDE always emits,
      and so does the EKF/RTS smoother, whose augmented state carries the controls and whose RTS
      pass returns a smoothed estimate of them;
    - a row that emits **none** -- raw, clamp -- is scored on controls **recovered** from its own
      state trajectory through the inverse of the known model.

    No row is scored on states alone. The rationale is comparability: a baseline that does not
    produce controls can only be scored on recovered ones, and a method that does produce them is
    scored on what it actually outputs.

    This is the INEQUALITY convention only. The dynamics residual and control recovery are
    untouched and still use a row's emitted controls where it has them.

    **Consequence, so it is not read as a defect:** clamp fails control bounds it has no
    mechanism to satisfy, because it projects states and has no dynamics step. On the simulated
    panel this moves clamp off exactly 0.0000, which is this convention landing rather than a
    regression.

    This supersedes earlier scoring conventions. The `--ineq-controls`
    flag that used to select between conventions is retired rather than re-defaulted: a
    convention switchable at the command line is how the two panels diverged.

    **NOTE, unrelated to the convention but load-bearing for anyone comparing values:** this
    reduction returns FLOAT32, because `jnp.mean` of a boolean array is float32 even under
    x64. Every other metric here is float64. Not yet changed, because
    a one-character fix moves every published rate at the last ulp.
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
    """Mean violation magnitude over violating steps."""
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
    """Mean one-step violation under the known physics model."""
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
    """Mean one-step violation under the learned augmented dynamics."""
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
    """Mean one-step violation under the ground-truth physics model."""
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
    """Mean per-step distance to ground-truth states."""
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

    Most metrics use ``u_corrected`` — the control that the method under
    evaluation actually produced.  The two dynamics-violation metrics
    (``dynamics_violation_known`` and ``dynamics_violation_true``) ask whether
    ``x_corrected`` is consistent with *some* control signal driven through the
    physics model.  For MaDE variants that infer controls via :math:`\\mathcal{I}`,
    ``u_corrected`` is the right choice because it is the model's own prediction
    of what control produced the corrected state sequence.  For non-MaDE baselines
    (Clamp, MLP, FAB) the method does not infer controls at all and
    ``u_corrected`` is a zero placeholder (see ``apply_baseline_over_trajectory``).
    Comparing the corrected states against a zero-control free-coast integration
    is therefore meaningless — Clamp trivially achieves ~0.2 even though it does
    nothing to enforce dynamics.

    The optional ``u_for_dynamics`` argument overrides the control used *only*
    for these two dynamics-violation calls.  Pass the ground-truth control
    trajectory (from the test split) for non-MaDE baselines so the metric
    measures whether the corrected states are consistent with the actual actuated
    rollout, not a free coast.  Leave ``u_for_dynamics=None`` (the default) for
    MaDE variants to preserve their inferred-control semantics.

    ``dynamics_violation_learned`` uses ``u_corrected`` by default, regardless of
    ``u_for_dynamics``, because for a row with a learned model it measures
    cycle-consistency between the corrected trajectory and that model — an
    internal coherence metric that should stay tied to the model's own control
    output. ``u_for_dyn_learned`` is provided for rows that have **no**
    learned model of their own and borrow the canonical MaDE model's; there
    ``u_corrected`` is a zero placeholder and would make the metric meaningless.
    """
    del u_gt
    u_dyn = u_corrected if u_for_dynamics is None else u_for_dynamics
    # Each metric may take its own control set, because the rules differ.
    #   inequality  -- emitted where the row emits, recovered through the KNOWN model otherwise;
    #   Dyn.-K      -- recovered through the KNOWN model, every row;
    #   Dyn.-T      -- overrides the Dyn.-K rule for the
    #                  dynamic bicycle. Where the known model IS the true model (the three
    #                  fully specified systems), it is recovered through that model and so
    #                  equals Dyn.-K by construction. Where the known model is NOT the true
    #                  model (the underspecified dynamic bicycle), it takes the row's OWN
    #                  emitted controls where the row emits them, and controls recovered
    #                  through the KNOWN (kinematic-bicycle) model's inverse where it emits
    #                  none. The TRUE model's inverse is NOT used on the dynamic bicycle:
    #                  evaluating it on states off its own manifold is what made that column
    #                  explode. The caller supplies the choice via `u_for_dyn_true`;
    #                  `scripts/evaluate.py` is where the rule is applied.
    #   Dyn.-L      -- For a row that HAS a learned model this is its own emitted
    #                  control, an internal cycle-consistency check. A row with NO learned model
    #                  emits nothing to check, so it is scored by borrowing the canonical MaDE
    #                  model for that system and seed: controls recovered through MaDE's
    #                  LEARNED inverse, residual taken under MaDE's learned augmented dynamics,
    #                  with the KNOWN model's inverse as the documented fallback where the
    #                  learned one does not resolve. The caller supplies the controls through
    #                  `u_for_dyn_learned` and the model through `dynamics_learned`, and says
    #                  which recovery it used; `dyn_learned_available=False` still omits the
    #                  metric rather than filling it with a placeholder that would read as a
    #                  measurement.
    # Defaults preserve the previous behaviour exactly, so an un-updated caller is unaffected.
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


# ---------------------------------------------------------------------------
# Real-data (E2 / E3) metric helpers — additive siblings of the E1 helpers
# above. See ralplan-real-data-metrics-v1.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalEnvelope:
    """Per-dimension empirical state and control bounds.

    Estimated from a training-split ground-truth distribution (states +
    inferred controls) by :func:`estimate_empirical_envelope`. Used as a
    dataset-support proxy for inequality metrics on real-data trajectories
    where physical actuator limits are unknown.

    Bounds are NOT physical limits — see ``ralplan-real-data-metrics-v1``
    pre-mortem 8 / Risk 3.
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

    Returns a ``(M, D)`` array, where ``M`` is the total number of valid
    timesteps.  When ``lengths`` is ``None``, every entry is treated as
    valid.  This mirrors the inD pipeline, which pads ragged trajectories
    in a fixed-length tensor.
    """
    if array.ndim == 2:
        return array
    if lengths is None:
        return array.reshape(-1, array.shape[-1])
    # Build a boolean mask (N, T) where t < lengths[i] is True.
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
    """Round lower bounds DOWN and upper bounds UP to multiples of ``rounding``."""
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

    ``states`` may be ``(N, T, state_dim)`` or ``(T, state_dim)``;
    ``controls`` may be ``(N, T-1, control_dim)`` or ``(T-1, control_dim)``.
    When ``lengths`` is provided (inD pipeline), padding entries past
    ``lengths[i]`` are masked out before quantile estimation.

    Bounds are computed at the 1st / 99th percentile (configurable) and
    rounded outward when increments are provided. The helper has no IO and
    is callable from JIT contexts.
    """
    state_lengths = lengths
    flat_states = _flatten_with_lengths(states, state_lengths)
    if flat_states.shape[0] == 0:
        raise ValueError("estimate_empirical_envelope: no valid states after masking.")
    s_low = jnp.quantile(flat_states, lower_quantile, axis=0)
    s_high = jnp.quantile(flat_states, upper_quantile, axis=0)
    s_low, s_high = _round_outward(s_low, s_high, state_rounding)

    # Controls have one fewer step per trajectory than states; subtract 1
    # from each length so the mask aligns with the (T-1) axis.
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

    Used when the caller passes ``u=None`` to the empirical-envelope helpers.
    The control box is set to ``[-inf, +inf]`` so the control-side terms
    contribute zero violation under any zero-control alignment.
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
    """
    return BoxConstraints(
        state_min=envelope.state_min,
        state_max=envelope.state_max,
        control_min=envelope.control_min,
        control_max=envelope.control_max,
    )


def _envelope_constraint(envelope: EmpiricalEnvelope) -> ConstraintSet:
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
    """
    if u is None:
        constraints = _envelope_state_only_constraint(envelope)
        # Pass a zero-control placeholder; control bounds are ±inf so they
        # contribute zero violations.
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

    Thin alias of :func:`fidelity` for real-data tables. ``fidelity`` is
    retained because Experiment 1 imports it; this name is the conventional
    label in trajectory-prediction benchmarks. This is the FULL-STATE norm
    (all state dimensions), kept for E01 and for the ``e05-v3`` locked
    ``results_e05.json``. The printed inD tables use
    :func:`position_ade` / :func:`position_fde` instead, which restrict the
    norm to the (x, y) position indices.
    """
    return fidelity(x_corrected, x_gt)


def _trajectory_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    if x_corrected.shape[0] == 0:
        return jnp.asarray(0.0, dtype=x_corrected.dtype)
    return jnp.linalg.norm(x_corrected[-1] - x_gt[-1])


def fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Final Displacement Error — L2 distance between final timesteps."""
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_fde)(x_corrected, x_gt))
    return _trajectory_fde(x_corrected, x_gt)


def _trajectory_position_ade(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Mean over steps of the Euclidean (x, y) distance -- state indices 0 and 1."""
    return _trajectory_fidelity(x_corrected[..., :2], x_gt[..., :2])


def _trajectory_position_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """Euclidean (x, y) distance at the final step."""
    return _trajectory_fde(x_corrected[..., :2], x_gt[..., :2])


def position_ade(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """inD ADE, position only, metres. Batched ([K,F,D]) or single ([F,D])."""
    if _is_batched(x_corrected):
        return jnp.mean(jax.vmap(_trajectory_position_ade)(x_corrected, x_gt))
    return _trajectory_position_ade(x_corrected, x_gt)


def position_fde(x_corrected: jax.Array, x_gt: jax.Array) -> jax.Array:
    """inD FDE, position only, metres. Batched or single."""
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

    Returns ``method_residual / max(gt_reference_residual, eps) - 1.0``.
    A value near ``0.0`` is the desired calibration target — the method
    matches the dataset's intrinsic known-physics residual. Large positive
    values indicate drift; negative values indicate over-projection toward
    the simplified known model.

    .. warning:: structural-circularity caveat
        When the trajectory was generated by the same KB+L Heun stencil
        that ``dynamics_violation_known`` evaluates against, the numerator
        (and the typical reference residual) is bounded by Heun
        discretisation error; the resulting ratio minus one is uninformative.
        This metric is intended for trajectories that were NOT produced by
        the same KB+L Heun stencil — i.e. real inD trajectories or model
        rollouts that exit the KB+L manifold. See
        ``tests/test_metrics.py::test_gt_normalised_residual_nontrivial_on_non_heun_trajectory``.
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

    For each trajectory ``i`` we compute the per-trajectory residual via
    ``_trajectory_dynamics_violation_known(states_i, controls_i, ...)`` and
    take the mean across trajectories (masked by ``lengths`` when provided
    so padding is excluded).  Returns a Python ``float`` so the value can
    be embedded in JSON metadata without further conversion.
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
    """
    if _is_batched(x):
        return jnp.mean(
            jax.vmap(lambda x_i: _trajectory_heading_jerk(x_i, dt, heading_index))(x)
        )
    return _trajectory_heading_jerk(x, dt, heading_index)


def _perturb_base_vector(envelope: EmpiricalEnvelope) -> jax.Array:
    """Locked-spec base perturbation vector for E02 multiplier sweep.

    State dim 4 (KB ordering x, y, theta, v). Heading dim (idx 2) is ABSOLUTE
    radians (0.1) per locked spec D8/§3.3 — NOT envelope-relative.
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
    """Single-pair perturbation: x_t + m * base * sign."""
    return state_curr + multiplier * base_vec * sign


def _stationary_infeasible_mask(v_avg: jax.Array, threshold: float = 0.5) -> jax.Array:
    """True iff |v_avg| < threshold (KB inverse degenerates near v=0)."""
    return jnp.abs(v_avg) < threshold


def _i_known(
    x_prev: jax.Array,
    x_curr: jax.Array,
    dt: float,
    *,
    wheelbase: float = 2.7,
    stationary_speed_threshold: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """NEW pair-wise primitive — known-physics inverse on a single (x_prev, x_curr).

    Returns (u, is_stationary_carry). u has shape (2,). is_stationary_carry is a
    scalar bool. DISTINCT from `kinematic_bicycle_inverse_controls` (trajectory-
    major) — leave that function untouched.
    """
    # FieldData deliberately: this predicate is used on inD, where the option-D fallback
    # belongs. Using the analytic base here would change a published real-data
    # feasibility metric -- the fallback is scoped to move off
    # SIMULATED data only.
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
    # The four-way split on the PHYSICAL set, recorded as separate fields rather
    # than derived at render time from the combined figure. The components are the primary
    # record; the combined values above are the derived quantity.
    #
    # **The total is NOT the sum of the components** -- the magnitude combines in quadrature
    # and the rate is an "any" over the union -- so reconstructing one channel by subtracting
    # the other from the total gives a wrong number that looks plausible.
    #
    # It is the only way the clamp baseline reads honestly: clamp projects states and has no
    # control field, so its state-bound violation is zero by construction while its
    # control-bound violation is whatever its implied controls score. As one number those two
    # facts cancel into something that describes neither.
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

    Wraps ``KinematicBicycle.known_control_prior`` (`src/made/physics/
    kinematic_bicycle.py:59`) — the Heun-exact inverse for ZOH-constant
    ``(δ, a)`` with a fixed wheelbase ``L``.

    Stationary-frame filter
    -----------------------
    When ``|v_avg| < stationary_speed_threshold`` (default ``0.5 m/s``) the
    closed-form inverse degenerates to ``δ = arctan(L * dθ / (v_avg * dt))``
    near a singularity. To avoid spurious ±π/2 spikes from sensor jitter on
    parked vehicles, we carry forward the previous frame's ``δ`` (and use
    ``δ = 0`` at ``t == 0`` as the boundary case). The auxiliary dict
    returned alongside controls records ``stationary_frame_count``.

    .. warning:: structural-circularity caveat
        This estimator is the exact-against-Heun inverse for the KB stencil
        with wheelbase ``L``. When this estimator is composed with
        :func:`gt_normalised_dynamics_residual` on a trajectory generated by
        the same KB+L Heun stencil, the residual is bounded by Heun
        discretisation error plus the ``v_avg``-regularisation; the
        resulting metric is informative only when the trajectory was NOT
        generated by the same KB+L Heun stencil. inD real-world trajectories
        satisfy this condition; Heun-perfect synthetic test trajectories
        do not.

    Returns
    -------
    controls : ``(N, T-1, 2)`` or ``(T-1, 2)``
        Recovered ``[δ, a]`` per consecutive state pair.
    aux : dict
        ``{"stationary_frame_count": int}`` reporting how many transitions
        fell into the stationary-frame branch (across all trajectories).
    """
    del eps  # reserved for future use; v_avg regularisation is inside known_control_prior
    physics = KinematicBicycle()
    params = jnp.asarray([wheelbase], dtype=jnp.float64)

    def _per_traj(traj: jax.Array) -> tuple[jax.Array, jax.Array]:
        # traj: (T, 4) -> controls (T-1, 2) plus a (T-1,) stationary mask.
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
        def step(prev_delta, inputs):
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
