# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Constraint definitions for each supported system."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import field

import jax
import jax.numpy as jnp

from made.physics.base import ConstraintSet


class BoxConstraints(ConstraintSet):
    """Elementwise box constraints on state and control."""

    state_min: jax.Array
    state_max: jax.Array
    control_min: jax.Array
    control_max: jax.Array

    def __call__(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the raw box inequality values, feasible when all are <= 0.

        Args:
            state: State vector.
            control: Control vector.

        Returns:
            Concatenated state-max, state-min, control-max, control-min residuals.
        """
        return jnp.concatenate(
            [
                state - self.state_max,
                self.state_min - state,
                control - self.control_max,
                self.control_min - control,
            ]
        )


class SpeedNormConstraint(ConstraintSet):
    """Nonlinear speed-norm bound for the double integrator."""

    velocity_indices: tuple[int, int]
    v_max: float

    @property
    def state_min(self) -> jax.Array:
        """Not defined for this constraint.

        Returns:
            Never returns.

        Raises:
            AttributeError: Always; this constraint defines no sampling bounds.
        """
        raise AttributeError("SpeedNormConstraint does not define sampling bounds.")

    @property
    def state_max(self) -> jax.Array:
        """Not defined for this constraint.

        Returns:
            Never returns.

        Raises:
            AttributeError: Always; this constraint defines no sampling bounds.
        """
        raise AttributeError("SpeedNormConstraint does not define sampling bounds.")

    @property
    def control_min(self) -> jax.Array:
        """Not defined for this constraint.

        Returns:
            Never returns.

        Raises:
            AttributeError: Always; this constraint defines no sampling bounds.
        """
        raise AttributeError("SpeedNormConstraint does not define sampling bounds.")

    @property
    def control_max(self) -> jax.Array:
        """Not defined for this constraint.

        Returns:
            Never returns.

        Raises:
            AttributeError: Always; this constraint defines no sampling bounds.
        """
        raise AttributeError("SpeedNormConstraint does not define sampling bounds.")

    def __call__(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the speed-norm inequality value, feasible when <= 0.

        Args:
            state: State vector.
            control: Control vector (unused).

        Returns:
            One-element array ``||v|| - v_max``.
        """
        del control
        velocity = jnp.take(state, jnp.asarray(self.velocity_indices))
        return jnp.array([jnp.linalg.norm(velocity) - self.v_max], dtype=state.dtype)


class CompositeConstraints(ConstraintSet):
    """Constraint set that concatenates several constraint vectors."""

    constraints: tuple[ConstraintSet, ...] = field(default_factory=tuple)

    @property
    def state_min(self) -> jax.Array:
        """Lower state bound of the first constraint.

        Returns:
            The bound array.
        """
        return self.constraints[0].state_min

    @property
    def state_max(self) -> jax.Array:
        """Upper state bound of the first constraint.

        Returns:
            The bound array.
        """
        return self.constraints[0].state_max

    @property
    def control_min(self) -> jax.Array:
        """Lower control bound of the first constraint.

        Returns:
            The bound array.
        """
        return self.constraints[0].control_min

    @property
    def control_max(self) -> jax.Array:
        """Upper control bound of the first constraint.

        Returns:
            The bound array.
        """
        return self.constraints[0].control_max

    def __call__(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the concatenated inequality values of all member constraints.

        Args:
            state: State vector.
            control: Control vector.

        Returns:
            Concatenated constraint values.
        """
        return jnp.concatenate(
            [constraint(state, control) for constraint in self.constraints]
        )


def double_integrator_constraints(v_max: float = 5.0, a_max: float = 2.0) -> ConstraintSet:
    """Default constraints for the double-integrator system.

    Args:
        v_max: Speed bound.
        a_max: Acceleration bound.

    Returns:
        Box plus speed-norm constraints.
    """
    box = BoxConstraints(
        state_min=jnp.array([-10.0, -10.0, -v_max, -v_max]),
        state_max=jnp.array([10.0, 10.0, v_max, v_max]),
        control_min=jnp.array([-a_max, -a_max]),
        control_max=jnp.array([a_max, a_max]),
    )
    return CompositeConstraints((box, SpeedNormConstraint((2, 3), v_max)))


def unicycle_constraints(
    v_max: float = 5.0,
    delta_max: float = 1.0,
    a_max: float = 2.0,
) -> BoxConstraints:
    """Default constraints for the unicycle system.

    Args:
        v_max: Speed bound.
        delta_max: Turn-rate bound.
        a_max: Acceleration bound.

    Returns:
        Box constraints.
    """
    return BoxConstraints(
        state_min=jnp.array([-20.0, -20.0, -jnp.pi, 0.0]),
        state_max=jnp.array([20.0, 20.0, jnp.pi, v_max]),
        control_min=jnp.array([-delta_max, -a_max]),
        control_max=jnp.array([delta_max, a_max]),
    )


def kinematic_bicycle_constraints(
    v_max: float = 13.9,
    delta_max: float = 0.5,
    a_max: float = 3.0,
) -> BoxConstraints:
    """Default constraints for the kinematic bicycle system.

    Args:
        v_max: Speed bound.
        delta_max: Steering-angle bound.
        a_max: Acceleration bound.

    Returns:
        Box constraints.
    """
    return BoxConstraints(
        state_min=jnp.array([-50.0, -50.0, -jnp.pi, 0.0]),
        state_max=jnp.array([50.0, 50.0, jnp.pi, v_max]),
        control_min=jnp.array([-delta_max, -a_max]),
        control_max=jnp.array([delta_max, a_max]),
    )


def dynamic_bicycle_constraints(
    v_max: float = 20.0,
    delta_max: float = 0.5,
    a_max: float = 3.0,
    yaw_rate_max: float = 1.0,
) -> BoxConstraints:
    """Default constraints for the dynamic bicycle system.

    Args:
        v_max: Speed bound.
        delta_max: Steering-angle bound.
        a_max: Acceleration bound.
        yaw_rate_max: Yaw-rate bound.

    Returns:
        Box constraints.
    """
    lateral_v_max = 5.0
    return BoxConstraints(
        state_min=jnp.array([-50.0, -50.0, -jnp.pi, 0.0, -lateral_v_max, -yaw_rate_max]),
        state_max=jnp.array([50.0, 50.0, jnp.pi, v_max, lateral_v_max, yaw_rate_max]),
        control_min=jnp.array([-delta_max, -a_max]),
        control_max=jnp.array([delta_max, a_max]),
    )


def drop_position_bounds(
    box: BoxConstraints,
    *,
    position_indices: Sequence[int] = (0, 1),
) -> BoxConstraints:
    """Return a copy of *box* with x,y bounds widened to ±inf.

    The four supported physics models carry positional coordinates at state
    indices 0 and 1. Use the result only for INEQUALITY computations on the
    inD/field-data path — pass the original box to the simulated-experiment samplers, clamp
    baselines, and bound_violation perturbations that need a finite
    positional range.

    Args:
        box: Box constraints to copy.
        position_indices: State indices of the positional coordinates.

    Returns:
        Box with the positional bounds set to +/-inf.
    """
    inf = jnp.asarray(jnp.inf, dtype=box.state_min.dtype)
    idx = jnp.asarray(position_indices)
    state_min = box.state_min.at[idx].set(-inf)
    state_max = box.state_max.at[idx].set(inf)
    return BoxConstraints(
        state_min=state_min,
        state_max=state_max,
        control_min=box.control_min,
        control_max=box.control_max,
    )


def _assert_xy_unbounded(
    constraints: BoxConstraints,
    *,
    position_indices: Sequence[int] = (0, 1),
) -> None:
    """Raise ValueError if positional rows of *constraints* are finite.

    Guard for the inD/field-data construction surface. Simulated-experiment paths must NOT
    call this — their factories return finite positional bounds by design.

    Args:
        constraints: Box constraints to check.
        position_indices: State indices of the positional coordinates.

    Raises:
        ValueError: If any positional bound is finite.
    """
    idx = list(position_indices)
    if not (
        bool(jnp.all(jnp.isinf(constraints.state_min[jnp.asarray(idx)])))
        and bool(jnp.all(jnp.isinf(constraints.state_max[jnp.asarray(idx)])))
    ):
        raise ValueError(
            "x,y must be ±inf on inD construction path; wrap with "
            "drop_position_bounds or use inD_physical_constraints"
        )


_ASSERT_XY_FLAG_ENABLED = os.environ.get("MADE_DEBUG_ASSERT_XY", "0") == "1"


def _maybe_assert_xy_zero(
    violation: jax.Array,
    *,
    position_indices: Sequence[int] = (0, 1),
    state_dim: int | None = None,
) -> jax.Array:
    """Optionally runtime-check that positional violation rows are zero.

    No-op when ``MADE_DEBUG_ASSERT_XY`` env var is unset. Returns the input
    unchanged. False-positive-free for the simulated experiments: in-box samples produce zero
    at positional rows regardless of finite vs infinite bounds.

    Args:
        violation: Constraint-violation vector.
        position_indices: State indices of the positional coordinates.
        state_dim: State dimension; defaults to half the violation length.

    Returns:
        ``violation``, unchanged.
    """
    if not _ASSERT_XY_FLAG_ENABLED:
        return violation
    if state_dim is None:
        # Best-effort default; callers can pass state_dim for correctness.
        state_dim = violation.shape[-1] // 2
    pos = jnp.asarray(list(position_indices))
    upper = violation[..., pos]
    lower = violation[..., state_dim + pos]
    jax.debug.check(
        jnp.all(upper == 0.0) & jnp.all(lower == 0.0),
        "x,y inequality violation must be zero on inD path; cell constraints "
        "may carry stale finite x,y bounds.",
    )
    return violation


def inD_physical_constraints(
    v_max: float = 22.0,
    delta_max: float = 0.5,
    a_min: float = -8.0,
    a_max: float = 4.0,
) -> BoxConstraints:
    """Feasibility constraints for inD (kinematic-bicycle, 4D state).

    x, y are positional and intentionally UNBOUNDED. Remaining bounds reflect
    typical-vehicle physical limits, NOT the inD camera FOV or the empirical
    envelope's noise-dominated quantiles.

    Args:
        v_max: Speed upper bound.
        delta_max: Steering-angle bound.
        a_min: Minimum acceleration.
        a_max: Maximum acceleration.

    Returns:
        Box constraints with unbounded position.
    """
    inf = jnp.inf
    return BoxConstraints(
        state_min=jnp.array([-inf, -inf, -jnp.pi, 0.0]),
        state_max=jnp.array([+inf, +inf, +jnp.pi, v_max]),
        control_min=jnp.array([-delta_max, a_min]),
        control_max=jnp.array([+delta_max, a_max]),
    )


# Empirical speed bound, measured on the inD TRAINING split: 99th percentile (1.0000% of
# 1,572,732 training frames above it), between the 13.9 clamp box (below the data's 99th
# percentile, clips real vehicles) and the 22.0 scored box (above the 99.9th percentile of
# 19.73). Observed maximum is 27.476.
IND_EMPIRICAL_V_MAX: float = 16.376399999999997


def inD_empirical_constraints(
    v_max: float = IND_EMPIRICAL_V_MAX,
    delta_max: float = 0.5,
    a_min: float = -8.0,
    a_max: float = 4.0,
) -> BoxConstraints:
    """The recorded-data inequality set. One object for clamp AND metric.

    - Speed upper bound is empirical: the 99th percentile of training-split speed. Lower
      bound stays 0 (physical, not empirical), so the set is one-sided.
    - Steering and acceleration are hard-coded at `inD_physical_constraints()`'s values:
      empirical control bounds can't be recovered reliably since controls aren't observed.
    - Position stays dropped (±inf), as in every inD constructor.
    - Heading is a wrap, not a bound; carries no information at ±pi.

    One object passed to both clamp projection and metric, so they can't disagree (earlier
    candidate boxes had clamp project onto 13.9 m/s while the metric scored against 22.0).

    Args:
        v_max: Empirical speed upper bound.
        delta_max: Steering-angle bound.
        a_min: Minimum acceleration.
        a_max: Maximum acceleration.

    Returns:
        Box constraints with unbounded position.
    """
    inf = jnp.inf
    return BoxConstraints(
        state_min=jnp.array([-inf, -inf, -jnp.pi, 0.0]),
        state_max=jnp.array([+inf, +inf, +jnp.pi, v_max]),
        control_min=jnp.array([-delta_max, a_min]),
        control_max=jnp.array([+delta_max, a_max]),
    )


def inD_physical_constraints_db(
    v_max: float = 22.0,
    vy_max: float = 3.0,
    yaw_rate_max: float = 1.5,
    delta_max: float = 0.5,
    a_min: float = -8.0,
    a_max: float = 4.0,
) -> BoxConstraints:
    """Feasibility constraints for inD (dynamic-bicycle, 6D state).

    Args:
        v_max: Speed upper bound.
        vy_max: Lateral-speed bound.
        yaw_rate_max: Yaw-rate bound.
        delta_max: Steering-angle bound.
        a_min: Minimum acceleration.
        a_max: Maximum acceleration.

    Returns:
        Box constraints with unbounded position.
    """
    inf = jnp.inf
    return BoxConstraints(
        state_min=jnp.array([-inf, -inf, -jnp.pi, 0.0, -vy_max, -yaw_rate_max]),
        state_max=jnp.array([+inf, +inf, +jnp.pi, v_max, +vy_max, +yaw_rate_max]),
        control_min=jnp.array([-delta_max, a_min]),
        control_max=jnp.array([+delta_max, a_max]),
    )
