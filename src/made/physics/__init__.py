# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Physics layer."""

from typing import Any

import jax
import jax.numpy as jnp

from made.physics.base import ConstraintSet, PhysicsModel
from made.physics.constraints import (
    BoxConstraints,
    CompositeConstraints,
    SpeedNormConstraint,
    _assert_xy_unbounded,
    _maybe_assert_xy_zero,
    double_integrator_constraints,
    drop_position_bounds,
    dynamic_bicycle_constraints,
    inD_physical_constraints,
    inD_physical_constraints_db,
    kinematic_bicycle_constraints,
    unicycle_constraints,
)
from made.physics.double_integrator import DoubleIntegrator
from made.physics.dynamic_bicycle import DynamicBicycle
from made.physics.kinematic_bicycle import (
    KinematicBicycle,
    KinematicBicycleFieldData,
)
from made.physics.simulator import generate_trajectories
from made.physics.unicycle import Unicycle

__all__ = [
    "BoxConstraints",
    "CompositeConstraints",
    "ConstraintSet",
    "DoubleIntegrator",
    "DynamicBicycle",
    "KinematicBicycle",
    "KinematicBicycleAsDynamicState",
    "KinematicBicycleFieldData",
    "PARAMETER_ORDER",
    "PhysicsModel",
    "SpeedNormConstraint",
    "Unicycle",
    "_assert_xy_unbounded",
    "_maybe_assert_xy_zero",
    "build_system",
    "build_system_for_model",
    "double_integrator_constraints",
    "drop_position_bounds",
    "dynamic_bicycle_constraints",
    "generate_trajectories",
    "inD_physical_constraints",
    "inD_physical_constraints_db",
    "kinematic_bicycle_constraints",
    "resolve_params",
    "unicycle_constraints",
]

# Must match simulation_data._SIMULATION_REGISTRY. Double integrator and unicycle
# have no physics parameters.
PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "double_integrator": (),
    "unicycle": (),
    "kinematic_bicycle": ("L",),
    "dynamic_bicycle": ("C_f", "C_r", "m", "I_z", "l_f", "l_r"),
}

# Used when the overrides dict is incomplete.
_PARAMETER_DEFAULTS: dict[str, dict[str, float]] = {
    "double_integrator": {},
    "unicycle": {},
    "kinematic_bicycle": {"L": 2.7},
    "dynamic_bicycle": {
        "C_f": 19000.0,
        "C_r": 20000.0,
        "m": 1500.0,
        "I_z": 3000.0,
        "l_f": 1.2,
        "l_r": 1.5,
    },
}

_SYSTEM_REGISTRY: dict[str, tuple] = {
    "double_integrator": (DoubleIntegrator, double_integrator_constraints),
    "unicycle": (Unicycle, unicycle_constraints),
    "kinematic_bicycle": (KinematicBicycle, kinematic_bicycle_constraints),
    "dynamic_bicycle": (DynamicBicycle, dynamic_bicycle_constraints),
}


class KinematicBicycleAsDynamicState(PhysicsModel):
    """Kinematic-bicycle known physics over the dynamic-bicycle state layout."""

    _kinematic: KinematicBicycle

    def __init__(self) -> None:
        """Wrap a ``KinematicBicycle``."""
        self._kinematic = KinematicBicycle()

    @property
    def state_dim(self) -> int:
        """Dimension of the dynamic-bicycle state vector.

        Returns:
            6.
        """
        return 6

    @property
    def control_dim(self) -> int:
        """Dimension of the control vector.

        Returns:
            2.
        """
        return 2

    @property
    def param_dim(self) -> int:
        """Dimension of the parameter vector.

        Returns:
            1.
        """
        return 1

    def vector_field(
        self, state: jax.Array, control: jax.Array, params: jax.Array, t: float
    ) -> jax.Array:
        """Kinematic vector field padded with zero lateral-velocity and yaw-rate derivatives.

        Args:
            state: Dynamic-bicycle state vector.
            control: Control vector.
            params: Physical parameters.
            t: Time.

        Returns:
            State derivative of length 6.
        """
        kinematic_state = jnp.asarray([state[0], state[1], state[2], state[3]], dtype=state.dtype)
        kin_dot = self._kinematic.vector_field(kinematic_state, control, params, t)
        return jnp.asarray(
            [kin_dot[0], kin_dot[1], kin_dot[2], kin_dot[3], 0.0, 0.0],
            dtype=state.dtype,
        )

    @property
    def param_scales(self) -> jax.Array:
        """Characteristic parameter scales of the wrapped kinematic bicycle.

        Returns:
            Parameter scales.
        """
        return self._kinematic.param_scales

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Kinematic control prior computed on the first four state components.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.
            dt: Step length.

        Returns:
            Control estimate.
        """
        kinematic_prev = jnp.asarray(
            [x_prev[0], x_prev[1], x_prev[2], x_prev[3]], dtype=x_prev.dtype
        )
        kinematic_curr = jnp.asarray(
            [x_curr[0], x_curr[1], x_curr[2], x_curr[3]], dtype=x_curr.dtype
        )
        return self._kinematic.known_control_prior(kinematic_prev, kinematic_curr, params, dt)


def build_system(name: str) -> tuple[PhysicsModel, ConstraintSet]:
    """Instantiate a physics model and its constraint set by name.

    Args:
        name: Registered system name.

    Returns:
        Tuple ``(physics, constraints)``.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in _SYSTEM_REGISTRY:
        supported = sorted(_SYSTEM_REGISTRY)
        raise ValueError(f"Unknown physics system '{name}'. Supported: {supported}")
    model_cls, constraint_factory = _SYSTEM_REGISTRY[name]
    return model_cls(), constraint_factory()


def build_system_for_model(
    true_system: str,
    known_system: str | None,
    envelope: Any = None,
) -> tuple[PhysicsModel, ConstraintSet]:
    """Build the physics/constraints pair used by MaDE for a true/known system pair.

    If ``envelope`` is an :class:`~made.evaluation.metrics.EmpiricalEnvelope`, the
    hardcoded constraint factory is replaced by an empirical-envelope-derived
    ``BoxConstraints``.

    Args:
        true_system: Name of the system that generated the data.
        known_system: Name of the known-physics system; defaults to ``true_system``.
        envelope: Optional ``EmpiricalEnvelope`` replacing the hardcoded constraints.

    Returns:
        Tuple ``(physics, constraints)``.
    """
    resolved_known = known_system or true_system
    if true_system == "dynamic_bicycle" and resolved_known == "kinematic_bicycle":
        physics = KinematicBicycleAsDynamicState()
        constraints = dynamic_bicycle_constraints()
    else:
        physics, constraints = build_system(resolved_known)
    if envelope is not None:
        from made.evaluation.metrics import envelope_constraint as _env_con  # avoid circular import
        constraints = _env_con(envelope)
    return physics, constraints


def resolve_params(system_name: str, overrides: dict[str, float]) -> jax.Array:
    """Convert a parameter override dict to a canonical-order float64 jax.Array.

    Missing keys use defaults from ``_PARAMETER_DEFAULTS``. Returns shape (param_dim,)
    in ``PARAMETER_ORDER`` sequence.

    Args:
        system_name: Registered system name.
        overrides: Parameter values overriding the defaults.

    Returns:
        Parameter vector.

    Raises:
        ValueError: If the system or any override name is unknown.
    """
    if system_name not in PARAMETER_ORDER:
        supported = sorted(PARAMETER_ORDER)
        raise ValueError(f"Unknown physics system '{system_name}'. Supported: {supported}")
    param_names = PARAMETER_ORDER[system_name]
    defaults = _PARAMETER_DEFAULTS[system_name]

    unknown = set(overrides) - set(param_names)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for system '{system_name}'. "
            f"Valid parameters: {list(param_names)}"
        )

    values = [float(overrides.get(name, defaults[name])) for name in param_names]
    return jnp.asarray(values, dtype=jnp.float64)
