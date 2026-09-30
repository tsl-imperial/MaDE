"""Physics layer."""

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

# Canonical per-system parameter ordering. Must match simulation_data._SIMULATION_REGISTRY.
# Double integrator and unicycle have no physics parameters.
PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "double_integrator": (),
    "unicycle": (),
    "kinematic_bicycle": ("L",),
    "dynamic_bicycle": ("C_f", "C_r", "m", "I_z", "l_f", "l_r"),
}

# Default parameter values per system (used when overrides dict is incomplete).
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

    def __init__(self):
        self._kinematic = KinematicBicycle()

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 1

    def vector_field(self, state: jax.Array, control: jax.Array, params: jax.Array, t: float) -> jax.Array:
        kinematic_state = jnp.asarray([state[0], state[1], state[2], state[3]], dtype=state.dtype)
        kin_dot = self._kinematic.vector_field(kinematic_state, control, params, t)
        return jnp.asarray(
            [kin_dot[0], kin_dot[1], kin_dot[2], kin_dot[3], 0.0, 0.0],
            dtype=state.dtype,
        )

    @property
    def param_scales(self) -> jax.Array:
        return self._kinematic.param_scales

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        kinematic_prev = jnp.asarray([x_prev[0], x_prev[1], x_prev[2], x_prev[3]], dtype=x_prev.dtype)
        kinematic_curr = jnp.asarray([x_curr[0], x_curr[1], x_curr[2], x_curr[3]], dtype=x_curr.dtype)
        return self._kinematic.known_control_prior(kinematic_prev, kinematic_curr, params, dt)


def build_system(name: str) -> tuple[PhysicsModel, ConstraintSet]:
    """Instantiate a physics model and its constraint set by name.

    Returns a (PhysicsModel instance, ConstraintSet instance) tuple.
    Raises ValueError for unknown names.
    """
    if name not in _SYSTEM_REGISTRY:
        supported = sorted(_SYSTEM_REGISTRY)
        raise ValueError(f"Unknown physics system '{name}'. Supported: {supported}")
    model_cls, constraint_factory = _SYSTEM_REGISTRY[name]
    return model_cls(), constraint_factory()


def build_system_for_model(
    true_system: str,
    known_system: str | None,
    envelope=None,
) -> tuple[PhysicsModel, ConstraintSet]:
    """Build the physics/constraints pair used by MaDE for a true/known system pair.

    When ``envelope`` is an :class:`~made.evaluation.metrics.EmpiricalEnvelope`,
    the hardcoded constraint factory is replaced by an empirical-envelope-derived
    ``BoxConstraints``.  Callers that do not pass ``envelope`` (E01, tests) are
    unaffected.
    """
    resolved_known = known_system or true_system
    if true_system == "dynamic_bicycle" and resolved_known == "kinematic_bicycle":
        physics = KinematicBicycleAsDynamicState()
        constraints = dynamic_bicycle_constraints()
    else:
        physics, constraints = build_system(resolved_known)
    if envelope is not None:
        from made.evaluation.metrics import envelope_constraint as _env_con  # lazy — avoids circular import
        constraints = _env_con(envelope)
    return physics, constraints


def resolve_params(system_name: str, overrides: dict[str, float]) -> jax.Array:
    """Convert a parameter override dict to a canonical-order float64 jax.Array.

    Args:
        system_name: One of the supported physics system names.
        overrides: Dict of {param_name: value}. Unknown keys raise ValueError.
                   Missing keys use defaults from _PARAMETER_DEFAULTS.

    Returns:
        jax.Array of shape (param_dim,) with dtype float64 in PARAMETER_ORDER sequence.
    """
    if system_name not in PARAMETER_ORDER:
        supported = sorted(PARAMETER_ORDER)
        raise ValueError(f"Unknown physics system '{system_name}'. Supported: {supported}")
    param_names = PARAMETER_ORDER[system_name]
    defaults = _PARAMETER_DEFAULTS[system_name]

    # Validate override keys
    unknown = set(overrides) - set(param_names)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for system '{system_name}'. "
            f"Valid parameters: {list(param_names)}"
        )

    values = [float(overrides.get(name, defaults[name])) for name in param_names]
    return jnp.asarray(values, dtype=jnp.float64)
