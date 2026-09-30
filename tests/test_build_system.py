"""Tests for build_system factory."""

import pytest
import jax
import jax.numpy as jnp

from made.models import MaDECell
from made.physics import (
    build_system,
    build_system_for_model,
    ConstraintSet,
    DoubleIntegrator,
    DynamicBicycle,
    KinematicBicycleAsDynamicState,
    KinematicBicycle,
    PhysicsModel,
    Unicycle,
    resolve_params,
)
from made.utils.config import CorrectorConfig, ModelConfig


@pytest.mark.parametrize(
    "name,expected_model_type",
    [
        ("double_integrator", DoubleIntegrator),
        ("unicycle", Unicycle),
        ("kinematic_bicycle", KinematicBicycle),
        ("dynamic_bicycle", DynamicBicycle),
    ],
)
def test_build_system_returns_correct_types(name, expected_model_type):
    physics, constraints = build_system(name)
    assert isinstance(physics, expected_model_type)
    assert isinstance(physics, PhysicsModel)


def test_build_system_returns_constraint_set():
    for name in ("double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle"):
        _, constraints = build_system(name)
        assert isinstance(constraints, ConstraintSet)


def test_build_system_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown physics system"):
        build_system("nonexistent_system")


def test_build_system_unknown_lists_supported():
    with pytest.raises(ValueError) as exc_info:
        build_system("bad_name")
    msg = str(exc_info.value)
    for name in ("double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle"):
        assert name in msg


def test_build_system_returns_fresh_instances():
    """Each call returns a new instance, not a cached singleton."""
    physics1, _ = build_system("double_integrator")
    physics2, _ = build_system("double_integrator")
    assert physics1 is not physics2


def test_dynamic_true_kinematic_known_is_dimension_compatible():
    physics, constraints = build_system_for_model("dynamic_bicycle", "kinematic_bicycle")
    assert isinstance(physics, KinematicBicycleAsDynamicState)
    assert physics.state_dim == 6
    assert physics.control_dim == 2
    assert physics.param_dim == 1
    assert isinstance(constraints, ConstraintSet)

    cell = MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(train_steps=1),
        key=jax.random.key(0),
    )
    x_corr, u_corr = cell(
        jnp.zeros((6,)),
        jnp.ones((6,)) * 0.1,
        resolve_params("kinematic_bicycle", {}),
        0.1,
    )
    assert x_corr.shape == (6,)
    assert u_corr.shape == (2,)
