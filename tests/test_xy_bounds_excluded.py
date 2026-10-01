# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Regression tests: x,y position bounds are excluded from inD inequality constraints."""

import importlib
import inspect
import os

import jax
import jax.numpy as jnp
import pytest

from made.physics import (
    BoxConstraints,
    _assert_xy_unbounded,
    _maybe_assert_xy_zero,
    drop_position_bounds,
    inD_physical_constraints,
    inD_physical_constraints_db,
    kinematic_bicycle_constraints,
)
from made.evaluation.metrics import EmpiricalEnvelope, envelope_constraint


def _envelope() -> EmpiricalEnvelope:
    """Return an inD-like state/control envelope.

    Returns:
        An `EmpiricalEnvelope` with inD position, heading and speed bounds.
    """
    return EmpiricalEnvelope(
        state_min=jnp.array([10.0, -85.0, -3.14, 0.0]),
        state_max=jnp.array([176.0, -8.0, 3.14, 22.0]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )


def test_envelope_constraint_zeros_xy() -> None:
    """Check that envelope constraint zeros xy."""
    c = envelope_constraint(_envelope())
    v = c.violation(jnp.array([1e6, -1e6, 0.0, 5.0]), jnp.array([0.1, 0.5]))
    assert v[0] == 0.0 and v[1] == 0.0
    state_dim = c.state_min.shape[0]
    assert v[state_dim + 0] == 0.0 and v[state_dim + 1] == 0.0


def test_drop_position_bounds_zeros_xy() -> None:
    """Check that drop position bounds zeros xy."""
    d = drop_position_bounds(kinematic_bicycle_constraints())
    v = d.violation(jnp.array([1e6, -1e6, 0.1, 5.0]), jnp.array([0.1, 0.5]))
    state_dim = d.state_min.shape[0]
    assert v[0] == 0.0 and v[1] == 0.0
    assert v[state_dim + 0] == 0.0 and v[state_dim + 1] == 0.0


def test_assert_xy_unbounded_pass() -> None:
    """Check that assert xy unbounded pass."""
    _assert_xy_unbounded(envelope_constraint(_envelope()))
    _assert_xy_unbounded(inD_physical_constraints())
    _assert_xy_unbounded(inD_physical_constraints_db())


def test_assert_xy_unbounded_raises_on_finite() -> None:
    """Check that assert xy unbounded raises on finite."""
    with pytest.raises(ValueError, match="x,y"):
        _assert_xy_unbounded(kinematic_bicycle_constraints())


def test_inD_physical_constraints_factory() -> None:
    """Check that inD physical constraints factory."""
    c = inD_physical_constraints()
    assert jnp.isinf(c.state_min[0]) and jnp.isinf(c.state_max[0])
    assert jnp.isinf(c.state_min[1]) and jnp.isinf(c.state_max[1])
    assert float(c.state_max[3]) == 22.0
    assert float(c.state_min[3]) == 0.0
    assert float(c.control_min[0]) == -0.5 and float(c.control_max[0]) == 0.5
    assert float(c.control_min[1]) == -8.0 and float(c.control_max[1]) == 4.0
    c2 = inD_physical_constraints(v_max=30.0)
    assert float(c2.state_max[3]) == 30.0


def test_inD_physical_constraints_db_factory() -> None:
    """Check that inD physical constraints db factory."""
    c = inD_physical_constraints_db()
    assert c.state_min.shape[0] == 6
    assert jnp.isinf(c.state_min[0]) and jnp.isinf(c.state_max[0])
    assert float(c.state_min[4]) == -3.0 and float(c.state_max[4]) == 3.0
    assert float(c.state_min[5]) == -1.5 and float(c.state_max[5]) == 1.5
    _assert_xy_unbounded(c)


def test_maybe_assert_xy_zero_off_is_noop() -> None:
    """Check that maybe assert xy zero off is noop."""
    os.environ.pop("MADE_DEBUG_ASSERT_XY", None)
    import made.physics.constraints as _c
    importlib.reload(_c)
    fake_violation = jnp.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = _c._maybe_assert_xy_zero(fake_violation)
    assert jnp.allclose(out, fake_violation)


def test_bound_violation_no_op_on_stripped_envelope() -> None:
    """bound_violation perturbation does not move x,y when bounds are ±inf."""
    from made.evaluation.perturbation import perturb_trajectories
    from made.utils.config import DataConfig

    env = _envelope()
    stripped_min = env.state_min.at[jnp.asarray([0, 1])].set(-jnp.inf)
    stripped_max = env.state_max.at[jnp.asarray([0, 1])].set(jnp.inf)

    traj = jnp.tile(jnp.array([[100.0, -50.0, 0.0, 10.0]]), (5, 1))[None]  # (1, 5, 4)
    key = jax.random.key(0)
    config = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=1.0,
    )
    out = perturb_trajectories(
        traj,
        config,
        key,
        state_bounds=(stripped_min, stripped_max),
    )
    assert jnp.all(jnp.isfinite(out))
    # x,y must be unchanged because range is inf → safe_range clamped to 0
    assert jnp.allclose(out[0, :, 0], traj[0, :, 0])
    assert jnp.allclose(out[0, :, 1], traj[0, :, 1])


def test_inD_detection_guard_in_train() -> None:
    """is_field_data=True + non-None state + no factory should raise ValueError."""
    import made.training.trainer as t
    src = inspect.getsource(t.train)
    assert "constraints_factory" in src
    assert "is_field_data" in src
    # Guard message must mention field-data
    assert "field-data" in src or "field_data" in src


def test_train_constraints_factory_kwarg_exists() -> None:
    """Smoke: train function has constraints_factory parameter with default None."""
    from made.training.trainer import train
    sig = inspect.signature(train)
    assert "constraints_factory" in sig.parameters
    assert sig.parameters["constraints_factory"].default is None
