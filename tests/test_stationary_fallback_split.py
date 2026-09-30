"""The option-D stationary fallback applies to FIELD DATA ONLY.

Three conventions coexist deliberately. These tests pin which is which, because the
difference is silent: both classes return a well-formed control and only differ on
transitions below 0.5 m/s, which is where the published kinematic-bicycle result moved by
three orders.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from made.physics import KinematicBicycle, KinematicBicycleFieldData, build_system_for_model

DT = 0.1
PARAMS = jnp.asarray([2.7])


def _pair(v_prev, v_curr, dtheta=0.02):
    x_prev = jnp.asarray([0.0, 0.0, 0.0, v_prev])
    x_curr = jnp.asarray([0.0, 0.0, dtheta, v_curr])
    return x_prev, x_curr


def test_below_threshold_base_is_analytic_fielddata_is_zero():
    """The whole point of the split: same input, different delta, only when slow."""
    xp, xc = _pair(0.2, 0.2)          # v_avg = 0.2 < 0.5
    base = KinematicBicycle().known_control_prior(xp, xc, PARAMS, DT)
    field = KinematicBicycleFieldData().known_control_prior(xp, xc, PARAMS, DT)
    assert float(field[0]) == 0.0, "field data must zero delta below the threshold"
    assert abs(float(base[0])) > 1e-3, "base must keep the analytic arctan"
    np.testing.assert_allclose(float(base[1]), float(field[1]),
                               err_msg="acceleration is ZOH-exact and must never diverge")


def test_above_threshold_the_two_agree_exactly():
    xp, xc = _pair(8.0, 8.2)          # v_avg = 8.1 > 0.5
    base = KinematicBicycle().known_control_prior(xp, xc, PARAMS, DT)
    field = KinematicBicycleFieldData().known_control_prior(xp, xc, PARAMS, DT)
    np.testing.assert_array_equal(np.asarray(base), np.asarray(field))


def test_threshold_boundary_is_strict_less_than():
    for v, expect_zero in ((0.499, True), (0.5, False), (0.501, False)):
        xp, xc = _pair(v, v)
        d = float(KinematicBicycleFieldData().known_control_prior(xp, xc, PARAMS, DT)[0])
        assert (d == 0.0) is expect_zero, f"v_avg={v} gave delta={d}"


@pytest.mark.parametrize("system", ["kinematic_bicycle", "double_integrator", "unicycle"])
def test_simulated_construction_never_yields_the_fallback(system):
    """E01 builds through build_system_for_model, which must hand back the ANALYTIC base.

    This is the regression that matters: simulated data has no sensor jitter, so a guard
    against jitter must not reach it. (Separately: the fallback IS what shifted the published
    simulated numbers on the seeds that ran under it.)
    """
    physics, _ = build_system_for_model(system, None)
    assert not isinstance(physics, KinematicBicycleFieldData), (
        f"{system} must not receive the field-data fallback"
    )


def test_underspecified_db_known_model_is_not_field_data():
    """DB-underspecified uses KB as its KNOWN model and is simulated, so it takes the base.

    It measured bit-identical to published because 0.00% of its transitions fall below the
    threshold -- a data property, not a code boundary. This pins the code boundary too.
    """
    physics, _ = build_system_for_model("dynamic_bicycle", "kinematic_bicycle")
    assert not isinstance(physics, KinematicBicycleFieldData)
    inner = getattr(physics, "_kinematic", None)
    assert inner is None or not isinstance(inner, KinematicBicycleFieldData)
