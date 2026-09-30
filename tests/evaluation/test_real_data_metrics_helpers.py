"""Unit tests for E02 metrics helper primitives (§3.1).

Covers:
- METRIC_VERSION constant
- _perturb_base_vector: shape, dtype, heading invariance, v-range scaling
- _perturb_pair: pure additivity (zero multiplier identity, nonzero correctness)
- _stationary_infeasible_mask: threshold boundary behaviour
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from made.evaluation.metrics import (
    METRIC_VERSION,
    EmpiricalEnvelope,
    _perturb_base_vector,
    _perturb_pair,
    _stationary_infeasible_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(range_v: float = 20.0) -> EmpiricalEnvelope:
    return EmpiricalEnvelope(
        state_min=jnp.array([0.0, 0.0, -jnp.pi, 0.0]),
        state_max=jnp.array([10.0, 10.0, jnp.pi, range_v]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )


# ---------------------------------------------------------------------------
# METRIC_VERSION
# ---------------------------------------------------------------------------

def test_metric_version():
    assert METRIC_VERSION == "real-data-v3-gaussian"


# ---------------------------------------------------------------------------
# _perturb_base_vector
# ---------------------------------------------------------------------------

def test_perturb_base_vector_shape_and_dtype():
    env = _make_env(20.0)
    base = _perturb_base_vector(env)
    assert base.shape == (4,), f"expected shape (4,), got {base.shape}"
    assert base.dtype == jnp.float64, f"expected float64, got {base.dtype}"


def test_perturb_base_vector_heading_is_abs_0p1_regardless_of_range_v():
    """Heading dim (idx 2) must be exactly 0.1 rad for both envelope sizes."""
    env_5 = _make_env(range_v=5.0)
    env_50 = _make_env(range_v=50.0)
    base_5 = _perturb_base_vector(env_5)
    base_50 = _perturb_base_vector(env_50)
    assert float(base_5[2]) == pytest.approx(0.1, abs=1e-12), (
        f"heading base with range_v=5: expected 0.1, got {float(base_5[2])}"
    )
    assert float(base_50[2]) == pytest.approx(0.1, abs=1e-12), (
        f"heading base with range_v=50: expected 0.1, got {float(base_50[2])}"
    )
    # The heading values must be bitwise identical (same absolute constant)
    assert float(base_5[2]) == float(base_50[2])


def test_perturb_base_vector_v_dim_scales_with_envelope():
    """Velocity dim (idx 3) must differ between the two envelope sizes."""
    env_5 = _make_env(range_v=5.0)
    env_50 = _make_env(range_v=50.0)
    base_5 = _perturb_base_vector(env_5)
    base_50 = _perturb_base_vector(env_50)
    # base[3] = 0.1 * range_v  (range_v differs so values must differ)
    assert float(base_5[3]) == pytest.approx(0.1 * 5.0, abs=1e-12)
    assert float(base_50[3]) == pytest.approx(0.1 * 50.0, abs=1e-12)
    assert float(base_5[3]) != float(base_50[3])


def test_perturb_base_vector_infinite_dims_zero():
    """Dimensions with infinite range (x, y on inD) contribute zero."""
    env = EmpiricalEnvelope(
        state_min=jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, 0.0]),
        state_max=jnp.array([jnp.inf, jnp.inf, jnp.pi, 20.0]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )
    base = _perturb_base_vector(env)
    assert float(base[0]) == pytest.approx(0.0, abs=1e-12), "x dim should be 0 (inf range)"
    assert float(base[1]) == pytest.approx(0.0, abs=1e-12), "y dim should be 0 (inf range)"
    assert float(base[2]) == pytest.approx(0.1, abs=1e-12), "heading must still be 0.1"


# ---------------------------------------------------------------------------
# _perturb_pair
# ---------------------------------------------------------------------------

def test_perturb_pair_zero_multiplier_is_identity():
    """_perturb_pair(x, 0, b, s) == x for any b, s."""
    x = jnp.array([1.0, 2.0, 0.5, 8.0])
    b = jnp.array([1.0, 1.0, 0.1, 2.0])
    s = jnp.array([1.0, -1.0, 1.0, -1.0])
    result = _perturb_pair(x, jnp.asarray(0.0), b, s)
    assert jnp.allclose(result, x, atol=1e-14), f"expected identity, got diff {result - x}"


def test_perturb_pair_nonzero_additivity():
    """_perturb_pair(x, m, b, s) == x + m*b*s exactly."""
    x = jnp.array([3.0, 4.0, -0.2, 10.0])
    b = jnp.array([2.0, 2.0, 0.1, 2.2])
    s = jnp.array([1.0, -1.0, 1.0, 1.0])
    m = jnp.asarray(1.5)
    expected = x + m * b * s
    result = _perturb_pair(x, m, b, s)
    assert jnp.allclose(result, expected, atol=1e-14), (
        f"expected {expected}, got {result}"
    )


def test_perturb_pair_negative_multiplier():
    """Negative multiplier flips the perturbation direction."""
    x = jnp.array([0.0, 0.0, 0.0, 5.0])
    b = jnp.array([1.0, 1.0, 0.1, 2.0])
    s = jnp.array([1.0, 1.0, 1.0, 1.0])
    m_pos = jnp.asarray(2.0)
    m_neg = jnp.asarray(-2.0)
    r_pos = _perturb_pair(x, m_pos, b, s)
    r_neg = _perturb_pair(x, m_neg, b, s)
    # r_pos + r_neg should equal 2*x
    assert jnp.allclose(r_pos + r_neg, 2.0 * x, atol=1e-14)


# ---------------------------------------------------------------------------
# _stationary_infeasible_mask
# ---------------------------------------------------------------------------

def test_stationary_mask_below_threshold():
    """|v_avg| strictly below 0.5 → True."""
    assert bool(_stationary_infeasible_mask(jnp.asarray(0.0)))
    assert bool(_stationary_infeasible_mask(jnp.asarray(0.1)))
    assert bool(_stationary_infeasible_mask(jnp.asarray(0.49)))
    assert bool(_stationary_infeasible_mask(jnp.asarray(-0.3)))


def test_stationary_mask_at_threshold():
    """|v_avg| == 0.5 → False (not strictly below)."""
    assert not bool(_stationary_infeasible_mask(jnp.asarray(0.5)))
    assert not bool(_stationary_infeasible_mask(jnp.asarray(-0.5)))


def test_stationary_mask_above_threshold():
    """|v_avg| > 0.5 → False."""
    assert not bool(_stationary_infeasible_mask(jnp.asarray(1.0)))
    assert not bool(_stationary_infeasible_mask(jnp.asarray(10.0)))
    assert not bool(_stationary_infeasible_mask(jnp.asarray(-0.6)))


def test_stationary_mask_custom_threshold():
    """Custom threshold is respected."""
    assert bool(_stationary_infeasible_mask(jnp.asarray(0.4), threshold=1.0))
    assert not bool(_stationary_infeasible_mask(jnp.asarray(1.5), threshold=1.0))
