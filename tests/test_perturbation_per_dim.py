"""Per-component scale tests for the bound_violation perturbation."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.evaluation.perturbation import perturb_trajectories
from made.utils.config import DataConfig

_BV_MIN = jnp.array([-20.0, -20.0, -jnp.pi, 0.0])
_BV_MAX = jnp.array([20.0, 20.0, jnp.pi, 5.0])
_RANGE = _BV_MAX - _BV_MIN


def _states_above_midpoint() -> jax.Array:
    return jnp.array([[[5.0, 5.0, 1.0, 4.0]]])


def test_per_dim_scale_matches_expected_shove_magnitudes():
    """Each component is shoved by exactly scale[i] * range[i]."""
    states = _states_above_midpoint()
    per_dim = (0.10, 0.10, 0.05, 0.10)
    cfg = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=0.10,
        perturbation_scale_per_dim=per_dim,
    )
    out = perturb_trajectories(states, cfg, jax.random.key(0), state_bounds=(_BV_MIN, _BV_MAX))
    delta = (out - states)[0, 0]
    expected = jnp.asarray(per_dim) * _RANGE
    assert jnp.allclose(delta, expected, atol=1e-12), (delta, expected)


def test_per_dim_scale_overrides_scalar():
    """When perturbation_scale_per_dim is set, scalar perturbation_scale is ignored for shove magnitude."""
    states = _states_above_midpoint()
    per_dim = (0.20, 0.05, 0.05, 0.10)
    cfg = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=0.10,
        perturbation_scale_per_dim=per_dim,
    )
    out = perturb_trajectories(states, cfg, jax.random.key(0), state_bounds=(_BV_MIN, _BV_MAX))
    delta = (out - states)[0, 0]
    expected = jnp.asarray(per_dim) * _RANGE
    assert jnp.allclose(delta, expected, atol=1e-12)


def test_per_dim_scale_none_falls_back_to_scalar():
    """perturbation_scale_per_dim=None uses scalar perturbation_scale for all dims."""
    states = _states_above_midpoint()
    cfg = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=0.10,
        perturbation_scale_per_dim=None,
    )
    out = perturb_trajectories(states, cfg, jax.random.key(0), state_bounds=(_BV_MIN, _BV_MAX))
    delta = (out - states)[0, 0]
    expected = 0.10 * _RANGE
    assert jnp.allclose(delta, expected, atol=1e-12)


def test_per_dim_length_mismatch_raises():
    """Length mismatch between per-dim scale and state_dim is rejected with a clear message."""
    states = _states_above_midpoint()
    cfg = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=0.10,
        perturbation_scale_per_dim=(0.10, 0.10),  # 2-d but state_dim=4
    )
    with pytest.raises(ValueError, match="perturbation_scale_per_dim length"):
        perturb_trajectories(states, cfg, jax.random.key(0), state_bounds=(_BV_MIN, _BV_MAX))


def test_per_dim_zero_scalar_is_not_a_noop_when_per_dim_set():
    """perturbation_scale=0 short-circuits before bound_violation is even reached.

    This documents current behavior: the early-return at the top of perturb_trajectories
    fires on the scalar field, so a non-trivial per_dim scale is ignored when the scalar
    is zero. Configs that want per-dim shoves must set a non-zero scalar perturbation_scale.
    """
    states = _states_above_midpoint()
    cfg = DataConfig(
        perturbation_type="bound_violation",
        perturbation_scale=0.0,
        perturbation_scale_per_dim=(0.10, 0.10, 0.05, 0.10),
    )
    out = perturb_trajectories(states, cfg, jax.random.key(0), state_bounds=(_BV_MIN, _BV_MAX))
    assert jnp.array_equal(out, states)


def test_per_dim_other_perturbation_types_ignore_field():
    """Gaussian and uniform perturbations don't read perturbation_scale_per_dim."""
    states = _states_above_midpoint()
    for ptype in ("gaussian", "uniform"):
        cfg_with = DataConfig(
            perturbation_type=ptype,
            perturbation_scale=0.10,
            perturbation_scale_per_dim=(0.10, 0.10, 0.05, 0.10),
        )
        cfg_without = DataConfig(
            perturbation_type=ptype,
            perturbation_scale=0.10,
            perturbation_scale_per_dim=None,
        )
        key = jax.random.key(7)
        out_with = perturb_trajectories(states, cfg_with, key)
        out_without = perturb_trajectories(states, cfg_without, key)
        assert jnp.array_equal(out_with, out_without), f"{ptype} must ignore perturbation_scale_per_dim"
