"""Perturbation determinism tests (US-022)."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.evaluation.perturbation import perturb_trajectories
from made.utils.config import DataConfig


def _make_states(seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (4, 10, 4), dtype=jnp.float64)


def test_same_seed_same_metrics():
    """Two calls with the same PRNG key produce byte-identical perturbed arrays."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.1, perturbation_type="gaussian")
    key = jax.random.key(42)

    out1 = perturb_trajectories(states, cfg, key)
    out2 = perturb_trajectories(states, cfg, key)

    assert jnp.array_equal(out1, out2), "Same seed must produce identical perturbation"


def test_same_seed_is_not_identity():
    """With scale > 0 the output differs from the input."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.1, perturbation_type="gaussian")
    out = perturb_trajectories(states, cfg, jax.random.key(42))
    assert not jnp.array_equal(out, states)


def test_different_seed_different_perturbation():
    """Different PRNG keys produce different perturbed arrays."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.1, perturbation_type="gaussian")

    out1 = perturb_trajectories(states, cfg, jax.random.key(0))
    out2 = perturb_trajectories(states, cfg, jax.random.key(1))

    assert not jnp.array_equal(out1, out2), "Different seeds must yield different noise"


def test_zero_scale_returns_original():
    """perturbation_scale=0 is a no-op."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.0)
    out = perturb_trajectories(states, cfg, jax.random.key(0))
    assert jnp.array_equal(out, states)


def test_uniform_perturbation_determinism():
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.2, perturbation_type="uniform")
    key = jax.random.key(7)
    out1 = perturb_trajectories(states, cfg, key)
    out2 = perturb_trajectories(states, cfg, key)
    assert jnp.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# bound_violation tests
# ---------------------------------------------------------------------------

_BV_MIN = jnp.array([-10.0, -10.0, -5.0, -5.0])
_BV_MAX = jnp.array([10.0, 10.0, 5.0, 5.0])


def test_bound_violation_determinism():
    """Same input + bounds → byte-identical output (branch is deterministic; key unused)."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.1, perturbation_type="bound_violation")
    key = jax.random.key(42)
    out1 = perturb_trajectories(states, cfg, key, state_bounds=(_BV_MIN, _BV_MAX))
    out2 = perturb_trajectories(states, cfg, key, state_bounds=(_BV_MIN, _BV_MAX))
    assert jnp.array_equal(out1, out2), "bound_violation must be deterministic given the same inputs"


def test_bound_violation_produces_infeasibility():
    """With scale=0.5 every state dimension is pushed outside [state_min, state_max]."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.5, perturbation_type="bound_violation")
    key = jax.random.key(0)
    out = perturb_trajectories(states, cfg, key, state_bounds=(_BV_MIN, _BV_MAX))
    outside = (out > _BV_MAX) | (out < _BV_MIN)
    assert jnp.all(outside), "Every element must be outside its bound after scale=0.5 push"


def test_bound_violation_zero_scale_is_noop():
    """perturbation_scale=0.0 early-returns the original array unchanged."""
    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.0, perturbation_type="bound_violation")
    key = jax.random.key(0)
    out = perturb_trajectories(states, cfg, key, state_bounds=(_BV_MIN, _BV_MAX))
    assert jnp.array_equal(out, states), "Zero scale must be a no-op"


def test_bound_violation_missing_bounds_raises():
    """Calling without state_bounds raises ValueError mentioning 'state_bounds'."""
    import pytest

    states = _make_states()
    cfg = DataConfig(perturbation_scale=0.1, perturbation_type="bound_violation")
    with pytest.raises(ValueError, match="state_bounds"):
        perturb_trajectories(states, cfg, jax.random.key(0))
