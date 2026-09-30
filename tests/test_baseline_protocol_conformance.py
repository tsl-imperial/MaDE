"""Tests that all baselines conform to the CorrectionBaseline protocol (US-021).

The CorrectionBaseline protocol returns (corrected_x_prev, corrected_x_curr):
both elements are state-dim arrays representing the corrected consecutive pair.
This is distinct from MaDECell which returns (x_corrected, u_corrected).
"""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.baselines import ClampBaseline, FABBaseline, MLPBaseline
from made.physics import DoubleIntegrator, double_integrator_constraints


@pytest.fixture
def di_dims():
    physics = DoubleIntegrator()
    return physics.state_dim, physics.control_dim


@pytest.fixture
def di_constraints():
    return double_integrator_constraints()


@pytest.fixture
def key():
    return jax.random.key(0)


def _dummy_state(state_dim: int) -> jax.Array:
    return jnp.ones(state_dim, dtype=jnp.float64) * 0.1


def test_clamp_returns_state_pair(di_dims, di_constraints):
    state_dim, _ = di_dims
    baseline = ClampBaseline(constraints=di_constraints)
    x_prev = _dummy_state(state_dim)
    x_curr = _dummy_state(state_dim) * 2.0
    out_prev, out_curr = baseline.correct_pair(x_prev, x_curr)
    assert out_prev.shape == (state_dim,)
    assert out_curr.shape == (state_dim,)


def test_clamp_clips_to_bounds(di_dims, di_constraints):
    state_dim, _ = di_dims
    baseline = ClampBaseline(constraints=di_constraints)
    x_huge = jnp.ones(state_dim, dtype=jnp.float64) * 1e6
    out_prev, out_curr = baseline.correct_pair(x_huge, x_huge)
    assert jnp.all(out_prev <= di_constraints.state_max + 1e-9)
    assert jnp.all(out_curr <= di_constraints.state_max + 1e-9)


def test_mlp_returns_state_pair(di_dims, key):
    state_dim, _ = di_dims
    baseline = MLPBaseline(state_dim=state_dim, hidden=(16, 16), key=key)
    x_prev = _dummy_state(state_dim)
    x_curr = _dummy_state(state_dim)
    out_prev, out_curr = baseline.correct_pair(x_prev, x_curr)
    assert out_prev.shape == (state_dim,)
    assert out_curr.shape == (state_dim,)


def test_fab_returns_state_pair(di_dims, key):
    state_dim, _ = di_dims
    baseline = FABBaseline(state_dim=state_dim, latent_dim=8, hidden=(16, 16), key=key)
    x_prev = _dummy_state(state_dim)
    x_curr = _dummy_state(state_dim)
    out_prev, out_curr = baseline.correct_pair(x_prev, x_curr)
    assert out_prev.shape == (state_dim,)
    assert out_curr.shape == (state_dim,)


@pytest.mark.parametrize(
    "baseline_name",
    ["ClampBaseline", "MLPBaseline", "FABBaseline"],
)
def test_protocol_both_elements_are_state_dim(baseline_name, di_dims, di_constraints, key):
    """Both return elements must have state_dim shape (not control_dim)."""
    state_dim, _ = di_dims
    x_prev = _dummy_state(state_dim)
    x_curr = _dummy_state(state_dim)

    if baseline_name == "ClampBaseline":
        baseline = ClampBaseline(constraints=di_constraints)
    elif baseline_name == "MLPBaseline":
        baseline = MLPBaseline(state_dim=state_dim, hidden=(16, 16), key=key)
    else:
        baseline = FABBaseline(state_dim=state_dim, latent_dim=8, hidden=(16, 16), key=key)

    out0, out1 = baseline.correct_pair(x_prev, x_curr)
    assert out0.shape == (state_dim,), f"{baseline_name} first element shape mismatch"
    assert out1.shape == (state_dim,), f"{baseline_name} second element shape mismatch"


def test_all_baselines_return_finite_values(di_dims, di_constraints, key):
    state_dim, _ = di_dims
    x_prev = _dummy_state(state_dim)
    x_curr = _dummy_state(state_dim)
    baselines = [
        ClampBaseline(constraints=di_constraints),
        MLPBaseline(state_dim=state_dim, hidden=(16, 16), key=key),
        FABBaseline(state_dim=state_dim, latent_dim=8, hidden=(16, 16), key=key),
    ]
    for baseline in baselines:
        out0, out1 = baseline.correct_pair(x_prev, x_curr)
        assert jnp.all(jnp.isfinite(out0)), f"{type(baseline).__name__} out0 not finite"
        assert jnp.all(jnp.isfinite(out1)), f"{type(baseline).__name__} out1 not finite"
