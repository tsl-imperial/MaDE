"""DynamicBicycle physics contract tests (US-015)."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.physics import (
    DynamicBicycle,
    dynamic_bicycle_constraints,
    generate_trajectories,
    resolve_params,
)

_PARAMS = resolve_params("dynamic_bicycle", {})
_DB = DynamicBicycle()
_BENIGN_STATE = jnp.array([0.0, 0.0, 0.0, 5.0, 0.1, 0.01], dtype=jnp.float64)
_BENIGN_CONTROL = jnp.array([0.05, 0.2], dtype=jnp.float64)


def test_vector_field_shape():
    out = _DB.vector_field(_BENIGN_STATE, _BENIGN_CONTROL, _PARAMS, 0.0)
    assert out.shape == (6,)
    assert jnp.all(jnp.isfinite(out))


def test_vmap_over_batch():
    batch_size = 8
    states = jnp.broadcast_to(_BENIGN_STATE[None], (batch_size, 6))
    controls = jnp.broadcast_to(_BENIGN_CONTROL[None], (batch_size, 2))
    out = jax.vmap(lambda s, c: _DB.vector_field(s, c, _PARAMS, 0.0))(states, controls)
    assert out.shape == (batch_size, 6)
    assert jnp.all(jnp.isfinite(out))


def test_finite_diff_vs_autodiff():
    """Jacobian via finite differences vs autodiff (rel tol 1e-4) at non-zero v_x."""
    def f(s: jax.Array) -> jax.Array:
        return _DB.vector_field(s, _BENIGN_CONTROL, _PARAMS, 0.0)

    jac_auto = jax.jacobian(f)(_BENIGN_STATE)

    eps = 1e-5
    rows = []
    for i in range(6):
        e = jnp.zeros(6, dtype=jnp.float64).at[i].set(eps)
        rows.append((f(_BENIGN_STATE + e) - f(_BENIGN_STATE - e)) / (2.0 * eps))
    jac_fd = jnp.stack(rows, axis=1)  # (6, 6)

    norm = jnp.linalg.norm(jac_auto)
    rel_err = jnp.linalg.norm(jac_auto - jac_fd) / (norm + 1e-10)
    assert float(rel_err) < 1e-4, f"Relative Jacobian error {float(rel_err):.2e} exceeds 1e-4"


def test_simulator_smoke():
    """Generate 4 feasible DynamicBicycle trajectories of length 8."""
    constraints = dynamic_bicycle_constraints()
    states, controls = generate_trajectories(
        _DB, constraints, 4, 8, 0.05, jax.random.key(0), _PARAMS
    )
    assert states.shape == (4, 8, 6)
    assert controls.shape == (4, 7, 2)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(jnp.isfinite(controls))


def test_simulator_runbook_dt_regression():
    """Generate DynamicBicycle trajectories at the default runbook dt."""
    constraints = dynamic_bicycle_constraints()
    states, controls = generate_trajectories(
        _DB, constraints, 8, 16, 0.1, jax.random.key(0), _PARAMS
    )
    assert states.shape == (8, 16, 6)
    assert controls.shape == (8, 15, 2)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(jnp.isfinite(controls))
