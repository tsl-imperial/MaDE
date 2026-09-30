"""Tests for `return_diagnostics` threaded through `apply_made_trajectory_with_controls`.

`apply_made_trajectory_with_controls` applies a frozen `MaDECell` autoregressively
via `jax.lax.scan`. With `return_diagnostics=True` (only permitted under
`correction_mode="eval_adaptive"`), each per-timestep `CorrectorDiagnostics`
(iteration count + cap-hit flag) must be emitted as a scan output and stacked
into arrays of shape `[T]`, matching how `states`/`controls` are stacked. The
default (`return_diagnostics=False`, the implicit default for every existing
caller) must keep returning the plain `(states, controls)` 2-tuple.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from made.models import MaDECell
from made.models.corrector import CorrectorDiagnostics
from made.physics import KinematicBicycleAsDynamicState, dynamic_bicycle_constraints
from made.upstream import apply_made_trajectory_with_controls
from made.utils import CorrectorConfig, ModelConfig

_DT = 0.1
_PARAMS = jnp.array([2.7])
_T = 4


def _db_cell(key: jax.Array, *, eval_max_steps: int = 3) -> MaDECell:
    """eval_tol=-1e9 forces the adaptive loop to always run every one of
    `eval_max_steps` iterations -- a deterministic, non-degenerate reference
    for `n_iterations`/`cap_hit` (same trick as
    `tests/test_corrector_box_projection.py`)."""
    physics = KinematicBicycleAsDynamicState()
    constraints = dynamic_bicycle_constraints()
    return MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(
            step_size=0.1,
            momentum=0.0,
            train_steps=2,
            eval_max_steps=eval_max_steps,
            eval_tol=-1e9,
            mode="enabled",
        ),
        key=key,
        dt=_DT,
    )


def _x_pred(state_dim: int) -> jnp.ndarray:
    base = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
    assert base.shape[0] == state_dim
    return jnp.stack([base * (1.0 + 0.01 * i) for i in range(_T)], axis=0)


def test_default_signature_unaffected_by_diagnostics_kwarg():
    """Every existing caller never passes `return_diagnostics`; the default
    return shape must remain the plain `(states, controls)` 2-tuple."""
    key = jax.random.key(0)
    cell = _db_cell(key)
    x_pred = _x_pred(cell.augmented_dynamics.physics.state_dim)
    x0 = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

    result = apply_made_trajectory_with_controls(
        cell, x_pred, _PARAMS, _DT, correction_mode="eval_adaptive", x0=x0
    )
    assert len(result) == 2
    states, controls = result
    assert states.shape == x_pred.shape
    assert controls.shape[0] == _T


def test_return_diagnostics_rejected_outside_eval_adaptive():
    key = jax.random.key(0)
    cell = _db_cell(key)
    x_pred = _x_pred(cell.augmented_dynamics.physics.state_dim)
    x0 = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="eval_adaptive"):
        apply_made_trajectory_with_controls(
            cell,
            x_pred,
            _PARAMS,
            _DT,
            correction_mode="full_fixed",
            x0=x0,
            return_diagnostics=True,
        )


def test_diagnostics_shape_and_matches_manual_per_timestep_calls_with_x0():
    """With an explicit `x0`, every row of `x_pred` is corrected. The stacked
    `[T]` diagnostics from the scan must exactly match calling
    `cell(..., return_diagnostics=True)` once per timestep in a Python loop."""
    key = jax.random.key(0)
    cell = _db_cell(key)
    x_pred = _x_pred(cell.augmented_dynamics.physics.state_dim)
    x0 = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

    states, controls, diag = apply_made_trajectory_with_controls(
        cell,
        x_pred,
        _PARAMS,
        _DT,
        correction_mode="eval_adaptive",
        x0=x0,
        return_diagnostics=True,
    )
    assert isinstance(diag, CorrectorDiagnostics)
    assert diag.n_iterations.shape == (_T,)
    assert diag.cap_hit.shape == (_T,)

    # Manual per-timestep reference: same autoregressive scan, done in Python.
    x_prev = x0
    expected_n_iter = []
    expected_cap_hit = []
    expected_states = []
    expected_controls = []
    for t in range(_T):
        x_next, u_next, diag_t = cell(
            x_prev, x_pred[t], _PARAMS, _DT, training=False, return_diagnostics=True
        )
        expected_states.append(x_next)
        expected_controls.append(u_next)
        expected_n_iter.append(int(diag_t.n_iterations))
        expected_cap_hit.append(bool(diag_t.cap_hit))
        x_prev = x_next

    np_states = jnp.stack(expected_states, axis=0)
    np_controls = jnp.stack(expected_controls, axis=0)
    assert jnp.allclose(states, np_states)
    assert jnp.allclose(controls, np_controls)
    assert list(map(int, diag.n_iterations)) == expected_n_iter
    assert list(map(bool, diag.cap_hit)) == expected_cap_hit
    # eval_tol=-1e9 forces the cap every step -- a non-degenerate reference.
    assert all(n == 3 for n in expected_n_iter)
    assert all(expected_cap_hit)


def test_diagnostics_legacy_first_row_is_zero_placeholder():
    """Without `x0`, the first row is passed through uncorrected (as with
    `controls`, which are zero-filled for row 0). Its diagnostics entry must
    be a zero-filled placeholder (`n_iterations=0`, `cap_hit=False`), and rows
    1..T-1 must match direct per-timestep corrector calls."""
    key = jax.random.key(0)
    cell = _db_cell(key)
    x_pred = _x_pred(cell.augmented_dynamics.physics.state_dim)

    states, controls, diag = apply_made_trajectory_with_controls(
        cell,
        x_pred,
        _PARAMS,
        _DT,
        correction_mode="eval_adaptive",
        return_diagnostics=True,
    )
    assert diag.n_iterations.shape == (_T,)
    assert diag.cap_hit.shape == (_T,)
    assert int(diag.n_iterations[0]) == 0
    assert bool(diag.cap_hit[0]) is False
    np_first_state = x_pred[0]
    assert jnp.allclose(states[0], np_first_state)
    assert jnp.allclose(controls[0], jnp.zeros_like(controls[0]))

    x_prev = x_pred[0]
    for t in range(1, _T):
        x_next, u_next, diag_t = cell(
            x_prev, x_pred[t], _PARAMS, _DT, training=False, return_diagnostics=True
        )
        assert jnp.allclose(states[t], x_next)
        assert jnp.allclose(controls[t], u_next)
        assert int(diag.n_iterations[t]) == int(diag_t.n_iterations)
        assert bool(diag.cap_hit[t]) == bool(diag_t.cap_hit)
        x_prev = x_next


def test_vmap_over_batch_yields_kt_shaped_diagnostics():
    """`jax.vmap` over a batch of K trajectories must yield `[K, T]`-shaped
    diagnostics, matching how states/controls become `[K, T, D]`/`[K, T, U]`."""
    key = jax.random.key(0)
    cell = _db_cell(key)
    k = 3
    x_pred_single = _x_pred(cell.augmented_dynamics.physics.state_dim)
    x_pred_batch = jnp.broadcast_to(x_pred_single, (k,) + x_pred_single.shape)
    x0_single = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x0_batch = jnp.broadcast_to(x0_single, (k,) + x0_single.shape)

    def _apply(x_pred, x0):
        return apply_made_trajectory_with_controls(
            cell,
            x_pred,
            _PARAMS,
            _DT,
            correction_mode="eval_adaptive",
            x0=x0,
            return_diagnostics=True,
        )

    states, controls, diag = jax.vmap(_apply)(x_pred_batch, x0_batch)
    assert states.shape == (k, _T, x_pred_single.shape[-1])
    assert controls.shape[0] == k
    assert controls.shape[1] == _T
    assert diag.n_iterations.shape == (k, _T)
    assert diag.cap_hit.shape == (k, _T)
    assert bool(jnp.all(diag.n_iterations == 3))
    assert bool(jnp.all(diag.cap_hit))
