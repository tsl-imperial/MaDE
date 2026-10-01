# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for `return_diagnostics` threaded through `MaDECell.__call__`.

`Corrector.__call__`/`Corrector._correct_eval` already support an opt-in
`return_diagnostics=True` keyword that returns a `(x, u, CorrectorDiagnostics)`
3-tuple instead of the default `(x, u)` 2-tuple (see
`tests/test_corrector_diagnostics.py`). This file pins that `MaDECell.__call__`
is a thin, strictly opt-in forward of that same flag: default behaviour for
every existing caller (trainer, evaluation scripts, `apply_made_trajectory*`)
is unchanged, and `return_diagnostics=True` is only accepted with
`correction_mode="eval_adaptive"` (mirroring the corrector's own contract).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from made.models import MaDECell
from made.models.corrector import CorrectorDiagnostics
from made.physics import KinematicBicycleAsDynamicState, dynamic_bicycle_constraints
from made.utils import CorrectorConfig, ModelConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from made.physics import ConstraintSet
    from made.physics import DoubleIntegrator


def _build_db_cell(key: jax.Array, *, eval_max_steps: int = 3) -> MaDECell:
    """Small DB MaDECell with eval_tol=-1e9 so the adaptive loop always runs
    every one of `eval_max_steps` iterations -- a deterministic, non-degenerate
    `n_iterations`/`cap_hit` reference (same trick as
    `tests/test_corrector_box_projection.py`).

    Args:
        key: PRNG key for the cell initialisation.
        eval_max_steps: Cap on adaptive corrector iterations.

    Returns:
        The constructed cell.
    """
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
        dt=0.1,
    )


_X_PREV = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
_X_CURR = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
_PARAMS = jnp.array([2.7])
_DT = 0.1


def test_default_call_returns_2_tuple(
    small_cell: MaDECell,
    double_integrator: "tuple[DoubleIntegrator, ConstraintSet]",
) -> None:
    """No `return_diagnostics` kwarg at all -- the pre-existing call shape."""
    physics, _ = double_integrator
    x_prev = jnp.zeros(physics.state_dim)
    x_curr = jnp.ones(physics.state_dim) * 0.1
    params = jnp.zeros(physics.param_dim)
    result = small_cell(x_prev, x_curr, params, 0.1, training=False)
    assert len(result) == 2


def test_explicit_false_returns_2_tuple(
    small_cell: MaDECell,
    double_integrator: "tuple[DoubleIntegrator, ConstraintSet]",
) -> None:
    """`return_diagnostics=False` passed explicitly is equivalent to omitting it."""
    physics, _ = double_integrator
    x_prev = jnp.zeros(physics.state_dim)
    x_curr = jnp.ones(physics.state_dim) * 0.1
    params = jnp.zeros(physics.param_dim)
    result = small_cell(x_prev, x_curr, params, 0.1, training=False, return_diagnostics=False)
    assert len(result) == 2


def test_return_diagnostics_true_yields_3_tuple_matching_corrector() -> None:
    """`MaDECell.__call__(return_diagnostics=True)` under `eval_adaptive` is a
    thin forward: it must equal what you get by manually computing (I -> T)
    and then calling `cell.corrector` directly with the same flag."""
    key = jax.random.key(0)
    cell = _build_db_cell(key)

    x_cell, u_cell, diag_cell = cell(
        _X_PREV,
        _X_CURR,
        _PARAMS,
        _DT,
        training=False,
        return_diagnostics=True,
    )
    assert isinstance(diag_cell, CorrectorDiagnostics)

    u_manual = cell.inverse_dynamics(_X_PREV, _X_CURR, _PARAMS)
    x_pred_manual = cell.augmented_dynamics.integrate(_X_PREV, u_manual, _PARAMS, _DT)
    x_direct, u_direct, diag_direct = cell.corrector(
        x_pred_manual,
        u_manual,
        cell.constraints,
        cell.augmented_dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        mode="eval_adaptive",
        return_diagnostics=True,
    )

    assert jnp.array_equal(x_cell, x_direct)
    assert jnp.array_equal(u_cell, u_direct)
    assert int(diag_cell.n_iterations) == int(diag_direct.n_iterations)
    assert bool(diag_cell.cap_hit) == bool(diag_direct.cap_hit)
    # eval_tol=-1e9 forces the cap every time -- a non-degenerate reference.
    assert int(diag_cell.n_iterations) == 3
    assert bool(diag_cell.cap_hit) is True


def test_return_diagnostics_rejected_outside_eval_adaptive() -> None:
    """`training=True` resolves to `mode="train_fixed"` inside the corrector,
    which must reject `return_diagnostics=True` with the same ValueError the
    corrector itself raises (MaDECell adds no separate check -- it forwards)."""
    key = jax.random.key(0)
    cell = _build_db_cell(key)
    with pytest.raises(ValueError, match="eval_adaptive"):
        cell(_X_PREV, _X_CURR, _PARAMS, _DT, training=True, return_diagnostics=True)


def test_return_diagnostics_rejected_when_corrector_disabled() -> None:
    """corrector_mode='disabled' forces mode='it_only' regardless of `training`;
    `return_diagnostics=True` must still be rejected consistently."""
    physics = KinematicBicycleAsDynamicState()
    constraints = dynamic_bicycle_constraints()
    key = jax.random.key(1)
    cell = MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(mode="disabled"),
        key=key,
        dt=0.1,
    )
    with pytest.raises(ValueError, match="eval_adaptive"):
        cell(_X_PREV, _X_CURR, _PARAMS, _DT, training=False, return_diagnostics=True)


def test_vmap_over_batch_yields_stacked_diagnostics() -> None:
    """`jax.vmap` over a batch of (x_prev, x_curr) pairs must yield a
    `CorrectorDiagnostics` whose leaves have a leading batch axis, matching how
    x/u themselves are vmapped."""
    key = jax.random.key(0)
    cell = _build_db_cell(key)
    batch = 5
    x_prev_batch = jnp.broadcast_to(_X_PREV, (batch,) + _X_PREV.shape)
    x_curr_batch = jnp.broadcast_to(_X_CURR, (batch,) + _X_CURR.shape)
    params_batch = jnp.broadcast_to(_PARAMS, (batch,) + _PARAMS.shape)

    def _call(
        x_prev: jax.Array, x_curr: jax.Array, params: jax.Array
    ) -> tuple[jax.Array, jax.Array, CorrectorDiagnostics]:
        """Call the cell with diagnostics on, for `jax.vmap`.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.

        Returns:
            The `(x, u, diagnostics)` triple from the cell.
        """
        return cell(
            x_prev, x_curr, params, _DT, training=False, return_diagnostics=True
        )

    x_out, u_out, diag_out = jax.vmap(_call)(x_prev_batch, x_curr_batch, params_batch)
    assert x_out.shape == (batch,) + _X_PREV.shape
    assert u_out.shape[0] == batch
    assert diag_out.n_iterations.shape == (batch,)
    assert diag_out.cap_hit.shape == (batch,)
    assert bool(jnp.all(diag_out.n_iterations == 3))
    assert bool(jnp.all(diag_out.cap_hit))
