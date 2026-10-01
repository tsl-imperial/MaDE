# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for eval-time corrector iteration-count diagnostics.

`Corrector._correct_eval` / `Corrector.__call__` can optionally report a
`CorrectorDiagnostics` tuple (iteration count + cap-hit flag) via the
keyword-only `return_diagnostics=True` flag. This is a strictly opt-in
extension: the default (``return_diagnostics=False``) return signature is
unchanged, so every existing caller -- `MaDECell.__call__`, the evaluation
scripts, `made/upstream/*` -- keeps working unmodified.
The training path (`mode="train_fixed"`, `_correct_train`'s `fori_loop`) is
untouched entirely: it gained no new parameter and no new behaviour.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from made.models.augmented_dynamics import AugmentedDynamics, ZeroResidual
from made.models.corrector import Corrector, CorrectorDiagnostics
from made.physics import KinematicBicycleAsDynamicState, dynamic_bicycle_constraints
from made.utils import CorrectorConfig
from made.models import MaDECell
from made.physics import ConstraintSet
from made.physics import DoubleIntegrator

# Scenario reused from tests/test_corrector_box_projection.py: a control that
# starts outside the box (delta=0.6 vs delta_max=0.5), so the corrector loop
# has real gradient-descent work to do and a nonzero, non-degenerate violation
# trajectory to report on.
_X_PREV = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
_X_PRED = jnp.array([0.05, 0.0, 0.05, 5.0, 0.0, 0.0])
_U0 = jnp.array([0.6, 0.0])
_PARAMS = jnp.array([2.7])
_DT = 0.1


def _db_dynamics_and_constraints() -> tuple[AugmentedDynamics, object]:
    """Build zero-residual dynamics and constraints for the dynamic-state bicycle.

    Returns:
        Tuple of (dynamics, constraints).
    """
    physics = KinematicBicycleAsDynamicState()
    constraints = dynamic_bicycle_constraints()
    dynamics = AugmentedDynamics(physics=physics, residual=ZeroResidual(physics.state_dim))
    return dynamics, constraints


def test_correct_eval_diagnostics_reports_exact_iteration_count() -> None:
    """diag.n_iterations exactly matches the number of `_body` applications.

    The reference count is derived, not hard-coded: run `_correct_train` for
    exactly 1 step (a deterministic `fori_loop`) to get the post-1-step state,
    then pick `eval_tol` strictly between the initial and post-1-step max
    violation. The `_correct_eval` `while_loop`'s own convergence check must
    then stop it after exactly 1 iteration.
    """
    dynamics, constraints = _db_dynamics_and_constraints()
    common = {"step_size": 0.05, "momentum": 0.0}

    ref = Corrector(CorrectorConfig(train_steps=1, **common))
    violation0 = float(jnp.max(constraints(_X_PRED, _U0)))
    x1, u1 = ref._correct_train(
        _X_PRED, _U0, constraints, dynamics, _X_PREV, _PARAMS, _DT, solver=None, adjoint=None
    )
    violation1 = float(jnp.max(constraints(x1, u1)))
    assert violation1 < violation0, "a gradient step should reduce the max constraint violation"

    eval_tol = (violation0 + violation1) / 2.0
    probe = Corrector(CorrectorConfig(eval_max_steps=10, eval_tol=eval_tol, **common))
    x_final, u_final, diag = probe._correct_eval(
        _X_PRED,
        _U0,
        constraints,
        dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        solver=None,
        adjoint=None,
        return_diagnostics=True,
    )

    assert isinstance(diag, CorrectorDiagnostics)
    assert int(diag.n_iterations) == 1
    assert bool(diag.cap_hit) is False
    assert jnp.allclose(x_final, x1, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(u_final, u1, rtol=1e-12, atol=1e-12)


def test_correct_eval_diagnostics_flags_cap_hit() -> None:
    """eval_tol=-1e9 keeps `max(constraints(x, u)) > eval_tol` permanently true,
    so the loop must run every one of `eval_max_steps` iterations. Per the
    spec, a cap hit is `step == eval_max_steps` AND the residual violation is
    still above tolerance -- exactly this case.
    """
    dynamics, constraints = _db_dynamics_and_constraints()
    max_steps = 3
    corrector = Corrector(
        CorrectorConfig(step_size=0.05, momentum=0.0, eval_max_steps=max_steps, eval_tol=-1e9)
    )
    x_final, u_final, diag = corrector._correct_eval(
        _X_PRED,
        _U0,
        constraints,
        dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        solver=None,
        adjoint=None,
        return_diagnostics=True,
    )
    assert bool(jnp.all(jnp.isfinite(x_final)))
    assert bool(jnp.all(jnp.isfinite(u_final)))
    assert int(diag.n_iterations) == max_steps
    assert bool(diag.cap_hit) is True


def test_correct_eval_default_signature_unchanged() -> None:
    """Callers that never pass `return_diagnostics` (every current call site)
    keep the plain `(x, u)` 2-tuple return -- the opt-in contract."""
    dynamics, constraints = _db_dynamics_and_constraints()
    corrector = Corrector(CorrectorConfig(eval_max_steps=4, eval_tol=1e-6))
    result = corrector._correct_eval(
        _X_PRED, _U0, constraints, dynamics, _X_PREV, _PARAMS, _DT, solver=None, adjoint=None
    )
    assert len(result) == 2
    x_final, u_final = result
    assert x_final.shape == _X_PRED.shape
    assert u_final.shape == _U0.shape


def test_call_return_diagnostics_matches_correct_eval_direct() -> None:
    """`Corrector.__call__(mode="eval_adaptive", return_diagnostics=True)` is a
    thin forward to `_correct_eval` and must return an identical 3-tuple."""
    dynamics, constraints = _db_dynamics_and_constraints()
    corrector = Corrector(CorrectorConfig(eval_max_steps=4, eval_tol=1e-6))
    x_direct, u_direct, diag_direct = corrector._correct_eval(
        _X_PRED,
        _U0,
        constraints,
        dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        solver=None,
        adjoint=None,
        return_diagnostics=True,
    )
    result = corrector(
        _X_PRED,
        _U0,
        constraints,
        dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        mode="eval_adaptive",
        return_diagnostics=True,
    )
    assert len(result) == 3
    x_call, u_call, diag_call = result
    assert jnp.array_equal(x_call, x_direct)
    assert jnp.array_equal(u_call, u_direct)
    assert int(diag_call.n_iterations) == int(diag_direct.n_iterations)
    assert bool(diag_call.cap_hit) == bool(diag_direct.cap_hit)


def test_return_diagnostics_rejected_outside_eval_adaptive() -> None:
    """`return_diagnostics=True` is only meaningful for the adaptive eval loop;
    requesting it under `train_fixed` (or any other mode) must fail loudly
    instead of silently returning a 2-tuple or bogus diagnostics."""
    dynamics, constraints = _db_dynamics_and_constraints()
    corrector = Corrector(CorrectorConfig(train_steps=1, step_size=0.05, momentum=0.0))
    with pytest.raises(ValueError, match="eval_adaptive"):
        corrector(
            _X_PRED,
            _U0,
            constraints,
            dynamics,
            _X_PREV,
            _PARAMS,
            _DT,
            mode="train_fixed",
            return_diagnostics=True,
        )


def test_correct_train_untouched_by_diagnostics_plumbing() -> None:
    """The training path (`_correct_train`, `mode="train_fixed"`) gained no new
    parameter and no new behaviour: it still returns a plain 2-tuple and does
    not accept `return_diagnostics` at all."""
    dynamics, constraints = _db_dynamics_and_constraints()
    corrector = Corrector(CorrectorConfig(train_steps=2, step_size=0.05, momentum=0.0))

    with pytest.raises(TypeError):
        corrector._correct_train(
            _X_PRED,
            _U0,
            constraints,
            dynamics,
            _X_PREV,
            _PARAMS,
            _DT,
            solver=None,
            adjoint=None,
            return_diagnostics=True,
        )

    result = corrector._correct_train(
        _X_PRED, _U0, constraints, dynamics, _X_PREV, _PARAMS, _DT, solver=None, adjoint=None
    )
    assert len(result) == 2

    via_call = corrector(
        _X_PRED,
        _U0,
        constraints,
        dynamics,
        _X_PREV,
        _PARAMS,
        _DT,
        mode="train_fixed",
    )
    assert len(via_call) == 2
    x_direct, u_direct = result
    x_call, u_call = via_call
    assert jnp.array_equal(x_direct, x_call)
    assert jnp.array_equal(u_direct, u_call)


def test_made_cell_default_calls_unaffected(
    small_cell: MaDECell,
    double_integrator: tuple[DoubleIntegrator, ConstraintSet],
) -> None:
    """End-to-end regression: the full `MaDECell.__call__` path (as used by
    the trainer and every evaluation script) never passes
    `return_diagnostics` and must keep returning a plain `(x, u)` 2-tuple in
    both training and eval mode."""
    physics, _ = double_integrator
    x_prev = jnp.zeros(physics.state_dim)
    x_curr = jnp.ones(physics.state_dim) * 0.1
    params = jnp.zeros(physics.param_dim)

    x_train, u_train = small_cell(x_prev, x_curr, params, 0.1, training=True)
    assert x_train.shape == x_prev.shape
    assert u_train.shape[0] == physics.control_dim

    x_eval, u_eval = small_cell(x_prev, x_curr, params, 0.1, training=False)
    assert x_eval.shape == x_prev.shape
    assert u_eval.shape[0] == physics.control_dim
