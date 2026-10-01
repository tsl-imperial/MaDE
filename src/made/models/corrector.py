# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Inequality-correcting control update loop."""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from made.models.augmented_dynamics import AugmentedDynamics
from made.physics import ConstraintSet, _maybe_assert_xy_zero
from made.utils import CorrectorConfig


class CorrectorDiagnostics(NamedTuple):
    """Eval-time corrector loop diagnostics (opt-in, `mode="eval_adaptive"` only).

    A pytree of arrays so it stays vmap/jit-safe and stacks cleanly across a batch.
    """

    n_iterations: jax.Array
    """Number of `_body` applications executed before `_cond` became false."""
    cap_hit: jax.Array
    """True iff the loop exited at `eval_max_steps` with residual violation still above
    `eval_tol` -- cut off by the step cap rather than by convergence."""


class Corrector(eqx.Module):
    """Gradient-based corrector that stays on the dynamics manifold."""

    step_size: float
    momentum: float
    train_steps: int
    eval_max_steps: int
    eval_tol: float
    proximity_gamma: float
    tracking_gamma: float

    def __init__(self, config: CorrectorConfig) -> None:
        """Copy the corrector hyperparameters from ``config``.

        Args:
            config: Corrector configuration.
        """
        self.step_size = config.step_size
        self.momentum = config.momentum
        self.train_steps = config.train_steps
        self.eval_max_steps = config.eval_max_steps
        self.eval_tol = config.eval_tol
        self.proximity_gamma = getattr(config, "proximity_gamma", 0.0)
        self.tracking_gamma = getattr(config, "tracking_gamma", 0.0)

    def _violation_loss(
        self,
        u: jax.Array,
        constraints: ConstraintSet,
        dynamics: AugmentedDynamics,
        x_prev: jax.Array,
        params: jax.Array,
        dt: float,
        solver: diffrax.AbstractSolver | None,
        adjoint: diffrax.AbstractAdjoint | None,
        fast: bool | None = None,
        u_ref: jax.Array | None = None,
        gamma: float | None = None,
    ) -> jax.Array:
        """Corrector objective. With `u_ref` given, adds the quadratic term on `u - u_hat`.

        `J_gamma(u) = ||ReLU(g(T(x,u),u))||^2 + gamma||u - u_hat||^2`, where `u_hat` is the
        inverse-dynamics proposal the correction starts from.

        Two different callers use this with separate gamma fields, so one cannot move the
        other by accident:

        | caller | `gamma` | field |
        |---|---|---|
        | `_correct_eval` | omitted, so `proximity_gamma` | `proximity_gamma` (inference-only, frozen models) |
        | `_correct_train` | passed explicitly | `tracking_gamma` (differentiated through in phase 2) |

        Both fields default to 0.0; when the applicable one is zero the term is not built.

        Args:
            u: Candidate control.
            constraints: Constraint set.
            dynamics: Augmented dynamics used to propagate ``x_prev``.
            x_prev: Previous state.
            params: Physical parameters.
            dt: Step length.
            solver: Optional diffrax solver.
            adjoint: Optional diffrax adjoint.
            fast: Optional override for the single-step Heun fast path.
            u_ref: Inverse-dynamics proposal for the quadratic term, or None.
            gamma: Weight of the quadratic term; None uses ``proximity_gamma``.

        Returns:
            Scalar objective.
        """
        x = dynamics.integrate(
            x_prev, u, params, dt, solver=solver, adjoint=adjoint, fast=fast
        )
        violation = constraints.violation(x, u)
        violation = _maybe_assert_xy_zero(violation)
        loss = jnp.sum(violation ** 2)
        # getattr, not self.proximity_gamma: checkpoints saved before this field existed
        # deserialise into a Corrector with no such attribute, and the eval path reads it on
        # every restored checkpoint. A pre-field checkpoint reads as gamma = 0.
        gamma = getattr(self, "proximity_gamma", 0.0) if gamma is None else gamma
        if u_ref is not None and gamma != 0.0:
            # Guarded on both u_ref supplied and gamma non-zero, so the gamma = 0 arm takes
            # the same code path as before the term existed, not one that merely evaluates
            # to the same number.
            loss = loss + gamma * jnp.sum((u - u_ref) ** 2)
        return loss

    def _correct_train(
        self,
        x_pred: jax.Array,
        u: jax.Array,
        constraints: ConstraintSet,
        dynamics: AugmentedDynamics,
        x_prev: jax.Array,
        params: jax.Array,
        dt: float,
        solver: diffrax.AbstractSolver | None,
        adjoint: diffrax.AbstractAdjoint | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Fixed-step momentum descent on the violation loss (differentiable).

        Args:
            x_pred: Predicted next state.
            u: Initial control proposal.
            constraints: Constraint set.
            dynamics: Augmented dynamics.
            x_prev: Previous state.
            params: Physical parameters.
            dt: Step length.
            solver: Optional diffrax solver.
            adjoint: Optional diffrax adjoint for the outer integration.

        Returns:
            Tuple ``(x_final, u_final)``.
        """
        grad_adjoint = diffrax.DirectAdjoint() if adjoint is None else adjoint

        # `tracking_gamma` is a static float field, so this is a Python-level branch taken
        # once at trace time: when it is 0.0 the loop closes over the same arguments as
        # before the term existed.
        tracking_gamma = getattr(self, "tracking_gamma", 0.0)
        u_hat = u if tracking_gamma != 0.0 else None

        def _body(
            _: int,
            carry: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """One momentum step.

            Args:
                _: Loop index (unused).
                carry: ``(u, velocity, x)``.

            Returns:
                Updated ``(u, velocity, x)``.
            """
            u_curr, velocity, _ = carry
            grad_u = jax.grad(self._violation_loss)(
                u_curr,
                constraints,
                dynamics,
                x_prev,
                params,
                dt,
                solver,
                grad_adjoint,
                None,
                u_hat,
                tracking_gamma,
            )
            new_velocity = self.momentum * velocity - self.step_size * grad_u
            new_u = u_curr + new_velocity
            # Defense-in-depth: keep u inside the constraint box so the next integrate (and
            # the gradient through KinematicBicycle.vector_field's tan(δ)) never sees |δ|
            # close to π/2. Each GD step on u could otherwise push δ past the box.
            new_u = jnp.clip(new_u, constraints.control_min, constraints.control_max)
            new_x = dynamics.integrate(
                x_prev,
                new_u,
                params,
                dt,
                solver=solver,
                adjoint=adjoint,
            )
            return new_u, new_velocity, new_x

        u_final, _, x_final = jax.lax.fori_loop(
            0,
            self.train_steps,
            _body,
            (u, jnp.zeros_like(u), x_pred),
        )
        return x_final, u_final

    def _correct_eval(
        self,
        x_pred: jax.Array,
        u: jax.Array,
        constraints: ConstraintSet,
        dynamics: AugmentedDynamics,
        x_prev: jax.Array,
        params: jax.Array,
        dt: float,
        solver: diffrax.AbstractSolver | None,
        adjoint: diffrax.AbstractAdjoint | None,
        *,
        return_diagnostics: bool = False,
    ) -> (
        tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, CorrectorDiagnostics]
    ):
        """Adaptive momentum descent that stops at tolerance or the step cap.

        Args:
            x_pred: Predicted next state.
            u: Initial control proposal.
            constraints: Constraint set.
            dynamics: Augmented dynamics.
            x_prev: Previous state.
            params: Physical parameters.
            dt: Step length.
            solver: Optional diffrax solver.
            adjoint: Optional diffrax adjoint.
            return_diagnostics: Also return a ``CorrectorDiagnostics``.

        Returns:
            ``(x_final, u_final)``, plus diagnostics when requested.
        """
        adjoint = diffrax.DirectAdjoint() if adjoint is None else adjoint

        def _cond(carry: tuple[jax.Array, jax.Array, jax.Array, int]) -> jax.Array:
            """Continue while violated and under the step cap.

            Args:
                carry: ``(u, velocity, x, step)``.

            Returns:
                Boolean scalar.
            """
            u_curr, velocity, x_curr, step = carry
            del velocity
            return (jnp.max(constraints(x_curr, u_curr)) > self.eval_tol) & (
                step < self.eval_max_steps
            )

        # The proximity reference is the inverse-dynamics proposal the correction starts
        # from, captured before the loop so every iterate is measured against the original
        # u_hat rather than the previous iterate.
        u_hat = u

        def _body(
            carry: tuple[jax.Array, jax.Array, jax.Array, int],
        ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
            """One momentum step.

            Args:
                carry: ``(u, velocity, x, step)``.

            Returns:
                Updated ``(u, velocity, x, step)``.
            """
            u_curr, velocity, _, step = carry
            grad_u = jax.grad(self._violation_loss)(
                u_curr,
                constraints,
                dynamics,
                x_prev,
                params,
                dt,
                solver,
                adjoint,
                None,
                u_hat,
            )
            new_velocity = self.momentum * velocity - self.step_size * grad_u
            new_u = u_curr + new_velocity
            new_u = jnp.clip(new_u, constraints.control_min, constraints.control_max)
            new_x = dynamics.integrate(x_prev, new_u, params, dt, solver=solver, adjoint=adjoint)
            return new_u, new_velocity, new_x, step + 1

        u_final, _, x_final, step_final = jax.lax.while_loop(
            _cond,
            _body,
            (u, jnp.zeros_like(u), x_pred, 0),
        )
        if not return_diagnostics:
            return x_final, u_final
        residual = jnp.max(constraints(x_final, u_final))
        cap_hit = (step_final == self.eval_max_steps) & (residual > self.eval_tol)
        diagnostics = CorrectorDiagnostics(
            n_iterations=jnp.asarray(step_final),
            cap_hit=cap_hit,
        )
        return x_final, u_final, diagnostics

    def __call__(
        self,
        x_pred: jax.Array,
        u: jax.Array,
        constraints: ConstraintSet,
        dynamics: AugmentedDynamics,
        x_prev: jax.Array,
        params: jax.Array,
        dt: float,
        *,
        training: bool = True,
        mode: str | None = None,
        solver: diffrax.AbstractSolver | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, CorrectorDiagnostics]
    ):
        """Correct the control proposal according to ``mode``.

        Args:
            x_pred: Predicted next state.
            u: Initial control proposal.
            constraints: Constraint set.
            dynamics: Augmented dynamics.
            x_prev: Previous state.
            params: Physical parameters.
            dt: Step length.
            training: Selects ``train_fixed`` when True and ``eval_adaptive`` otherwise, if
                ``mode`` is None.
            mode: One of ``train_fixed``, ``eval_adaptive``, ``it_only``,
                ``detached_correction``.
            solver: Optional diffrax solver.
            adjoint: Optional diffrax adjoint.
            return_diagnostics: Also return diagnostics (``eval_adaptive`` only).

        Returns:
            ``(x, u)``, plus ``CorrectorDiagnostics`` when requested.

        Raises:
            ValueError: If diagnostics are requested outside ``eval_adaptive`` or ``mode`` is
                unsupported.
        """
        mode = ("train_fixed" if training else "eval_adaptive") if mode is None else mode
        if return_diagnostics and mode != "eval_adaptive":
            raise ValueError(
                "return_diagnostics=True is only supported for mode='eval_adaptive' "
                f"(iteration-count diagnostics are only computed by the adaptive loop); got "
                f"mode='{mode}'."
            )
        if mode == "it_only":
            return x_pred, u
        if mode == "train_fixed":
            return self._correct_train(
                x_pred,
                u,
                constraints,
                dynamics,
                x_prev,
                params,
                dt,
                solver,
                adjoint,
            )
        if mode == "eval_adaptive":
            return self._correct_eval(
                x_pred,
                u,
                constraints,
                dynamics,
                x_prev,
                params,
                dt,
                solver,
                adjoint,
                return_diagnostics=return_diagnostics,
            )
        if mode == "detached_correction":
            x_corr, u_corr = self._correct_train(
                x_pred,
                u,
                constraints,
                dynamics,
                x_prev,
                params,
                dt,
                solver,
                adjoint,
            )
            return (
                x_pred + jax.lax.stop_gradient(x_corr - x_pred),
                u + jax.lax.stop_gradient(u_corr - u),
            )
        raise ValueError(
            "Unsupported correction mode "
            f"'{mode}'. Expected train_fixed, eval_adaptive, it_only, or detached_correction."
        )
