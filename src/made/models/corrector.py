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
    """True iff the loop exited at `eval_max_steps` with residual violation still
    above `eval_tol` (i.e. the adaptive loop was cut off by the step cap rather
    than by convergence)."""


class Corrector(eqx.Module):
    """Gradient-based corrector that stays on the dynamics manifold."""

    step_size: float
    momentum: float
    train_steps: int
    eval_max_steps: int
    eval_tol: float
    proximity_gamma: float
    tracking_gamma: float

    def __init__(self, config: CorrectorConfig):
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

        **The same expression serves two different experiments and they must not be confused.**

        | caller | `gamma` | field | experiment |
        |---|---|---|---|
        | `_correct_eval` | omitted, so `proximity_gamma` | `proximity_gamma` | X2, inference-only on frozen models, spec D-4 and D-5 |
        | `_correct_train` | passed explicitly | `tracking_gamma` | Y2, the tracking arm, so phase 2 differentiates through it |

        Implementing X2 does not necessarily change `_correct_train`; gamma stays
        evaluation-time there. Y2 changes it DELIBERATELY and under a separate
        field, so that X2's sweep cannot move the training objective by accident.

        **Both fields default to 0.0**, and when the applicable one is zero the term is not
        built and multiplied by zero -- it is absent, and the code path is the published one.
        """
        x = dynamics.integrate(
            x_prev, u, params, dt, solver=solver, adjoint=adjoint, fast=fast
        )
        violation = constraints.violation(x, u)
        violation = _maybe_assert_xy_zero(violation)
        loss = jnp.sum(violation ** 2)
        # getattr, NOT self.proximity_gamma. Checkpoints saved before this field existed
        # deserialise into a Corrector that HAS NO SUCH ATTRIBUTE -- confirmed against
        # outputs/e02-refresh/made_seed0, which raises AttributeError on a direct read. The
        # training path short-circuits on `u_ref is None` and never reaches it, but the
        # EVALUATION path would have raised on every restored checkpoint, i.e. on every
        # existing real-data and simulated evaluation. A pre-field checkpoint reads as
        # gamma = 0, which is the published behaviour.
        gamma = getattr(self, "proximity_gamma", 0.0) if gamma is None else gamma
        if u_ref is not None and gamma != 0.0:
            # Guarded on BOTH u_ref being supplied and gamma being non-zero, so the gamma = 0
            # arm of either experiment takes a code path identical to the published one rather
            # than one that merely evaluates to the same number.
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
        grad_adjoint = diffrax.DirectAdjoint() if adjoint is None else adjoint

        # Y2's tracking term. `tracking_gamma` is a STATIC float field, so this is a
        # Python-level branch taken once at trace time: when it is 0.0 the loop closes over
        # exactly the arguments it closed over before the term existed, and the published
        # training path is unchanged as a code path rather than merely as a number.
        tracking_gamma = getattr(self, "tracking_gamma", 0.0)
        u_hat = u if tracking_gamma != 0.0 else None

        def _body(
            _: int,
            carry: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
            # Layer 2 (defense-in-depth): keep u inside the constraint box so the
            # next integrate (and the gradient through KinematicBicycle.vector_field's
            # tan(δ)) never sees |δ| close to π/2. Load-bearing during corrector
            # iterations (training and eval) where each GD step on u could otherwise
            # push δ past the box.
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
        adjoint = diffrax.DirectAdjoint() if adjoint is None else adjoint

        def _cond(carry: tuple[jax.Array, jax.Array, jax.Array, int]) -> jax.Array:
            u_curr, velocity, x_curr, step = carry
            del velocity
            return (jnp.max(constraints(x_curr, u_curr)) > self.eval_tol) & (
                step < self.eval_max_steps
            )

        # X2 (spec D-5): the proximity reference is the inverse-dynamics proposal the
        # correction starts from, captured BEFORE the loop so every iterate is measured
        # against the original u_hat rather than against the previous iterate.
        u_hat = u

        def _body(
            carry: tuple[jax.Array, jax.Array, jax.Array, int],
        ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
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
