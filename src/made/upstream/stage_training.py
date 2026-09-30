"""Stage 1 upstream training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from made.models import InverseDynamics, MaDECell, MaDEModel
from made.models.corrector import CorrectorDiagnostics
from made.physics import ConstraintSet, PhysicsModel
from made.upstream.base import UpstreamPredictor
from made.utils import UpstreamConfig


def apply_made_trajectory_with_controls(
    frozen_cell: MaDECell | MaDEModel,
    x_pred: jax.Array,
    params: jax.Array,
    dt: float,
    *,
    correction_mode: str = "full_fixed",
    x0: jax.Array | None = None,
    return_diagnostics: bool = False,
) -> (
    tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, CorrectorDiagnostics]
):
    """Apply a frozen MaDE cell autoregressively; return corrected states AND controls.

    Args:
        correction_mode: ``"full_fixed"`` / ``"it_only"`` / ``"detached_correction"``
            run the training-mode corrector paths; ``"eval_adaptive"`` runs the cell
            with ``training=False`` (adaptive while-loop corrector honouring
            ``eval_max_steps`` / ``eval_tol``) — use this for plug-and-play evaluation.
        x0: optional feasible seed state (e.g. the last *observed* context state).
            When given, the correction scan starts from ``x0`` and EVERY row of
            ``x_pred`` is corrected — output length equals ``len(x_pred)`` and the
            E01 "rollout seed must be feasible" convention holds. When ``None``,
            ``x_pred[0]`` seeds the scan uncorrected (legacy uncorrected-seed behaviour) and
            its control slot is zero-filled.
        return_diagnostics: opt-in per-timestep corrector diagnostics (iteration
            count + cap-hit flag), stacked into a ``CorrectorDiagnostics`` whose
            leaves have shape ``[T]`` — one entry per row of the returned
            ``states``/``controls``. Only supported when
            ``correction_mode="eval_adaptive"`` (the adaptive loop is the only one
            that computes these diagnostics); requesting it otherwise raises
            ``ValueError``. When ``x0`` is ``None``, the first row is passed
            through uncorrected (as with ``controls``), so its diagnostics entry
            is a zero-filled placeholder (``n_iterations=0``, ``cap_hit=False``)
            rather than a real corrector run.

    Returns:
        ``(states, controls)``: with ``x0`` — ``[T, D]`` and ``[T, U]``; without —
        ``[T, D]`` (first row passed through) and ``[T, U]`` (first row zeros).
        With ``return_diagnostics=True``, a third ``CorrectorDiagnostics`` element
        is appended whose leaves have shape ``[T]``.
    """
    mode_map = {
        "full_fixed": "train_fixed",
        "it_only": "it_only",
        "detached_correction": "detached_correction",
        "eval_adaptive": None,
    }
    if correction_mode not in mode_map:
        raise ValueError(
            "correction_mode must be 'full_fixed', 'it_only', 'detached_correction',"
            " or 'eval_adaptive'."
        )
    if return_diagnostics and correction_mode != "eval_adaptive":
        raise ValueError(
            "return_diagnostics=True is only supported for correction_mode='eval_adaptive' "
            "(iteration-count diagnostics are only computed by the adaptive loop); got "
            f"correction_mode='{correction_mode}'."
        )
    training = correction_mode != "eval_adaptive"

    def _step(
        x_prev: jax.Array, x_curr: jax.Array
    ) -> tuple[
        jax.Array,
        tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, CorrectorDiagnostics],
    ]:
        if return_diagnostics:
            x_corrected, u_corrected, diagnostics = frozen_cell(
                x_prev,
                x_curr,
                params,
                dt,
                training=training,
                correction_mode=mode_map[correction_mode],
                return_diagnostics=True,
            )
            return x_corrected, (x_corrected, u_corrected, diagnostics)
        x_corrected, u_corrected = frozen_cell(
            x_prev,
            x_curr,
            params,
            dt,
            training=training,
            correction_mode=mode_map[correction_mode],
        )
        return x_corrected, (x_corrected, u_corrected)

    if x0 is not None:
        if return_diagnostics:
            _, (states, controls, diagnostics) = jax.lax.scan(_step, x0, x_pred)
            return states, controls, diagnostics
        _, (states, controls) = jax.lax.scan(_step, x0, x_pred)
        return states, controls

    first_state = x_pred[0]
    if return_diagnostics:
        _, (tail_states, tail_controls, tail_diagnostics) = jax.lax.scan(
            _step, first_state, x_pred[1:]
        )
        states = jnp.concatenate([first_state[None, :], tail_states], axis=0)
        controls = jnp.concatenate([jnp.zeros_like(tail_controls[:1]), tail_controls], axis=0)
        diagnostics = jax.tree_util.tree_map(
            lambda leaf: jnp.concatenate([jnp.zeros_like(leaf[:1]), leaf], axis=0),
            tail_diagnostics,
        )
        return states, controls, diagnostics

    _, (tail_states, tail_controls) = jax.lax.scan(_step, first_state, x_pred[1:])
    states = jnp.concatenate([first_state[None, :], tail_states], axis=0)
    controls = jnp.concatenate([jnp.zeros_like(tail_controls[:1]), tail_controls], axis=0)
    return states, controls


def apply_made_trajectory(
    frozen_cell: MaDECell | MaDEModel,
    x_pred: jax.Array,
    params: jax.Array,
    dt: float,
    *,
    correction_mode: str = "full_fixed",
    x0: jax.Array | None = None,
) -> jax.Array:
    """Apply a frozen MaDE cell autoregressively to a trajectory (states only)."""
    states, _ = apply_made_trajectory_with_controls(
        frozen_cell,
        x_pred,
        params,
        dt,
        correction_mode=correction_mode,
        x0=x0,
    )
    return states


def stage1_loss(
    upstream: UpstreamPredictor,
    frozen_I: InverseDynamics,
    frozen_enc: Any,
    physics: PhysicsModel,
    constraints: ConstraintSet,
    context: jax.Array,
    x_gt: jax.Array,
    params: jax.Array,
    dt: float,
    config: UpstreamConfig,
    metadata: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Stage 1 upstream loss."""
    if metadata is not None and frozen_enc is not None:
        params = frozen_enc(metadata)
    x_pred = upstream(context)
    l_mse = jnp.mean((x_pred - x_gt) ** 2)
    repeated_params = jnp.broadcast_to(params, (x_pred.shape[0] - 1, params.shape[0]))
    u_hat = jax.vmap(frozen_I)(x_pred[:-1], x_pred[1:], repeated_params)
    ineq = jax.vmap(constraints.violation)(x_pred[1:], u_hat)
    l_ineq = jnp.mean(jnp.sum(ineq ** 2, axis=-1))
    x_dyn = x_pred[:-1] + dt * jax.vmap(physics.vector_field)(
        x_pred[:-1],
        u_hat,
        repeated_params,
        jnp.zeros((x_pred.shape[0] - 1,)),
    )
    l_dyn = jnp.mean((x_pred[1:] - x_dyn) ** 2)
    if config.stage1_loss == "mse":
        total = l_mse
    elif config.stage1_loss == "mse_ineq":
        total = l_mse + config.stage1_mu_ineq * l_ineq
    elif config.stage1_loss == "full":
        total = l_mse + config.stage1_mu_ineq * l_ineq + config.stage1_mu_dyn * l_dyn
    else:
        raise ValueError("stage1_loss must be 'mse', 'mse_ineq', or 'full'.")
    return total, {
        "mse": l_mse,
        "ineq_penalty": l_ineq,
        "dynamics_penalty": l_dyn,
    }


