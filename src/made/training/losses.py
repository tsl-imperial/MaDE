# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Loss functions for MaDE training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx

from made.models import MaDECell, MaDEModel
from made.physics import _maybe_assert_xy_zero
from made.utils import TrainingConfig

MaDELike = MaDECell | MaDEModel


def _stop_arrays(tree: Any) -> Any:
    """Apply ``stop_gradient`` to every array leaf of a pytree.

    Args:
        tree: Pytree to stop gradients through.

    Returns:
        Pytree with the same structure.
    """
    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        tree,
    )


def _cell(model: MaDELike) -> MaDECell:
    """Return the underlying ``MaDECell``.

    Args:
        model: A cell or a wrapper model.

    Returns:
        The cell.
    """
    return model.cell if isinstance(model, MaDEModel) else model


def _replace_cell(model: MaDELike, cell: MaDECell) -> MaDELike:
    """Replace the cell of a wrapper model, or return the new cell for a bare cell.

    Args:
        model: A cell or a wrapper model.
        cell: Replacement cell.

    Returns:
        Model of the same kind as ``model``.
    """
    if isinstance(model, MaDEModel):
        return eqx.tree_at(lambda made_model: made_model.cell, model, cell)
    return cell


def stop_i_side(model: MaDELike) -> MaDELike:
    """Stop inverse-dynamics leaves while preserving forward values.

    Args:
        model: Model to freeze the inverse-dynamics side of.

    Returns:
        Model with inverse-dynamics gradients stopped.
    """
    cell = _cell(model)
    stopped_cell = eqx.tree_at(
        lambda made_cell: made_cell.inverse_dynamics,
        cell,
        _stop_arrays(cell.inverse_dynamics),
    )
    return _replace_cell(model, stopped_cell)


def stop_t_side(model: MaDELike) -> MaDELike:
    """Stop residual/encoder leaves while preserving forward values.

    Args:
        model: Model to freeze the forward-dynamics side of.

    Returns:
        Model with residual and encoder gradients stopped.
    """
    cell = _cell(model)
    stopped_cell = eqx.tree_at(
        lambda made_cell: made_cell.augmented_dynamics,
        cell,
        _stop_arrays(cell.augmented_dynamics),
    )
    if isinstance(model, MaDEModel) and model.encoder is not None:
        stopped_model = eqx.tree_at(lambda made_model: made_model.cell, model, stopped_cell)
        return eqx.tree_at(
            lambda made_model: made_model.encoder,
            stopped_model,
            _stop_arrays(model.encoder),
        )
    return _replace_cell(model, stopped_cell)


def forward_consistency_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    x_proposal: jax.Array | None = None,
) -> jax.Array:
    """Forward consistency loss for a single sample.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states.
        x_curr: Current states.
        params: Physical parameters.
        dt: Step length.
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``. The forward-
            consistency target always remains the unperturbed ``x_curr``.

    Returns:
        Scalar mean squared error between the predicted and observed next state.
    """
    x_proposal = x_curr if x_proposal is None else x_proposal
    u = cell.inverse_dynamics(x_prev, x_proposal, params)
    x_pred = cell.augmented_dynamics.integrate(x_prev, u, params, dt)
    return jnp.mean((x_pred - x_curr) ** 2)


def inverse_consistency_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    u_sampled: jax.Array,
    params: jax.Array,
    dt: float,
) -> jax.Array:
    """Inverse consistency loss for a single sample.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous state.
        u_sampled: Sampled control.
        params: Physical parameters.
        dt: Step length.

    Returns:
        Scalar mean squared control-recovery error.
    """
    x_next = cell.augmented_dynamics.integrate(x_prev, u_sampled, params, dt)
    u_recovered = cell.inverse_dynamics(x_prev, x_next, params)
    return jnp.mean((u_recovered - u_sampled) ** 2)


def minimum_norm_loss(
    cell: MaDELike,
    x: jax.Array,
    u: jax.Array,
    params: jax.Array,
) -> jax.Array:
    """Residual minimum-norm penalty for a single sample.

    Args:
        cell: Model or cell being trained.
        x: State.
        u: Control.
        params: Physical parameters.

    Returns:
        Scalar squared residual norm.
    """
    return cell.augmented_dynamics.residual_norm(x, u, params)


def inverse_residual_norm(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    x_proposal: jax.Array | None = None,
) -> jax.Array:
    """Squared magnitude of the learned inverse component, Delta I.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states.
        x_curr: Current states.
        params: Physical parameters.
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.

    Returns:
        Scalar squared norm.
    """
    x_proposal = x_curr if x_proposal is None else x_proposal
    return cell.inverse_dynamics.residual_norm(x_prev, x_proposal, params)


def inequality_violation_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    x_proposal: jax.Array | None = None,
) -> jax.Array:
    """Training-time inequality loss through the corrector.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states.
        x_curr: Current states.
        params: Physical parameters.
        dt: Step length.
        x_proposal: Optional perturbed state used as I-input to the cell. Defaults to ``x_curr``.

    Returns:
        Scalar squared constraint violation after correction.
    """
    x_proposal = x_curr if x_proposal is None else x_proposal
    x_corr, u_corr = cell(
        x_prev, x_proposal, params, dt, training=True, correction_mode="train_fixed"
    )
    violation = cell.constraints.violation(x_corr, u_corr)
    violation = _maybe_assert_xy_zero(violation)
    return jnp.sum(violation ** 2)


def _phase1_components(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Single-sample Phase 1 loss terms.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states.
        x_curr: Current states.
        params: Physical parameters.
        dt: Step length.
        u_sampled: Sampled controls.
        config: Training configuration.
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.

    Returns:
        Tuple ``(total, metrics)`` for one sample.
    """
    l_fwd = forward_consistency_loss(cell, x_prev, x_curr, params, dt, x_proposal=x_proposal)
    l_inv = inverse_consistency_loss(cell, x_prev, u_sampled, params, dt)
    l_norm = minimum_norm_loss(cell, x_prev, u_sampled, params)
    l_delta_i = inverse_residual_norm(cell, x_prev, x_curr, params, x_proposal=x_proposal)
    # lambda_delta_i_norm is part of the headline phase 1 total so `phase1_loss` /
    # `phase2_loss` (and the validation loss + early-stopping criterion) reflect the same
    # regularizer pressure as I-side training. On the T-side it is a constant offset (zero
    # gradient) since `phase1_t_loss` / `phase2_t_loss` route through `stop_i_side`,
    # freezing the inverse-dynamics leaves.
    total = (
        l_fwd
        + config.lambda_inv_consistency * l_inv
        + config.lambda_min_norm * l_norm
        + config.lambda_delta_i_norm * l_delta_i
    )
    metrics = {
        "forward_consistency": l_fwd,
        "inverse_consistency": l_inv,
        "minimum_norm": l_norm,
        "delta_i_norm": l_delta_i,
    }
    return total, metrics


def _phase2_components(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
    """Single-sample Phase 2 components: Phase 1 terms + inequality violation in one pass.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states.
        x_curr: Current states.
        params: Physical parameters.
        dt: Step length.
        u_sampled: Sampled controls.
        config: Training configuration.
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.

    Returns:
        Tuple ``(phase1_total, phase1_metrics, inequality_loss)`` for one sample.
    """
    phase1_total, phase1_metrics = _phase1_components(
        cell, x_prev, x_curr, params, dt, u_sampled, config, x_proposal=x_proposal
    )
    l_ineq = inequality_violation_loss(cell, x_prev, x_curr, params, dt, x_proposal=x_proposal)
    return phase1_total, phase1_metrics, l_ineq


def phase1_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Batch-reduced Phase 1 loss.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal
    totals, metrics = jax.vmap(
        lambda x_p, x_c, p, u_s, x_pr: _phase1_components(
            cell, x_p, x_c, p, dt, u_s, config, x_proposal=x_pr
        )
    )(x_prev, x_curr, params, u_sampled, x_proposal_for_vmap)
    return jnp.mean(totals), {key: jnp.mean(value) for key, value in metrics.items()}


def phase1_i_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """I-side Phase 1 loss with T/encoder leaves stopped.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    # lambda_delta_i_norm is part of `_phase1_components` and propagates into this
    # total via `phase1_loss`; do not add it again here.
    return phase1_loss(
        stop_t_side(cell),
        x_prev,
        x_curr,
        params,
        dt,
        u_sampled,
        config,
        x_proposal=x_proposal,
    )


def phase1_t_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """T-side Phase 1 loss without inverse-consistency gradients.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    stopped = stop_i_side(cell)
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal

    def _components(
        x_p: jax.Array, x_c: jax.Array, p: jax.Array, u_s: jax.Array, x_pr: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Per-sample T-side Phase 1 loss terms.

        Args:
            x_p: Previous state.
            x_c: Current state.
            p: Physical parameters.
            u_s: Sampled control.
            x_pr: Perturbed state used as I-input.

        Returns:
            Tuple ``(total, metrics)`` for one sample.
        """
        l_fwd = forward_consistency_loss(stopped, x_p, x_c, p, dt, x_proposal=x_pr)
        l_inv_metric = inverse_consistency_loss(stopped, x_p, u_s, p, dt)
        l_norm = minimum_norm_loss(stopped, x_p, u_s, p)
        l_delta_i = inverse_residual_norm(stopped, x_p, x_c, p, x_proposal=x_pr)
        total = l_fwd + config.lambda_min_norm * l_norm
        return total, {
            "forward_consistency": l_fwd,
            "inverse_consistency": l_inv_metric,
            "minimum_norm": l_norm,
            "delta_i_norm": l_delta_i,
        }

    totals, metrics = jax.vmap(_components)(x_prev, x_curr, params, u_sampled, x_proposal_for_vmap)
    return jnp.mean(totals), {key: jnp.mean(value) for key, value in metrics.items()}


def phase2_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Batch-reduced Phase 2 loss.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal
    totals, metrics_batched, l_ineqs = jax.vmap(
        lambda x_p, x_c, p, u_s, x_pr: _phase2_components(
            cell, x_p, x_c, p, dt, u_s, config, x_proposal=x_pr
        )
    )(x_prev, x_curr, params, u_sampled, x_proposal_for_vmap)
    l_ineq = jnp.mean(l_ineqs)
    total = jnp.mean(totals) + config.lambda_ineq * l_ineq
    metrics = {key: jnp.mean(value) for key, value in metrics_batched.items()}
    metrics["inequality_violation"] = l_ineq
    return total, metrics


def phase2_i_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """I-side Phase 2 loss with T/encoder leaves stopped.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    stopped = stop_t_side(cell)
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal
    # lambda_delta_i_norm propagates through _phase2_components -> _phase1_components.
    totals, metrics_batched, l_ineqs = jax.vmap(
        lambda x_p, x_c, p, u_s, x_pr: _phase2_components(
            stopped, x_p, x_c, p, dt, u_s, config, x_proposal=x_pr
        )
    )(x_prev, x_curr, params, u_sampled, x_proposal_for_vmap)
    l_ineq = jnp.mean(l_ineqs)
    total = jnp.mean(totals) + config.lambda_ineq * l_ineq
    metrics = {key: jnp.mean(value) for key, value in metrics_batched.items()}
    metrics["inequality_violation"] = l_ineq
    return total, metrics


def phase2_t_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """T/encoder Phase 2 loss without inverse-consistency gradients.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input. Defaults to
            ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` maps loss-term names to batch means.
    """
    stopped = stop_i_side(cell)
    phase1_total, phase1_metrics = phase1_t_loss(
        stopped,
        x_prev,
        x_curr,
        params,
        dt,
        u_sampled,
        config,
        x_proposal=x_proposal,
    )
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal
    l_ineq = jnp.mean(
        jax.vmap(
            lambda x_p, x_c, p, x_pr: inequality_violation_loss(
                stopped, x_p, x_c, p, dt, x_proposal=x_pr
            )
        )(x_prev, x_curr, params, x_proposal_for_vmap)
    )
    metrics = dict(phase1_metrics)
    metrics["inequality_violation"] = l_ineq
    # Isolate the inequality gradient from the forward model. stop_i_side above
    # freezes the INVERSE model's leaves only, so without this the inequality term
    # reaches augmented_dynamics.residual (T_theta) with an unblocked gradient. The
    # logged metric is unchanged either way; only the gradient is cut.
    l_ineq_for_total = (
        jax.lax.stop_gradient(l_ineq) if config.isolate_ineq_gradient else l_ineq
    )
    return phase1_total + config.lambda_ineq * l_ineq_for_total, metrics


def targeted_phase_loss(
    cell: MaDELike,
    x_prev: jax.Array,
    x_curr: jax.Array,
    params: jax.Array,
    dt: float,
    u_sampled: jax.Array,
    config: TrainingConfig,
    *,
    phase: int,
    target: str,
    x_proposal: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Dispatch to target-specific losses for alternating training.

    Args:
        cell: Model or cell being trained.
        x_prev: Previous states of shape ``(batch, state_dim)``.
        x_curr: Current states.
        params: Physical parameters per sample.
        dt: Step length.
        u_sampled: Sampled controls per sample.
        config: Training configuration.
        phase: Training phase, 1 or 2.
        target: Side being trained, ``"I"`` or ``"T"``.
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.

    Returns:
        Tuple ``(loss, metrics)`` of the selected phase/target loss.

    Raises:
        ValueError: If the phase/target combination is unsupported.
    """
    if phase == 1 and target == "I":
        return phase1_i_loss(
            cell, x_prev, x_curr, params, dt, u_sampled, config, x_proposal=x_proposal
        )
    if phase == 1 and target == "T":
        return phase1_t_loss(
            cell, x_prev, x_curr, params, dt, u_sampled, config, x_proposal=x_proposal
        )
    if phase == 2 and target == "I":
        return phase2_i_loss(
            cell, x_prev, x_curr, params, dt, u_sampled, config, x_proposal=x_proposal
        )
    if phase == 2 and target == "T":
        return phase2_t_loss(
            cell, x_prev, x_curr, params, dt, u_sampled, config, x_proposal=x_proposal
        )
    raise ValueError(f"Unsupported phase/target combination: phase={phase}, target={target}.")
