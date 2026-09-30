"""Loss functions for MaDE training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from made.models import MaDECell, MaDEModel
from made.physics import _maybe_assert_xy_zero
from made.utils import TrainingConfig

MaDELike = MaDECell | MaDEModel


def _stop_arrays(tree):
    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        tree,
    )


def _cell(model: MaDELike) -> MaDECell:
    return model.cell if isinstance(model, MaDEModel) else model


def _replace_cell(model: MaDELike, cell: MaDECell) -> MaDELike:
    if isinstance(model, MaDEModel):
        return eqx.tree_at(lambda made_model: made_model.cell, model, cell)
    return cell


def stop_i_side(model: MaDELike) -> MaDELike:
    """Stop inverse-dynamics leaves while preserving forward values."""
    cell = _cell(model)
    stopped_cell = eqx.tree_at(
        lambda made_cell: made_cell.inverse_dynamics,
        cell,
        _stop_arrays(cell.inverse_dynamics),
    )
    return _replace_cell(model, stopped_cell)


def stop_t_side(model: MaDELike) -> MaDELike:
    """Stop residual/encoder leaves while preserving forward values."""
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
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.
            The forward-consistency target always remains the unperturbed ``x_curr``.
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
    """Inverse consistency loss for a single sample."""
    x_next = cell.augmented_dynamics.integrate(x_prev, u_sampled, params, dt)
    u_recovered = cell.inverse_dynamics(x_prev, x_next, params)
    return jnp.mean((u_recovered - u_sampled) ** 2)


def minimum_norm_loss(
    cell: MaDELike,
    x: jax.Array,
    u: jax.Array,
    params: jax.Array,
) -> jax.Array:
    """Residual minimum-norm penalty for a single sample."""
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
        x_proposal: Optional perturbed state used as I-input. Defaults to ``x_curr``.
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
        x_proposal: Optional perturbed state used as I-input to the cell. Defaults to ``x_curr``.
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
    l_fwd = forward_consistency_loss(cell, x_prev, x_curr, params, dt, x_proposal=x_proposal)
    l_inv = inverse_consistency_loss(cell, x_prev, u_sampled, params, dt)
    l_norm = minimum_norm_loss(cell, x_prev, u_sampled, params)
    l_delta_i = inverse_residual_norm(cell, x_prev, x_curr, params, x_proposal=x_proposal)
    # NOTE: lambda_delta_i_norm is now part of the headline phase 1 total so that
    # `phase1_loss` / `phase2_loss` (and therefore the validation loss + early-
    # stopping criterion) reflect the same regularizer pressure that the I-side
    # training is applying. On the T-side the term contributes only a constant
    # offset (zero gradient) because `phase1_t_loss` / `phase2_t_loss` route the
    # cell through `stop_i_side`, freezing the inverse-dynamics leaves.
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
    """Single-sample Phase 2 components: Phase 1 terms + inequality violation in one pass."""
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
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input.
            Defaults to ``x_curr`` (byte-identical behaviour).
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
    """I-side Phase 1 loss with T/encoder leaves stopped."""
    # NOTE: lambda_delta_i_norm is part of `_phase1_components` and propagates
    # into this total automatically via `phase1_loss`. The previous explicit
    # addition here would now double-count and has been removed.
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
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input.
            Defaults to ``x_curr`` (byte-identical behaviour).
    """
    stopped = stop_i_side(cell)
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal

    def _components(x_p, x_c, p, u_s, x_pr):
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
        x_proposal: Optional ``(batch, state_dim)`` perturbed state used as I-input.
            Defaults to ``x_curr`` (byte-identical behaviour).
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
    """I-side Phase 2 loss with T/encoder leaves stopped."""
    stopped = stop_t_side(cell)
    x_proposal_for_vmap = x_curr if x_proposal is None else x_proposal
    # NOTE: lambda_delta_i_norm propagates through _phase2_components → _phase1_components.
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
    """T/encoder Phase 2 loss without inverse-consistency gradients."""
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
    """Dispatch to target-specific losses for alternating training."""
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
