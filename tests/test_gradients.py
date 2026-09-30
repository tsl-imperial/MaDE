# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.training import inequality_violation_loss, phase1_loss
from made.training.losses import phase1_i_loss, phase1_t_loss
from made.utils import TrainingConfig


def test_phase1_gradients_finite(small_cell):
    config = TrainingConfig()
    batch = {
        "x_prev": jnp.zeros((4, 4)),
        "x_curr": jnp.ones((4, 4)) * 0.1,
        "params": jnp.zeros((4, 0)),
        "u_gt": jnp.zeros((4, 2)),
    }
    (_, _), grads = eqx.filter_value_and_grad(
        lambda cell: phase1_loss(
            cell,
            batch["x_prev"],
            batch["x_curr"],
            batch["params"],
            0.1,
            batch["u_gt"],
            config,
        ),
        has_aux=True,
    )(small_cell)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_phase1_i_loss_gradients_finite(small_cell):
    """I-side target loss gradients must be finite — validates target-specific loss wiring."""
    config = TrainingConfig()
    x_prev = jnp.zeros((4, 4))
    x_curr = jnp.ones((4, 4)) * 0.1
    params = jnp.zeros((4, 0))
    u_sampled = jnp.zeros((4, 2))
    (_, _), grads = eqx.filter_value_and_grad(
        lambda cell: phase1_i_loss(cell, x_prev, x_curr, params, 0.1, u_sampled, config),
        has_aux=True,
    )(small_cell)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_phase1_t_loss_gradients_finite(small_cell):
    """T-side target loss gradients must be finite — validates target-specific loss wiring."""
    config = TrainingConfig()
    x_prev = jnp.zeros((4, 4))
    x_curr = jnp.ones((4, 4)) * 0.1
    params = jnp.zeros((4, 0))
    u_sampled = jnp.zeros((4, 2))
    (_, _), grads = eqx.filter_value_and_grad(
        lambda cell: phase1_t_loss(cell, x_prev, x_curr, params, 0.1, u_sampled, config),
        has_aux=True,
    )(small_cell)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_inequality_gradient_finite(small_cell):
    grads = eqx.filter_grad(
        lambda cell: inequality_violation_loss(
            cell,
            jnp.zeros((4,)),
            jnp.ones((4,)) * 0.1,
            jnp.zeros((0,)),
            0.1,
        )
    )(small_cell)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_dynamic_bicycle_known_prior_gradient_finite():
    """Gradients through DynamicBicycle.known_control_prior are finite, non-NaN.

    Covers two autodiff paths:
    1. Grad w.r.t. x_prev (state input).
    2. Grad w.r.t. params (physics parameters).
    """
    from made.physics import DynamicBicycle, resolve_params

    physics = DynamicBicycle()
    params = resolve_params("dynamic_bicycle", {})
    dt = 0.1
    x_prev = jnp.array([0.0, 0.0, 0.0, 8.0, 0.2, 0.05], dtype=jnp.float64)
    x_curr = jnp.array([0.08, 0.0, 0.015, 8.15, 0.22, 0.06], dtype=jnp.float64)

    # Grad w.r.t. x_prev.
    def loss_xp(xp):
        return physics.known_control_prior(xp, x_curr, params, dt).sum()

    val_xp, grad_xp = eqx.filter_value_and_grad(loss_xp)(x_prev)
    assert jnp.isfinite(val_xp), f"Loss (x_prev grad) is non-finite: {val_xp}"
    assert jnp.all(jnp.isfinite(grad_xp)), f"Grad w.r.t. x_prev has non-finite values: {grad_xp}"

    # Grad w.r.t. params.
    def loss_p(p):
        return physics.known_control_prior(x_prev, x_curr, p, dt).sum()

    val_p, grad_p = eqx.filter_value_and_grad(loss_p)(params)
    assert jnp.isfinite(val_p), f"Loss (params grad) is non-finite: {val_p}"
    assert jnp.all(jnp.isfinite(grad_p)), f"Grad w.r.t. params has non-finite values: {grad_p}"
