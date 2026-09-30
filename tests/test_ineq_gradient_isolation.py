"""Y1 — the inequality term must not reach the forward model when isolation is on.

The defect this guards: `phase2_t_loss` calls `stop_i_side`, which freezes the INVERSE
model's leaves only. Without `isolate_ineq_gradient`, `lambda_ineq * L_ineq` reaches
`augmented_dynamics.residual` (T_theta) with an unblocked gradient, so the learned
residual is pulled toward constraint satisfaction rather than remaining the minimum-norm
complement to the known physics that the APHYNITY decomposition assumes.

The test is a gradient-identity check, not a value check: with isolation ON, the T-side
gradient w.r.t. the residual must be BIT-IDENTICAL across two different `lambda_ineq`
values. With isolation OFF it must differ -- which is what demonstrates the leak is real
and that the flag is what closes it.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from made.training.losses import phase2_t_loss
from made.utils.config import TrainingConfig


def _t_side_grad(cell, batch, cfg):
    """d(phase2_t_loss)/d(augmented_dynamics.residual), as the trainer takes it."""
    x_prev, x_curr, params, u_sampled, dt = batch

    def _loss(c):
        total, _ = phase2_t_loss(c, x_prev, x_curr, params, dt, u_sampled, cfg)
        return total

    grads = eqx.filter_grad(_loss)(cell)
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(grads.augmented_dynamics.residual, eqx.is_array)
    )
    assert leaves, "no residual gradient leaves found — the test would be vacuous"
    return leaves


@pytest.mark.parametrize("isolate,expect_identical", [(True, True), (False, False)])
def test_ineq_gradient_reaches_t_side_only_when_not_isolated(isolate, expect_identical):
    from made.models.made_cell import MaDECell
    from made.physics.constraints import kinematic_bicycle_constraints
    from made.physics.kinematic_bicycle import KinematicBicycle
    from made.utils.config import CorrectorConfig, ModelConfig

    cell = MaDECell.from_config(
        KinematicBicycle(),
        kinematic_bicycle_constraints(),
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16),
                    residual_init_scale=0.1),
        CorrectorConfig(step_size=0.1, momentum=0.0, train_steps=2,
                        eval_max_steps=2, eval_tol=-1e9, mode="enabled"),
        key=jax.random.key(0),
        dt=0.1,
    )

    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    b, sd, ud = 4, 4, 2
    x_prev = jax.random.normal(k1, (b, sd), dtype=jnp.float64)
    x_curr = x_prev + 0.01 * jax.random.normal(k2, (b, sd), dtype=jnp.float64)
    u_sampled = 0.01 * jax.random.normal(k3, (b, ud), dtype=jnp.float64)
    params = jnp.broadcast_to(jnp.asarray([2.7], dtype=jnp.float64), (b, 1))
    batch = (x_prev, x_curr, params, u_sampled, 0.1)

    base = TrainingConfig(isolate_ineq_gradient=isolate)
    g_lo = _t_side_grad(cell, batch, dataclasses.replace(base, lambda_ineq=0.0))
    g_hi = _t_side_grad(cell, batch, dataclasses.replace(base, lambda_ineq=1000.0))

    identical = all(
        bool(jnp.array_equal(a, b_)) for a, b_ in zip(g_lo, g_hi, strict=True)
    )
    if expect_identical:
        assert identical, (
            "isolate_ineq_gradient=True but the T-side residual gradient still changed "
            "with lambda_ineq — the inequality term is STILL reaching T_theta."
        )
    else:
        assert not identical, (
            "isolate_ineq_gradient=False yet the T-side residual gradient did NOT change "
            "with lambda_ineq — the leak this flag exists to close is not reproducible, "
            "so the test cannot demonstrate the flag does anything."
        )


def test_default_preserves_published_behaviour():
    """The published results were produced with the gradient REACHING T_theta."""
    assert TrainingConfig().isolate_ineq_gradient is False
