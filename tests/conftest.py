import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from made.models import MaDECell
from made.physics import DoubleIntegrator, double_integrator_constraints
from made.utils import CorrectorConfig, ModelConfig
from made.utils.jax_setup import configure

configure()


@pytest.fixture
def small_key() -> jax.Array:
    return jax.random.key(0)


@pytest.fixture
def double_integrator():
    return DoubleIntegrator(), double_integrator_constraints()


@pytest.fixture
def small_cell(double_integrator, small_key: jax.Array) -> MaDECell:
    physics, constraints = double_integrator
    return MaDECell.from_config(
        physics,
        constraints,
        ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16)),
        CorrectorConfig(train_steps=2, eval_max_steps=4),
        key=small_key,
    )


@pytest.fixture
def sample_batch(double_integrator):
    physics, _ = double_integrator
    batch_size = 8
    x_prev = jnp.zeros((batch_size, physics.state_dim))
    x_curr = jnp.ones((batch_size, physics.state_dim)) * 0.1
    params = jnp.zeros((batch_size, physics.param_dim))
    u_gt = jnp.zeros((batch_size, physics.control_dim))
    return {"x_prev": x_prev, "x_curr": x_curr, "params": params, "u_gt": u_gt}
