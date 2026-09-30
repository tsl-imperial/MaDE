"""Double-integrator dynamics."""

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class DoubleIntegrator(PhysicsModel):
    """Planar double-integrator with acceleration control."""

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 0

    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        del params, t
        return jnp.array([state[2], state[3], control[0], control[1]], dtype=state.dtype)

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        del params
        # Exact: acceleration = Δvelocity / dt
        return (x_curr[2:4] - x_prev[2:4]) / dt
