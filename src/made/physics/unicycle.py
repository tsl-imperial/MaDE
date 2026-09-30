"""Unicycle dynamics."""

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class Unicycle(PhysicsModel):
    """Unicycle model with turn-rate and acceleration control."""

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
        theta = state[2]
        velocity = state[3]
        delta = control[0]
        accel = control[1]
        return jnp.array(
            [
                velocity * jnp.cos(theta),
                velocity * jnp.sin(theta),
                delta,
                accel,
            ],
            dtype=state.dtype,
        )

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        del params
        # Exact finite-step inverse: delta = dtheta/dt, accel = dv/dt
        delta = (x_curr[2] - x_prev[2]) / dt
        accel = (x_curr[3] - x_prev[3]) / dt
        return jnp.array([delta, accel], dtype=x_prev.dtype)
