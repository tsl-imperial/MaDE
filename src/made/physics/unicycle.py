# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Unicycle dynamics."""

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class Unicycle(PhysicsModel):
    """Unicycle model with turn-rate and acceleration control."""

    @property
    def state_dim(self) -> int:
        """State dimension.

        Returns:
            State dimension.
        """
        return 4

    @property
    def control_dim(self) -> int:
        """Control dimension.

        Returns:
            Control dimension.
        """
        return 2

    @property
    def param_dim(self) -> int:
        """Parameter dimension.

        Returns:
            Parameter dimension.
        """
        return 0

    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        """Return the unicycle derivative.

        Args:
            state: State vector.
            control: Control vector.
            params: Physical parameters.
            t: Time (unused).

        Returns:
            State derivative.
        """
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
        """Exact finite-step inverse: turn rate ``dtheta / dt`` and acceleration ``dv / dt``.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.
            dt: Step length.

        Returns:
            Control estimate.
        """
        del params
        # Exact finite-step inverse: delta = dtheta/dt, accel = dv/dt
        delta = (x_curr[2] - x_prev[2]) / dt
        accel = (x_curr[3] - x_prev[3]) / dt
        return jnp.array([delta, accel], dtype=x_prev.dtype)
