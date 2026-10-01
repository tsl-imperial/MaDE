# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Double-integrator dynamics."""

import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class DoubleIntegrator(PhysicsModel):
    """Planar double-integrator with acceleration control."""

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
        """Return the double-integrator derivative.

        Args:
            state: State vector.
            control: Control vector.
            params: Physical parameters.
            t: Time (unused).

        Returns:
            State derivative.
        """
        del params, t
        return jnp.array([state[2], state[3], control[0], control[1]], dtype=state.dtype)

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Exact acceleration prior ``(v_curr - v_prev) / dt``.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.
            dt: Step length.

        Returns:
            Control estimate.
        """
        del params
        # Exact: acceleration = Δvelocity / dt
        return (x_curr[2:4] - x_prev[2:4]) / dt
