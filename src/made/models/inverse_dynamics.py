# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Inverse-dynamics model."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from made.physics.base import PhysicsModel


class InverseDynamics(eqx.Module):
    """Infer controls from consecutive states and system parameters.

    Implements I(x_prev, x_curr, params) = I_known + ΔI_phi when known_physics
    is provided, mirroring the APHYNITY decomposition used in AugmentedDynamics.
    """

    mlp: Optional[eqx.nn.MLP]
    known_physics: Optional[PhysicsModel]
    dt: float
    use_residual: bool

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        param_dim: int,
        hidden: tuple[int, ...],
        *,
        key: jax.Array,
        known_physics: Optional[PhysicsModel] = None,
        dt: float = 0.1,
        use_residual: bool = True,
        init_scale: float = 0.0,
    ) -> None:
        """Build the inverse-dynamics model.

        Args:
            state_dim: State dimension.
            control_dim: Control dimension.
            param_dim: Physical-parameter dimension.
            hidden: Hidden layer widths.
            key: PRNG key for weight initialisation.
            known_physics: Physics model providing the known control prior.
            dt: Step length.
            use_residual: Add the learned component to the known prior.
            init_scale: Multiplier applied to the final layer at initialisation.
        """
        self.known_physics = known_physics
        self.dt = dt
        self.use_residual = use_residual

        build_mlp = use_residual or known_physics is None
        if build_mlp:
            width = hidden[0] if hidden else max(control_dim, 1)
            depth = len(hidden)
            mlp = eqx.nn.MLP(
                in_size=2 * state_dim + param_dim,
                out_size=control_dim,
                width_size=width,
                depth=depth,
                activation=jax.nn.relu,
                key=key,
            )
            if init_scale != 1.0:
                final = mlp.layers[-1]
                scaled_weight = final.weight * init_scale
                scaled_bias = None if final.bias is None else final.bias * init_scale
                mlp = eqx.tree_at(
                    lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                    mlp,
                    (scaled_weight, scaled_bias),
                )
            self.mlp = mlp
        else:
            self.mlp = None

    def __call__(self, x_prev: jax.Array, x_curr: jax.Array, params: jax.Array) -> jax.Array:
        """Infer the control that moves ``x_prev`` to ``x_curr``.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.

        Returns:
            Control vector.
        """
        if self.known_physics is not None:
            u_prior = self.known_physics.known_control_prior(x_prev, x_curr, params, self.dt)
            if self.use_residual:
                return u_prior + self.learned_component(x_prev, x_curr, params)
            return u_prior
        return self.learned_component(x_prev, x_curr, params)

    def learned_component(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
    ) -> jax.Array:
        """Return the learned inverse component, Delta I.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.

        Returns:
            Learned control component.
        """
        if self.mlp is None:
            u_prior = self.known_physics.known_control_prior(x_prev, x_curr, params, self.dt)
            return jnp.zeros_like(u_prior)
        params_norm = (
            params / self.known_physics.param_scales
            if self.known_physics is not None and self.known_physics.param_dim > 0
            else params
        )
        inputs = jnp.concatenate([x_prev, x_curr, params_norm])
        return self.mlp(inputs)

    def residual_norm(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
    ) -> jax.Array:
        """Squared magnitude of the learned inverse component, Delta I.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.

        Returns:
            Scalar squared norm.
        """
        delta_i = self.learned_component(x_prev, x_curr, params)
        return jnp.sum(delta_i ** 2)
