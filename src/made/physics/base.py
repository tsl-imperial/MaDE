# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Base abstractions for physics models and constraint sets."""

from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class PhysicsModel(eqx.Module, ABC):
    """Abstract base class for single-sample continuous-time dynamics."""

    @abstractmethod
    def vector_field(
        self,
        state: jax.Array,
        control: jax.Array,
        params: jax.Array,
        t: float,
    ) -> jax.Array:
        """Return the continuous-time derivative at one state-control pair.

        Args:
            state: State vector.
            control: Control vector.
            params: Physical parameters.
            t: Time.

        Returns:
            State derivative.
        """

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Return an analytical control estimate from consecutive states.

        Approximate or exact inverse of the known discrete transition. Subclasses override
        with a closed-form per-system formula.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            params: Physical parameters.
            dt: Step length.

        Returns:
            Control estimate.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement known_control_prior."
        )

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector.

        Returns:
            State dimension.
        """

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of the control vector.

        Returns:
            Control dimension.
        """

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Dimension of the learnable/known parameter vector.

        Returns:
            Parameter dimension.
        """

    @property
    def param_scales(self) -> jax.Array:
        """Characteristic scales per parameter dimension, for normalising raw params before
        learned MLPs. Ones by default; override when params span very different
        magnitudes (e.g. DynamicBicycle).

        Returns:
            Array of scales with shape ``(param_dim,)``.
        """
        return jnp.ones(self.param_dim)


class ConstraintSet(eqx.Module, ABC):
    """Abstract base class for inequality constraints."""

    @abstractmethod
    def __call__(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the raw inequality values, feasible when all are <= 0.

        Args:
            state: State vector.
            control: Control vector.

        Returns:
            Inequality values.
        """

    def violation(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the positive part of the constraint vector.

        Args:
            state: State vector.
            control: Control vector.

        Returns:
            ``relu`` of the constraint values.
        """
        return jax.nn.relu(self(state, control))
