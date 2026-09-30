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
        """Return the continuous-time derivative at one state-control pair."""

    def known_control_prior(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        params: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Return an analytical control estimate from consecutive states.

        This is an approximate or exact inverse of the known discrete transition.
        Subclasses should override this with a closed-form per-system formula.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement known_control_prior."
        )

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of the control vector."""

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Dimension of the learnable/known parameter vector."""

    @property
    def param_scales(self) -> jax.Array:
        """Characteristic scales for each parameter dimension.

        Used to normalise raw param values before they enter learned MLPs.
        Returns ones by default (no normalisation). Override in subclasses
        whose params span very different magnitudes (e.g. DynamicBicycle).
        """
        return jnp.ones(self.param_dim)


class ConstraintSet(eqx.Module, ABC):
    """Abstract base class for inequality constraints."""

    @abstractmethod
    def __call__(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the raw inequality values, feasible when all are <= 0."""

    def violation(self, state: jax.Array, control: jax.Array) -> jax.Array:
        """Return the positive part of the constraint vector."""
        return jax.nn.relu(self(state, control))
