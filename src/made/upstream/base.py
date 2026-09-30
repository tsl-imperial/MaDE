"""Base interface for upstream predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
import jax


class UpstreamPredictor(eqx.Module, ABC):
    """Abstract upstream predictor."""

    @abstractmethod
    def __call__(self, context: jax.Array) -> jax.Array:
        """Map context features to a predicted state trajectory."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the output state dimension."""
