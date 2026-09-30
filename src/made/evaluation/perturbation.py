"""Perturbation helpers for robustness evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from made.utils.config import DataConfig


def perturb_trajectories(
    states: jax.Array,
    config: DataConfig,
    key: jax.Array,
    *,
    state_bounds: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Perturb trajectories according to the configured protocol.

    Args:
        states: Array of shape (N, T, state_dim).
        config: DataConfig carrying perturbation_type and perturbation_scale.
        key: PRNG key (used by stochastic types; ignored by deterministic ones).
        state_bounds: Optional ``(state_min, state_max)`` tuple of arrays with shape
            ``(state_dim,)``.  Required when ``config.perturbation_type ==
            "bound_violation"``; ignored by all other perturbation types.

    Returns:
        Perturbed states array with the same shape as *states*.
    """
    if config.perturbation_scale == 0.0:
        return states

    if config.perturbation_type == "gaussian":
        return states + config.perturbation_scale * jax.random.normal(key, states.shape)
    if config.perturbation_type == "uniform":
        noise = jax.random.uniform(key, states.shape) * 2.0 - 1.0
        return states + config.perturbation_scale * noise
    if config.perturbation_type == "per_dimension":
        scale = 1.0 + config.perturbation_scale * jax.random.normal(key, states.shape)
        return states * scale
    if config.perturbation_type == "bound_violation":
        if state_bounds is None:
            raise ValueError(
                "perturbation_type='bound_violation' requires state_bounds=(state_min, state_max). "
                "Pass them as a keyword argument to perturb_trajectories."
            )
        # ±inf range entries (unbounded dims like x,y on inD/field-data) result
        # in ZERO perturbation on that dimension. Callers wanting nonzero
        # positional perturbation must pass a finite-range box.
        state_min, state_max = state_bounds
        range_ = state_max - state_min
        safe_range = jnp.where(jnp.isfinite(range_), range_, 0.0)
        midpoint = 0.5 * (state_min + state_max)
        sign = jnp.where(states >= midpoint, 1.0, -1.0)
        if config.perturbation_scale_per_dim is not None:
            scale = jnp.asarray(config.perturbation_scale_per_dim, dtype=states.dtype)
            if scale.shape[0] != states.shape[-1]:
                raise ValueError(
                    f"perturbation_scale_per_dim length {scale.shape[0]} "
                    f"does not match state_dim {states.shape[-1]}"
                )
        else:
            scale = config.perturbation_scale
        return states + sign * scale * safe_range
    raise ValueError(f"Unknown perturbation type: {config.perturbation_type}")


def add_observation_noise(states: jax.Array, noise_scale: float, key: jax.Array) -> jax.Array:
    """Add zero-mean Gaussian observation noise."""
    if noise_scale == 0.0:
        return states
    return states + noise_scale * jax.random.normal(key, states.shape)
