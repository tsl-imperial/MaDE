# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Shared input-feature canonicalisation for upstream predictors."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

# Per-timestep feature layout produced by canonicalise_window:
#   [(x - x_last)/sx, (y - y_last)/sy, cos(theta), sin(theta), (v - mv)/sv]
WINDOW_FEATURE_DIM: int = 5


def static_metadata_features(
    metadata: jax.Array,
    location_embedding: Callable[[jax.Array], jax.Array],
    location_id_index: int,
) -> jax.Array:
    """Build the static conditioning vector from the inD 5-vec metadata.

    Layout: raw non-location columns (length, width, class one-hot) followed by
    the location embedding row for the vendor ID (``{1..N}`` → ``id - 1``).

    Args:
        metadata: Metadata vector, shape (5,).
        location_embedding: Embedding looked up by location id.
        location_id_index: Metadata column holding the location id.
    Returns:
        Static conditioning vector.
    """
    location_row = metadata[location_id_index].astype(jnp.int32) - 1
    embedded = location_embedding(location_row)
    return jnp.concatenate([metadata[:location_id_index], embedded])


def canonicalise_window(
    states: jax.Array,
    state_mean: jax.Array,
    state_std: jax.Array,
) -> jax.Array:
    """Map a raw context window ``[H, 4]`` to translation-invariant features ``[H, 5]``.

    Positions enter as offsets to the *last* context state so the predictor never
    sees absolute map coordinates; heading enters as cos/sin to avoid the ±π wrap.
    ``state_mean``/``state_std`` are frozen train-split statistics, not trainable
    parameters — they are stop-gradiented here so optimisers cannot drift them.

    Args:
        states: Raw context window, shape (H, 4).
        state_mean: Train-split state mean.
        state_std: Train-split state std.
    Returns:
        Features, shape (H, 5).
    """
    mean = jax.lax.stop_gradient(state_mean)
    std = jax.lax.stop_gradient(state_std)
    rel_xy = (states[:, :2] - states[-1, :2]) / std[:2]
    theta = states[:, 2]
    v = (states[:, 3:4] - mean[3:4]) / std[3:4]
    return jnp.concatenate(
        [rel_xy, jnp.cos(theta)[:, None], jnp.sin(theta)[:, None], v],
        axis=1,
    )
