# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""inD data adapter for the MaDE trainer.

Thin adapter over ``made.data.ind.load_ind_split`` / ``made.data.ind.iter_recordings``.
Converts their NumPy arrays into float64 JAX arrays in the ``(states, metadata, lengths)`` format expected by
``TrajectoryWindowTransform`` / ``create_data_loader``.

Schema consumed:
  ``metadata`` columns: [length, width, car_onehot, truck_bus_onehot, location_id]
  ``M = 5``  (2 float + 2 class one-hot + 1 location-id integer column)
  ``location_id ∈ {1, 2, 3, 4}`` (vendor canonical IDs, NOT zero-indexed).

The location embedding lookup (``id - 1`` → embedding row) is performed internally by
``MetadataEncoder`` when ``num_locations > 0``; this adapter does **not** zero-index the IDs.

If ``made.data.ind.load_ind_split`` raises ``NotImplementedError``, run the preprocessor first::

    python -m made.data.ind.preprocess \\
        --raw-dir data/inD-dataset \\
        --output-dir data/inD-preprocessed/v1

:func:`load_ind_split_stub` provides a stub path for unit tests without the real dataset.
"""

from __future__ import annotations

from typing import Any, Iterator

import jax.numpy as jnp
import numpy as np

__all__ = [
    "IND_LOCATION_ID_INDEX",
    "IND_METADATA_DIM",
    "IND_NUM_LOCATIONS",
    "IND_STATE_DIM",
    "create_ind_data_source",
    "load_ind_split",
    "load_ind_split_stub",
]

# Metadata columns: [length, width, class_one_hot[2], location_id]
IND_METADATA_DIM: int = 5
IND_LOCATION_ID_INDEX: int = 4  # trailing column
IND_NUM_LOCATIONS: int = 4  # vendor IDs {1..4}
IND_STATE_DIM: int = 4  # (x, y, theta_rad, v)


def load_ind_split(
    data_dir: str,
    split: str,
    *,
    location_filter: list[int] | None = None,
) -> tuple[Any, Any, Any]:
    """Load a preprocessed inD split via ``made.data.ind.load_ind_split``, casting
    states/metadata to float64 and lengths to int32.

    Args:
        data_dir: Path to ``data/inD-preprocessed/v1/`` (or the preprocessed root).
        location_filter: Optional list of location IDs to keep (vendor IDs 1-4).

    Raises:
        NotImplementedError: If the preprocessor has not been run yet.

    Returns:
        Tuple (states, metadata, lengths) as float64, float64 and int32 arrays.
    """
    from made.data.ind import load_ind_split as _plan0_load

    bundle = _plan0_load(data_dir, split, location_filter=location_filter)
    states = jnp.asarray(bundle.states, dtype=jnp.float64)
    metadata = jnp.asarray(bundle.metadata, dtype=jnp.float64)
    lengths = jnp.asarray(bundle.lengths, dtype=jnp.int32)
    return states, metadata, lengths


def iter_ind_recordings(
    data_dir: str,
    split: str,
) -> Iterator[dict[str, Any]]:
    """Iterate per-recording trajectory bundles from the preprocessed inD dataset.

    Thin wrapper over ``made.data.ind.iter_recordings``, converting arrays to
    float64 JAX arrays.

    Yields:
        Dicts with keys: ``states``, ``metadata``, ``lengths``, ``location_id``,
        ``recording_id``, ``track_ids``, ``start_frame_native``.

    Args:
        data_dir: Preprocessed inD root.
        split: Split name.
    """
    from made.data.ind import iter_recordings as _plan0_iter

    for bundle in _plan0_iter(data_dir, split):
        yield {
            "states": jnp.asarray(bundle.states, dtype=jnp.float64),
            "metadata": jnp.asarray(bundle.metadata, dtype=jnp.float64),
            "lengths": jnp.asarray(bundle.lengths, dtype=jnp.int32),
            "location_id": bundle.location_id,
            "recording_id": bundle.recording_id,
            "track_ids": jnp.asarray(bundle.track_ids, dtype=jnp.int32),
            "start_frame_native": jnp.asarray(bundle.start_frame_native, dtype=jnp.int32),
        }


def load_ind_split_stub(
    num_trajectories: int = 16,
    trajectory_length: int = 20,
    *,
    seed: int = 0,
) -> tuple[Any, Any, Any]:
    """Generate a synthetic inD-shaped split for testing (no real data required).

    Produces random float64 states and metadata conforming to the inD schema:
      - states: ``[num_trajectories, trajectory_length, 4]`` float64
      - metadata: ``[num_trajectories, 5]`` float64 with valid location IDs
      - lengths: ``[num_trajectories]`` int32 equal to trajectory_length

    Location IDs are sampled uniformly from {1, 2, 3, 4}.

    Args:
        num_trajectories: Number of trajectories.
        trajectory_length: Length of each trajectory.
        seed: Seed for the random generator.
    Returns:
        Tuple (states, metadata, lengths).
    """
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((num_trajectories, trajectory_length, IND_STATE_DIM))
    states = states.astype(np.float64)

    # Build metadata: [length_m, width_m, car_oh, truck_bus_oh, location_id]
    lengths_m = rng.uniform(3.5, 5.5, size=(num_trajectories, 1))
    widths_m = rng.uniform(1.5, 2.5, size=(num_trajectories, 1))
    # Random vehicle class (0=car, 1=truck_bus)
    cls_idx = rng.integers(0, 2, size=num_trajectories)
    cls_onehot = np.zeros((num_trajectories, 2), dtype=np.float64)
    cls_onehot[np.arange(num_trajectories), cls_idx] = 1.0
    # Location IDs from {1, 2, 3, 4}
    loc_ids = rng.integers(1, 5, size=(num_trajectories, 1)).astype(np.float64)
    metadata = np.concatenate([lengths_m, widths_m, cls_onehot, loc_ids], axis=1).astype(
        np.float64
    )

    lengths = np.full(num_trajectories, trajectory_length, dtype=np.int32)
    return (
        jnp.asarray(states),
        jnp.asarray(metadata),
        jnp.asarray(lengths),
    )


def create_ind_data_source(
    data_dir: str,
    split: str,
    *,
    location_filter: list[int] | None = None,
    use_stub: bool = False,
    stub_num_trajectories: int = 64,
    stub_trajectory_length: int = 20,
    stub_seed: int = 0,
) -> tuple[Any, Any, Any]:
    """Unified entry point for inD data loading.

    When ``use_stub=True``, returns synthetic data via :func:`load_ind_split_stub`
    (for smoke tests / CI where the real preprocessed data is unavailable).
    Otherwise delegates to :func:`load_ind_split`.

    Args:
        data_dir: Preprocessed inD root.
        split: Split name.
        location_filter: Optional location ids to keep.
        use_stub: Return synthetic data instead of the real split.
        stub_num_trajectories: Trajectory count for the stub.
        stub_trajectory_length: Trajectory length for the stub.
        stub_seed: Seed for the stub.
    Returns:
        Tuple (states, metadata, lengths).
    """
    if use_stub:
        return load_ind_split_stub(
            stub_num_trajectories,
            stub_trajectory_length,
            seed=stub_seed,
        )
    return load_ind_split(data_dir, split, location_filter=location_filter)
