"""inD data adapter for Plan 1 — MaDE Experiment 2.

This module is a thin adapter over the Plan 0 loader API
(``made.data.ind.load_ind_split`` / ``made.data.ind.iter_recordings``).  It
converts the NumPy arrays returned by Plan 0 into float64 JAX arrays and
exposes them in the ``(states, metadata, lengths)`` format expected by the
existing ``TrajectoryWindowTransform`` / ``create_data_loader`` pipeline.

Schema consumed (Plan 0 §3):
  ``metadata`` columns: [length, width, car_onehot, truck_bus_onehot, location_id]
  ``M = 5``  (2 float + 2 class one-hot + 1 location-id integer column)
  ``location_id ∈ {1, 2, 3, 4}`` (vendor canonical IDs, NOT zero-indexed).

The location embedding lookup (``id - 1`` → embedding row) is performed
internally by ``MetadataEncoder`` when ``num_locations > 0``; this adapter
does **not** zero-index the IDs.

Plan 0 status:
  If ``made.data.ind.load_ind_split`` raises ``NotImplementedError`` (Plan 0
  stub not yet replaced), callers should run the Plan 0 preprocessor first::

      python -m made.data.ind.preprocess \\
          --raw-dir data/inD-dataset \\
          --output-dir data/inD-preprocessed/v1

  A stub path is provided via :func:`load_ind_split_stub` for unit tests that
  do not need the real dataset.
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

# Metadata column count per Plan 0 §3:
#   [length, width, class_one_hot[2], location_id]
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
    """Load a preprocessed inD split and return (states, metadata, lengths) as JAX float64 arrays.

    Consumes ``made.data.ind.load_ind_split`` (Plan 0) and performs:
      - float64 cast on states and metadata
      - int32 cast on lengths

    Args:
        data_dir: Path to ``data/inD-preprocessed/v1/`` (or the preprocessed root).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        location_filter: Optional list of location IDs to keep (vendor IDs 1–4).

    Returns:
        Tuple of (states, metadata, lengths) as JAX arrays.

    Raises:
        NotImplementedError: If Plan 0 has not been run yet.
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

    Thin wrapper over ``made.data.ind.iter_recordings`` (Plan 0).  Converts
    arrays to float64 JAX arrays.

    Yields:
        Dicts with keys: ``states``, ``metadata``, ``lengths``, ``location_id``,
        ``recording_id``, ``track_ids``, ``start_frame_native``.
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
    """
    if use_stub:
        return load_ind_split_stub(
            stub_num_trajectories,
            stub_trajectory_length,
            seed=stub_seed,
        )
    return load_ind_split(data_dir, split, location_filter=location_filter)
