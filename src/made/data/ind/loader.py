"""Loader API for preprocessed inD artefacts.

Public surface:
  - InDSplitArrays    dataclass — JAX arrays (consumed by MaDE).
  - InDRecordingBundle dataclass — numpy arrays.
  - load_ind_split(...) — JAX-array path; lazily imports jax.numpy.
  - iter_recordings(...) — numpy-only path; safe to import without JAX
    installed.

Importing this module does **not** trigger a JAX import; jax.numpy is
imported lazily inside ``load_ind_split`` so numpy-only consumers can call
``iter_recordings`` without ``ModuleNotFoundError``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from made.data.ind.constants import CONSTRUCTION_RECORDINGS


@dataclass(frozen=True)
class InDSplitArrays:
    """All trajectories from one split, loaded as JAX arrays.

    Attributes
    ----------
    states : jax.Array
        Float64 ``[N_traj, T_max, 4]`` — (x, y, θ_rad, v), zero-padded.
    metadata : jax.Array
        Float64 ``[N_traj, 5]`` — (length, width, one_hot_car, one_hot_truck_bus,
        location_id_float).
    lengths : jax.Array
        Int32 ``[N_traj]`` — valid prefix length per trajectory.
    location_ids : jax.Array
        Int32 ``[N_traj]`` — canonical location ID (1–4).
    recording_ids : jax.Array
        Int32 ``[N_traj]``.
    track_ids : jax.Array
        Int32 ``[N_traj]``.
    map_variants : jax.Array
        Uint8 ``[N_traj]`` — 0 default, 1 bendplatz construction.
    start_frame_native : jax.Array
        Int32 ``[N_traj]`` — first 25 Hz frame retained, for downstream alignment.
    """

    states: "jax.Array"            # [N_traj, T_max, 4] float64
    metadata: "jax.Array"          # [N_traj, 5]        float64
    lengths: "jax.Array"           # [N_traj]           int32
    location_ids: "jax.Array"      # [N_traj]           int32
    recording_ids: "jax.Array"     # [N_traj]           int32
    track_ids: "jax.Array"         # [N_traj]           int32
    map_variants: "jax.Array"      # [N_traj]           uint8
    start_frame_native: "jax.Array"  # [N_traj]         int32


@dataclass(frozen=True)
class InDRecordingBundle:
    """All trajectories from a single recording, for per-recording batching.

    Arrays are ``np.ndarray`` (not ``jax.Array``) so this bundle can be
    consumed without JAX installed. MaDE's wrapper in
    ``made.data.ind_data.iter_ind_recordings`` re-wraps to ``jnp.asarray``
    on its side.

    Attributes
    ----------
    recording_id : int
    location_id : int
    map_variant : int
        0 for default, 1 for bendplatz construction.
    states : np.ndarray
        Float64 ``[N_traj_in_recording, T_max_in_recording, 4]``.
    metadata : np.ndarray
        Float64 ``[N_traj_in_recording, 5]``.
    lengths : np.ndarray
        Int32 ``[N_traj_in_recording]``.
    track_ids : np.ndarray
        Int32 ``[N_traj_in_recording]``.
    start_frame_native : np.ndarray
        Int32 ``[N_traj_in_recording]``.
    """

    recording_id: int
    location_id: int
    map_variant: int
    states: np.ndarray
    metadata: np.ndarray
    lengths: np.ndarray
    track_ids: np.ndarray
    start_frame_native: np.ndarray


def _load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_split_arrays(
    split_dir: Path,
    location_filter: "list[int] | None" = None,
) -> tuple[np.ndarray, ...]:
    """Load raw numpy arrays from a split directory, with optional location filtering."""
    npz = np.load(split_dir / "trajectories.npz")
    states = npz["states"].astype(np.float64)
    lengths = npz["lengths"].astype(np.int32)
    recording_ids = npz["recording_ids"].astype(np.int32)
    track_ids = npz["track_ids"].astype(np.int32)
    location_ids = npz["location_ids"].astype(np.int32)
    map_variants = npz["map_variants"].astype(np.uint8)
    start_frame_native = npz["start_frame_native"].astype(np.int32)
    metadata = np.load(split_dir / "metadata.npy").astype(np.float64)

    if location_filter is not None:
        mask = np.zeros(len(location_ids), dtype=bool)
        for loc in location_filter:
            mask |= location_ids == loc
        states = states[mask]
        lengths = lengths[mask]
        recording_ids = recording_ids[mask]
        track_ids = track_ids[mask]
        location_ids = location_ids[mask]
        map_variants = map_variants[mask]
        start_frame_native = start_frame_native[mask]
        metadata = metadata[mask]

    return states, metadata, lengths, location_ids, recording_ids, track_ids, map_variants, start_frame_native


def load_ind_split(
    data_dir: str,
    split: Literal["train", "val", "test"],
    *,
    location_filter: "list[int] | None" = None,
    return_metadata: bool = True,
) -> InDSplitArrays:
    """Load a preprocessed inD split as an :class:`InDSplitArrays`.

    Parameters
    ----------
    data_dir:
        Path to the versioned preprocessed root (e.g. ``data/inD-preprocessed/v1``).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    location_filter:
        If given, only trajectories whose location_id is in this list are returned.
    return_metadata:
        If ``False``, the ``metadata`` field contains a zero-shape placeholder
        (kept for API compatibility when metadata is unused).

    Returns
    -------
    InDSplitArrays
    """
    data_path = Path(data_dir)
    split_dir = data_path / split

    (
        states, metadata, lengths, location_ids,
        recording_ids, track_ids, map_variants, start_frame_native,
    ) = _load_split_arrays(split_dir, location_filter=location_filter)

    if not return_metadata:
        metadata = np.zeros((len(states), 0), dtype=np.float64)

    # Lazy JAX import: keeps module-load JAX-free for numpy-only consumers.
    import jax.numpy as jnp

    return InDSplitArrays(
        states=jnp.asarray(states, dtype=jnp.float64),
        metadata=jnp.asarray(metadata, dtype=jnp.float64),
        lengths=jnp.asarray(lengths, dtype=jnp.int32),
        location_ids=jnp.asarray(location_ids, dtype=jnp.int32),
        recording_ids=jnp.asarray(recording_ids, dtype=jnp.int32),
        track_ids=jnp.asarray(track_ids, dtype=jnp.int32),
        map_variants=jnp.asarray(map_variants),
        start_frame_native=jnp.asarray(start_frame_native, dtype=jnp.int32),
    )


def iter_recordings(
    data_dir: str,
    split: Literal["train", "val", "test"],
) -> Iterator[InDRecordingBundle]:
    """Iterate over all recordings in a split, yielding per-recording bundles.

    Each bundle contains only the trajectories from one recording, with a
    recording-local T_max (smallest padding that fits all tracks in the recording).
    This is the entry point Plan 2's per-recording batching uses.

    Parameters
    ----------
    data_dir:
        Path to the versioned preprocessed root.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.

    Yields
    ------
    InDRecordingBundle
    """
    data_path = Path(data_dir)
    split_dir = data_path / split

    (
        states_all, metadata_all, lengths_all, location_ids_all,
        recording_ids_all, track_ids_all, map_variants_all, start_frame_native_all,
    ) = _load_split_arrays(split_dir, location_filter=None)

    unique_recs = sorted(set(recording_ids_all.tolist()))

    for rec_id in unique_recs:
        mask = recording_ids_all == rec_id
        loc_id = int(location_ids_all[mask][0])
        map_variant = int(map_variants_all[mask][0])

        rec_lengths = lengths_all[mask]
        t_max_rec = int(rec_lengths.max())
        rec_states = states_all[mask, :t_max_rec, :]  # trim to recording-local T_max

        # Yield numpy arrays so callers without JAX installed can consume
        # bundles directly. MaDE's wrapper iter_ind_recordings re-wraps as
        # jnp.asarray on its side.
        yield InDRecordingBundle(
            recording_id=rec_id,
            location_id=loc_id,
            map_variant=map_variant,
            states=np.asarray(rec_states, dtype=np.float64),
            metadata=np.asarray(metadata_all[mask], dtype=np.float64),
            lengths=np.asarray(rec_lengths, dtype=np.int32),
            track_ids=np.asarray(track_ids_all[mask], dtype=np.int32),
            start_frame_native=np.asarray(start_frame_native_all[mask], dtype=np.int32),
        )


def load_manifest(data_dir: str) -> dict:
    """Load and return the manifest.json from a preprocessed inD root."""
    return _load_manifest(Path(data_dir))
