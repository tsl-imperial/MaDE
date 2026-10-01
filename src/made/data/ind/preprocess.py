# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""inD preprocessing orchestrator.

Entry point: preprocess_ind(raw_dir, output_dir, ...) -> dict

Output layout under output_dir/version/:
  manifest.json
  stats.json
  train/trajectories.npz
  train/metadata.npy
  train/index.csv
  val/  ... (same)
  test/ ... (same)

Note: pyarrow/pandas are unavailable here, so the per-trajectory index is written as a
plain CSV file (index.csv) instead of index.parquet.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from made.data.ind.constants import (
    CLASS_VOCAB,
    CONSTRUCTION_RECORDINGS,
    DEFAULT_SPLIT_SEED,
    DELTA_T,
    DOWNSAMPLE_PHASE,
    FORMAT_VERSION,
    LANELET_MAP_PATHS,
    LANELET_MAP_PATHS_FALLBACK,
    LOCATION_TABLE,
    MIN_FRAMES_5HZ,
    NUM_RECORDINGS,
    TARGET_HZ,
)
from made.data.ind.ingest import _load_recording
from made.data.ind.splits import assign_splits
from made.data.ind.transform import process_recording_tracks


def _git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if unavailable.

    Returns:
        The SHA, or "unknown".
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _resolve_lanelet_paths(raw_dir: Path) -> dict[str, str]:
    """Build relative lanelet map paths from the raw dir.

    Returns paths relative to the raw *dataset root* (parent of the ``data/`` subdirectory).
    When a key has a registered fallback (``LANELET_MAP_PATHS_FALLBACK``) and the primary
    path doesn't exist, the fallback is substituted — handles the vendor typo
    ``constuction`` vs ``construction``.

    Args:
        raw_dir: inD raw data/ directory.
    Returns:
        Mapping from map key to relative path.
    """
    dataset_root = raw_dir.parent  # raw_dir is the data/ subdir
    resolved: dict[str, str] = {}
    for key, rel_path in LANELET_MAP_PATHS.items():
        fallback = LANELET_MAP_PATHS_FALLBACK.get(key)
        if fallback is not None and not (dataset_root / rel_path).exists():
            rel_path = fallback
        resolved[key] = str(rel_path)  # store the chosen relative path
    return resolved


def _pad_and_stack(
    traj_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad a list of [Ti, 4] arrays to [N, T_max, 4] and return lengths.

    Args:
        traj_list: Trajectories of shape [Ti, 4].
    Returns:
        Tuple (states [N, T_max, 4], lengths [N]).
    """
    if not traj_list:
        return np.zeros((0, 0, 4), dtype=np.float64), np.zeros(0, dtype=np.int32)
    lengths = np.asarray([t.shape[0] for t in traj_list], dtype=np.int32)
    t_max = int(lengths.max())
    states = np.zeros((len(traj_list), t_max, 4), dtype=np.float64)
    for i, traj in enumerate(traj_list):
        states[i, : traj.shape[0]] = traj
    return states, lengths


def _write_split_artefacts(
    split_dir: Path,
    records: list[dict],
) -> int:
    """Write trajectories.npz, metadata.npy, index.csv for one split.

    Returns the number of trajectories written.

    Args:
        split_dir: Output directory of the split.
        records: Trajectory records.
    Returns:
        Number of trajectories written.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    if n == 0:
        # Write empty artefacts so loaders don't need special-casing.
        np.savez(
            split_dir / "trajectories.npz",
            states=np.zeros((0, 0, 4), dtype=np.float64),
            lengths=np.zeros(0, dtype=np.int32),
            recording_ids=np.zeros(0, dtype=np.int32),
            track_ids=np.zeros(0, dtype=np.int32),
            location_ids=np.zeros(0, dtype=np.int32),
            map_variants=np.zeros(0, dtype=np.uint8),
            start_frame_native=np.zeros(0, dtype=np.int32),
        )
        np.save(split_dir / "metadata.npy", np.zeros((0, 5), dtype=np.float64))
        _write_index_csv(split_dir / "index.csv", records=[])
        return 0

    traj_list = [r["states"] for r in records]
    states, lengths = _pad_and_stack(traj_list)

    recording_ids = np.asarray([r["recording_id"] for r in records], dtype=np.int32)
    track_ids = np.asarray([r["track_id"] for r in records], dtype=np.int32)
    location_ids = np.asarray([r["location_id"] for r in records], dtype=np.int32)
    map_variants = np.asarray(
        [1 if r["recording_id"] in CONSTRUCTION_RECORDINGS else 0 for r in records],
        dtype=np.uint8,
    )
    start_frame_native = np.asarray([r["start_frame_native"] for r in records], dtype=np.int32)
    metadata = np.stack([r["metadata"] for r in records], axis=0).astype(np.float64)

    np.savez(
        split_dir / "trajectories.npz",
        states=states,
        lengths=lengths,
        recording_ids=recording_ids,
        track_ids=track_ids,
        location_ids=location_ids,
        map_variants=map_variants,
        start_frame_native=start_frame_native,
    )
    np.save(split_dir / "metadata.npy", metadata)
    _write_index_csv(split_dir / "index.csv", records=records)
    return n


def _write_index_csv(path: Path, records: list[dict]) -> None:
    """Write a lightweight CSV index for per-trajectory filtering.

    Args:
        path: Output CSV path.
        records: Trajectory records.
    """
    fieldnames = [
        "traj_idx",
        "recording_id",
        "track_id",
        "location_id",
        "map_variant",
        "length",
        "start_frame_native",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow(
                {
                    "traj_idx": i,
                    "recording_id": r["recording_id"],
                    "track_id": r["track_id"],
                    "location_id": r["location_id"],
                    "map_variant": 1 if r["recording_id"] in CONSTRUCTION_RECORDINGS else 0,
                    "length": r["states"].shape[0],
                    "start_frame_native": r["start_frame_native"],
                }
            )


def preprocess_ind(
    raw_dir: str,
    output_dir: str,
    *,
    version: str = "v1",
    split_seed: int = DEFAULT_SPLIT_SEED,
    min_frames: int = MIN_FRAMES_5HZ,
    force: bool = False,
    _location_table: dict | None = None,
    _split_assignment: dict[int, str] | None = None,
    _recording_ids: list[int] | None = None,
    _construction_recordings: "frozenset[int] | None" = None,
) -> dict:
    """Preprocess all inD recordings into versioned NPZ artefacts.

    Args:
        raw_dir: Path to the inD ``data/`` directory containing ``XX_tracks.csv`` etc.
        output_dir: Root output directory. Artefacts written under ``output_dir/version/``.
        version: Version tag (e.g. ``"v1"``). A new tag makes a fresh tree; an existing tag requires
            ``force=True``.
        split_seed: Seed passed to :func:`assign_splits`.
        min_frames: Minimum trajectory length (5 Hz frames) after downsampling.
        force: Overwrite an existing versioned output directory.
        _location_table, _split_assignment, _recording_ids, _construction_recordings: Internal test
            overrides; module-level defaults used when None.

    Returns:
        dict: Summary / manifest dict (same content written to ``manifest.json``).
    """
    raw_path = Path(raw_dir)
    out_root = Path(output_dir) / version

    if out_root.exists() and not force:
        raise FileExistsError(
            f"{out_root} already exists.  Pass force=True to overwrite, or use a new version tag."
        )

    out_root.mkdir(parents=True, exist_ok=True)

    location_table = _location_table if _location_table is not None else LOCATION_TABLE
    recording_ids = (
        list(_recording_ids) if _recording_ids is not None
        else list(range(NUM_RECORDINGS))
    )

    # Ingest + transform all recordings.
    vehicle_counts_per_recording: dict[str, int] = {}

    if _split_assignment is not None:
        split_assignment = _split_assignment
    else:
        split_assignment = assign_splits(location_table=location_table, seed=split_seed)

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rec_id in recording_ids:
        tables = _load_recording(raw_path, rec_id)
        trajs = process_recording_tracks(
            track_rows=tables.track_rows,
            track_meta_rows=tables.track_meta_rows,
            location_id=tables.location_id,
            recording_id=rec_id,
            min_frames=min_frames,
            phase=DOWNSAMPLE_PHASE,
        )
        split_records[split_assignment[rec_id]].extend(trajs)
        vehicle_counts_per_recording[f"{rec_id:02d}"] = len(trajs)

    # Write split artefacts.
    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        n_written = _write_split_artefacts(
            out_root / split_name, split_records[split_name]
        )
        split_counts[split_name] = n_written

    # Compute train-only state statistics.
    stats = _compute_stats(split_records["train"])
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Build and write manifest.
    lanelet_paths = _resolve_lanelet_paths(raw_path)
    location_table_serialisable = {
        str(k): {
            "name": v["name"],
            "recordings": v["recordings"],
        }
        for k, v in location_table.items()
    }

    splits_manifest = {
        "train": sorted(rec for rec, s in split_assignment.items() if s == "train"),
        "val": sorted(rec for rec, s in split_assignment.items() if s == "val"),
        "test": sorted(rec for rec, s in split_assignment.items() if s == "test"),
    }

    manifest = {
        "format_version": FORMAT_VERSION,
        "frame_rate_hz": TARGET_HZ,
        "delta_t_seconds": DELTA_T,
        "downsample_phase": DOWNSAMPLE_PHASE,
        "class_vocab": CLASS_VOCAB,
        "location_table": location_table_serialisable,
        "construction_recordings": sorted(CONSTRUCTION_RECORDINGS),
        "lanelet_map_paths": lanelet_paths,
        "splits": splits_manifest,
        "split_seed": split_seed,
        "min_trajectory_length_5hz": min_frames,
        "vehicle_counts": vehicle_counts_per_recording,
        "trajectory_counts_per_split": split_counts,
        "total_trajectories": sum(split_counts.values()),
        "preprocess_git_sha": _git_sha(),
        "index_format": "csv",  # pyarrow unavailable; using CSV index instead of parquet
    }

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print one-screen summary
    total = sum(split_counts.values())
    print(
        f"inD preprocessing complete -> {out_root}\n"
        f"  train: {split_counts['train']} trajectories\n"
        f"  val:   {split_counts['val']} trajectories\n"
        f"  test:  {split_counts['test']} trajectories\n"
        f"  total: {total} trajectories\n"
        f"  git sha: {manifest['preprocess_git_sha']}"
    )

    return manifest


def _compute_stats(records: list[dict]) -> dict:
    """Compute per-state-dim mean/std from train records, and per-class counts.

    Args:
        records: Training trajectory records.
    Returns:
        Dict with state_mean, state_std and class_counts.
    """
    if not records:
        return {"state_mean": [0.0, 0.0, 0.0, 0.0], "state_std": [1.0, 1.0, 1.0, 1.0]}

    all_states = np.concatenate([r["states"] for r in records], axis=0)  # [N_steps, 4]
    mean = all_states.mean(axis=0).tolist()
    std = all_states.std(axis=0).tolist()

    class_counts: dict[str, int] = {c: 0 for c in CLASS_VOCAB}
    for r in records:
        meta = r["metadata"]
        # one-hot slots are at indices 2 and 3
        for i, cls in enumerate(CLASS_VOCAB):
            if meta[2 + i] > 0.5:
                class_counts[cls] += 1

    return {
        "state_mean": mean,
        "state_std": std,
        "class_counts": class_counts,
    }


# CLI entry point: python -m made.data.ind.preprocess
def _cli() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess inD dataset")
    parser.add_argument("--raw-dir", required=True, help="Path to inD data/ directory")
    parser.add_argument(
        "--output-dir", required=True, help="Output root directory"
    )
    parser.add_argument("--version", default="v1", help="Version tag (default: v1)")
    parser.add_argument(
        "--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Split seed"
    )
    parser.add_argument(
        "--min-frames", type=int, default=MIN_FRAMES_5HZ,
        help="Minimum frames after downsample"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()
    preprocess_ind(
        args.raw_dir,
        args.output_dir,
        version=args.version,
        split_seed=args.split_seed,
        min_frames=args.min_frames,
        force=args.force,
    )


if __name__ == "__main__":
    _cli()
