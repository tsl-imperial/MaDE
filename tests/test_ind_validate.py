# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for made.data.ind.validate.

Builds minimal preprocessed bundles in tmp_path and verifies that
validate_preprocessed() passes on well-formed bundles and fails with
descriptive errors on corrupt bundles.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from made.data.ind.validate import validate_preprocessed

_VALID_MANIFEST = {
    "format_version": "1.0",
    "frame_rate_hz": 5,
    "delta_t_seconds": 0.2,
    "downsample_phase": 0,
    "class_vocab": ["car", "truck_bus"],
    "location_table": {
        "4": {"name": "aseag", "recordings": [0, 1, 2, 3, 4]},
    },
    "construction_recordings": [],
    "splits": {
        "train": [0, 1, 2],
        "val": [3],
        "test": [4],
    },
}

_SPLIT_SIZES = {"train": 3, "val": 1, "test": 1}


def _make_trajectory(t: int = 10) -> np.ndarray:
    """Return a [t, 4] float64 state array with a plausible class one-hot.

    Args:
        t: Number of time steps.

    Returns:
        State array.
    """
    return np.random.default_rng(0).standard_normal((t, 4)).astype(np.float64)


def _write_split(split_dir: Path, n: int) -> None:
    """Write a synthetic split directory with ``n`` trajectories.

    Args:
        split_dir: Directory to write into.
        n: Number of trajectories.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    t_len = 10
    states = np.zeros((n, t_len, 4), dtype=np.float64) if n > 0 else np.zeros((0, 0, 4), dtype=np.float64)
    lengths = np.full(n, t_len, dtype=np.int32) if n > 0 else np.zeros(0, dtype=np.int32)
    recording_ids = np.arange(n, dtype=np.int32)
    track_ids = np.zeros(n, dtype=np.int32)
    location_ids = np.full(n, 4, dtype=np.int32)
    map_variants = np.zeros(n, dtype=np.uint8)
    start_frame_native = np.zeros(n, dtype=np.int32)

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

    # metadata: [length, width, car_onehot, truck_onehot, location_id] — valid class one-hot
    metadata = np.tile([4.5, 1.8, 1.0, 0.0, 4.0], (max(n, 1), 1)).astype(np.float64)
    if n == 0:
        metadata = np.zeros((0, 5), dtype=np.float64)
    np.save(split_dir / "metadata.npy", metadata)

    fieldnames = [
        "traj_idx", "recording_id", "track_id", "location_id",
        "map_variant", "length", "start_frame_native",
    ]
    with (split_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "traj_idx": i, "recording_id": i, "track_id": 0,
                "location_id": 4, "map_variant": 0, "length": t_len,
                "start_frame_native": 0,
            })


def _make_valid_bundle(root: Path) -> None:
    """Write a complete valid inD bundle.

    Args:
        root: Bundle root directory.
    """
    (root / "manifest.json").write_text(json.dumps(_VALID_MANIFEST), encoding="utf-8")
    (root / "stats.json").write_text("{}", encoding="utf-8")
    for split, n in _SPLIT_SIZES.items():
        _write_split(root / split, n)


def test_valid_bundle_passes(tmp_path: Path) -> None:
    """Checks valid bundle passes."""
    _make_valid_bundle(tmp_path)
    report = validate_preprocessed(str(tmp_path))
    assert report.passed, f"Expected pass; errors: {report.errors}"
    assert report.errors == []

def test_missing_manifest_key_fails(tmp_path: Path) -> None:
    """Checks missing manifest key fails."""
    _make_valid_bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["delta_t_seconds"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_preprocessed(str(tmp_path))
    assert not report.passed
    assert any("delta_t_seconds" in e for e in report.errors)


def test_index_csv_row_count_mismatch_fails(tmp_path: Path) -> None:
    """index.csv with extra rows vs trajectories.npz must raise an error."""
    _make_valid_bundle(tmp_path)
    index_path = tmp_path / "train" / "index.csv"
    content = index_path.read_text(encoding="utf-8")
    # Append a spurious extra row (copy of row 0 with traj_idx=999)
    extra_row = content.strip().split("\n")[1].replace("0,0,", "999,0,")
    index_path.write_text(content + extra_row + "\n", encoding="utf-8")

    report = validate_preprocessed(str(tmp_path))
    assert not report.passed
    assert any("index.csv" in e for e in report.errors)


def test_split_overlap_fails(tmp_path: Path) -> None:
    """Manifest with overlapping split recording lists must produce an error."""
    _make_valid_bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # Put rec 0 in both train and val
    manifest["splits"]["val"] = [0, 3]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_preprocessed(str(tmp_path))
    assert not report.passed
    assert any("overlap" in e.lower() for e in report.errors)


def test_missing_manifest_file_fails(tmp_path: Path) -> None:
    """Checks missing manifest file fails."""
    report = validate_preprocessed(str(tmp_path))
    assert not report.passed
    assert any("manifest.json" in e for e in report.errors)
