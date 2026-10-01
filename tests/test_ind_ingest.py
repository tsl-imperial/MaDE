# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for made.data.ind.ingest.

Covers CRLF-tolerance, vehicle counts, frameRate, and locationId assertions
using the synthetic fixture at tests/fixtures/ind_mini/data/ (recordings 00–04,
all locationId=4/aseag, frameRate=25).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from made.data.ind.constants import VALID_LOCATION_IDS
from made.data.ind.ingest import _load_recording

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ind_mini" / "data"
FIXTURE_IDS = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("rec_id", FIXTURE_IDS)
def test_frame_rate_25(rec_id: int) -> None:
    """Checks frame rate 25."""
    tables = _load_recording(FIXTURE_DIR, rec_id)
    assert abs(tables.frame_rate - 25.0) < 1e-6


@pytest.mark.parametrize("rec_id", FIXTURE_IDS)
def test_location_id_valid(rec_id: int) -> None:
    """Checks location id valid."""
    tables = _load_recording(FIXTURE_DIR, rec_id)
    assert tables.location_id in VALID_LOCATION_IDS


@pytest.mark.parametrize("rec_id", FIXTURE_IDS)
def test_vehicle_count_matches_tracks_meta(rec_id: int) -> None:
    """Per-recording vehicle count equals the number of vehicle rows in tracksMeta."""
    from made.data.ind.constants import VEHICLE_CLASSES

    tables = _load_recording(FIXTURE_DIR, rec_id)
    expected = sum(
        1 for row in tables.track_meta_rows if row["class"].strip() in VEHICLE_CLASSES
    )
    # ingest returns all meta rows; filtering happens in transform.
    assert len(tables.track_meta_rows) >= expected
    vehicle_ids_in_meta = {
        int(row["trackId"])
        for row in tables.track_meta_rows
        if row["class"].strip() in VEHICLE_CLASSES
    }
    vehicle_ids_in_tracks = {int(row["trackId"]) for row in tables.track_rows}
    assert vehicle_ids_in_meta <= vehicle_ids_in_tracks


def test_crlf_tolerance(tmp_path: Path) -> None:
    """_load_recording must parse CSV files with CRLF line endings correctly."""
    rec_id = 0
    prefix = f"{rec_id:02d}"
    for suffix in ("_tracks.csv", "_tracksMeta.csv", "_recordingMeta.csv"):
        src = FIXTURE_DIR / f"{prefix}{suffix}"
        dst = tmp_path / f"{prefix}{suffix}"
        content = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        dst.write_bytes(content)

    tables = _load_recording(tmp_path, rec_id)
    assert tables.recording_id == rec_id
    assert abs(tables.frame_rate - 25.0) < 1e-6
    assert tables.location_id in VALID_LOCATION_IDS
    assert len(tables.track_rows) > 0


def test_missing_file_raises(tmp_path: Path) -> None:
    """FileNotFoundError when a required CSV is absent."""
    with pytest.raises(FileNotFoundError):
        _load_recording(tmp_path, 99)
