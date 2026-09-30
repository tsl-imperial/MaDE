"""Phase 2 acceptance tests for made.data.ind.transform.

Covers vehicle filtering, downsampling stride, state shape, and the
process_recording_tracks end-to-end pipeline using synthetic row data.
"""

from __future__ import annotations

import math

import numpy as np

from made.data.ind.constants import DOWNSAMPLE_FACTOR, DOWNSAMPLE_PHASE, MIN_FRAMES_5HZ
from made.data.ind.transform import (
    _downsample_track,
    _filter_vehicle_track_ids,
    _rows_to_state,
    _wrap_to_pi,
    process_recording_tracks,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic row data
# ---------------------------------------------------------------------------

def _make_track_rows(
    track_id: int,
    n_frames: int,
    recording_id: int = 0,
    start_frame: int = 0,
) -> list[dict[str, str]]:
    """Minimal synthetic XX_tracks.csv rows for a single track."""
    rows = []
    for i in range(n_frames):
        rows.append(
            {
                "recordingId": str(recording_id),
                "trackId": str(track_id),
                "frame": str(start_frame + i),
                "xCenter": str(float(i)),
                "yCenter": str(float(i) * 0.5),
                "heading": "90.0",
                "xVelocity": "1.0",
                "yVelocity": "0.0",
            }
        )
    return rows


def _make_meta_row(track_id: int, class_: str = "car") -> dict[str, str]:
    return {
        "trackId": str(track_id),
        "width": "1.8",
        "length": "4.5",
        "class": class_,
    }


# ---------------------------------------------------------------------------
# _wrap_to_pi
# ---------------------------------------------------------------------------

def test_wrap_to_pi_in_range() -> None:
    for angle in [0.0, math.pi - 0.001, -math.pi + 0.001, math.pi * 1.5, -math.pi * 2.5]:
        result = _wrap_to_pi(angle)
        assert -math.pi < result <= math.pi, f"wrap_to_pi({angle}) = {result} out of (-π, π]"


# ---------------------------------------------------------------------------
# _filter_vehicle_track_ids
# ---------------------------------------------------------------------------

def test_filter_vehicle_track_ids_excludes_non_vehicles() -> None:
    meta = [
        _make_meta_row(0, "car"),
        _make_meta_row(1, "truck_bus"),
        _make_meta_row(2, "pedestrian"),
        _make_meta_row(3, "bicycle"),
    ]
    ids = _filter_vehicle_track_ids(meta)
    assert ids == {0, 1}


# ---------------------------------------------------------------------------
# _downsample_track
# ---------------------------------------------------------------------------

def test_downsample_stride() -> None:
    """Downsampled rows are exactly DOWNSAMPLE_FACTOR native frames apart."""
    rows = _make_track_rows(track_id=0, n_frames=25)
    down = _downsample_track(rows, phase=DOWNSAMPLE_PHASE)
    assert len(down) == 5  # 25 // 5
    for i, row in enumerate(down):
        expected_frame = i * DOWNSAMPLE_FACTOR
        assert int(row["frame"]) == expected_frame


def test_downsample_non_divisible_length_correct() -> None:
    """13 native frames with DOWNSAMPLE_PHASE=0 yields frames at indices 0, 5, 10."""
    rows = _make_track_rows(track_id=0, n_frames=13)
    down = _downsample_track(rows, phase=0)
    assert len(down) == 3
    assert [int(r["frame"]) for r in down] == [0, 5, 10]


# ---------------------------------------------------------------------------
# _rows_to_state
# ---------------------------------------------------------------------------

def test_rows_to_state_shape() -> None:
    rows = _make_track_rows(track_id=0, n_frames=8)
    states = _rows_to_state(rows)
    assert states.shape == (8, 4)
    assert states.dtype == np.float64


def test_rows_to_state_no_nan() -> None:
    rows = _make_track_rows(track_id=0, n_frames=10)
    states = _rows_to_state(rows)
    assert np.all(np.isfinite(states))


def test_rows_to_state_speed_from_velocity_components() -> None:
    """Speed = hypot(vx, vy), not the longitudinal velocity."""
    row = {
        "xCenter": "0.0", "yCenter": "0.0", "heading": "0.0",
        "xVelocity": "3.0", "yVelocity": "4.0",
    }
    states = _rows_to_state([row])
    assert abs(states[0, 3] - 5.0) < 1e-10  # hypot(3, 4) == 5


def test_rows_to_state_theta_is_wrapped() -> None:
    """Heading of 270 deg (= -90 deg) → theta in (-π, π]."""
    row = {
        "xCenter": "0.0", "yCenter": "0.0", "heading": "270.0",
        "xVelocity": "0.0", "yVelocity": "0.0",
    }
    states = _rows_to_state([row])
    theta = states[0, 2]
    assert -math.pi < theta <= math.pi


# ---------------------------------------------------------------------------
# process_recording_tracks (end-to-end)
# ---------------------------------------------------------------------------

def test_process_recording_tracks_state_shape() -> None:
    """End-to-end: output states have shape (T, 4) per trajectory."""
    n_native = MIN_FRAMES_5HZ * DOWNSAMPLE_FACTOR + DOWNSAMPLE_FACTOR  # enough to survive filter
    track_rows = _make_track_rows(track_id=0, n_frames=n_native)
    meta_rows = [_make_meta_row(0, "car")]
    results = process_recording_tracks(
        track_rows=track_rows,
        track_meta_rows=meta_rows,
        location_id=4,
        recording_id=0,
        min_frames=MIN_FRAMES_5HZ,
    )
    assert len(results) == 1
    assert results[0]["states"].shape[1] == 4
    assert results[0]["states"].dtype == np.float64


def test_process_recording_tracks_metadata_width_5() -> None:
    """Metadata vector has width 5: [length, width, car_oh, truck_oh, location_id]."""
    n_native = MIN_FRAMES_5HZ * DOWNSAMPLE_FACTOR + DOWNSAMPLE_FACTOR
    track_rows = _make_track_rows(track_id=0, n_frames=n_native)
    meta_rows = [_make_meta_row(0, "car")]
    results = process_recording_tracks(
        track_rows=track_rows,
        track_meta_rows=meta_rows,
        location_id=2,
        recording_id=18,
        min_frames=MIN_FRAMES_5HZ,
    )
    assert results[0]["metadata"].shape == (5,)
    assert float(results[0]["metadata"][4]) == 2.0  # location_id preserved


def test_process_recording_tracks_short_track_filtered() -> None:
    """Tracks shorter than min_frames after downsample are excluded."""
    short_native = (MIN_FRAMES_5HZ - 1) * DOWNSAMPLE_FACTOR
    track_rows = _make_track_rows(track_id=0, n_frames=short_native)
    meta_rows = [_make_meta_row(0, "car")]
    results = process_recording_tracks(
        track_rows=track_rows,
        track_meta_rows=meta_rows,
        location_id=4,
        recording_id=0,
        min_frames=MIN_FRAMES_5HZ,
    )
    assert len(results) == 0


def test_process_recording_tracks_non_vehicle_excluded() -> None:
    """Pedestrian / bicycle tracks are excluded."""
    n_native = MIN_FRAMES_5HZ * DOWNSAMPLE_FACTOR + DOWNSAMPLE_FACTOR
    track_rows = _make_track_rows(track_id=0, n_frames=n_native)
    meta_rows = [_make_meta_row(0, "pedestrian")]
    results = process_recording_tracks(
        track_rows=track_rows,
        track_meta_rows=meta_rows,
        location_id=4,
        recording_id=0,
        min_frames=MIN_FRAMES_5HZ,
    )
    assert len(results) == 0
