"""Per-track filtering, grouping, downsampling, and state extraction.

Downsample rule (Decision P5):
  Keep frames where (track_frame_index % DOWNSAMPLE_FACTOR == DOWNSAMPLE_PHASE),
  where track_frame_index = 0, 1, 2, ... counts from the track's first native frame.
  This is a per-track relative phase, not a global frame index.

State schema (Decision P7, P8):
  (x, y, theta_rad, v) where:
    x, y   -- xCenter, yCenter from the inD CSV (metres)
    theta  -- deg2rad(heading) wrapped to (-pi, pi]
    v      -- hypot(xVelocity, yVelocity)  [NOT lonVelocity -- body-frame gotcha]

Metadata schema (plan §3):
  [length, width, class_one_hot[0], class_one_hot[1], location_id_float]
  class_one_hot over ("car", "truck_bus") -- exactly two slots.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from made.data.ind.constants import (
    CLASS_VOCAB,
    DOWNSAMPLE_FACTOR,
    DOWNSAMPLE_PHASE,
    MIN_FRAMES_5HZ,
    VEHICLE_CLASSES,
)


def _filter_vehicle_track_ids(meta_rows: list[dict[str, str]]) -> set[int]:
    """Return the set of track IDs whose class is in VEHICLE_CLASSES."""
    return {
        int(row["trackId"])
        for row in meta_rows
        if row["class"].strip() in VEHICLE_CLASSES
    }


def _group_track_frames(
    track_rows: list[dict[str, str]],
    vehicle_ids: set[int],
) -> dict[int, list[dict[str, str]]]:
    """Group track rows by track ID, keeping only vehicle tracks, sorted by frame."""
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in track_rows:
        tid = int(row["trackId"])
        if tid in vehicle_ids:
            grouped[tid].append(row)
    # Sort each group by native frame number
    for tid in grouped:
        grouped[tid].sort(key=lambda r: int(r["frame"]))
    return dict(grouped)


def _downsample_track(
    rows: list[dict[str, str]],
    phase: int = DOWNSAMPLE_PHASE,
) -> list[dict[str, str]]:
    """Keep every DOWNSAMPLE_FACTOR-th frame starting at relative index == phase.

    The relative frame index starts at 0 for the track's first native frame,
    regardless of the global frame counter.  This gives Δt = 0.2 s per step.
    """
    return [row for i, row in enumerate(rows) if i % DOWNSAMPLE_FACTOR == phase]


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to (-pi, pi]."""
    wrapped = math.fmod(angle_rad, 2 * math.pi)
    # fmod preserves sign; shift into (-pi, pi]
    if wrapped > math.pi:
        wrapped -= 2 * math.pi
    elif wrapped <= -math.pi:
        wrapped += 2 * math.pi
    return wrapped


def _rows_to_state(rows: list[dict[str, str]]) -> np.ndarray:
    """Convert a list of track rows to a float64 state array of shape [T, 4].

    State columns: (x, y, theta_rad, v).
    """
    T = len(rows)
    states = np.empty((T, 4), dtype=np.float64)
    for i, row in enumerate(rows):
        x = float(row["xCenter"])
        y = float(row["yCenter"])
        heading_deg = float(row["heading"])
        theta = _wrap_to_pi(math.radians(heading_deg))
        vx = float(row["xVelocity"])
        vy = float(row["yVelocity"])
        v = math.hypot(vx, vy)  # ground-frame speed magnitude (Decision P8)
        states[i, 0] = x
        states[i, 1] = y
        states[i, 2] = theta
        states[i, 3] = v
    return states


def _class_one_hot(class_label: str) -> list[float]:
    """One-hot over CLASS_VOCAB = ["car", "truck_bus"]."""
    return [1.0 if class_label == item else 0.0 for item in CLASS_VOCAB]


def _track_metadata(
    meta_row: dict[str, str],
    location_id: int,
) -> np.ndarray:
    """Build the metadata vector for a single track.

    Schema: [length, width, one_hot_car, one_hot_truck_bus, location_id_float]
    Shape: [5] float64.
    """
    length = float(meta_row["length"])
    width = float(meta_row["width"])
    class_label = meta_row["class"].strip()
    one_hot = _class_one_hot(class_label)
    return np.asarray([length, width, *one_hot, float(location_id)], dtype=np.float64)


def process_recording_tracks(
    track_rows: list[dict[str, str]],
    track_meta_rows: list[dict[str, str]],
    location_id: int,
    recording_id: int,
    min_frames: int = MIN_FRAMES_5HZ,
    phase: int = DOWNSAMPLE_PHASE,
) -> list[dict]:
    """Full per-recording transform pipeline.

    Returns a list of dicts (one per accepted trajectory) with keys:
        states          np.ndarray [T, 4]   float64
        metadata        np.ndarray [5]      float64
        track_id        int
        location_id     int
        recording_id    int
        start_frame_native  int   (first native frame retained after downsample)
    """
    meta_by_id: dict[int, dict[str, str]] = {
        int(row["trackId"]): row for row in track_meta_rows
    }
    vehicle_ids = _filter_vehicle_track_ids(track_meta_rows)
    grouped = _group_track_frames(track_rows, vehicle_ids)

    results = []
    for tid in sorted(grouped.keys()):  # deterministic order
        rows_all = grouped[tid]
        rows_down = _downsample_track(rows_all, phase=phase)
        if len(rows_down) < min_frames:
            continue
        states = _rows_to_state(rows_down)
        meta_row = meta_by_id[tid]
        metadata = _track_metadata(meta_row, location_id)
        start_frame_native = int(rows_down[0]["frame"])
        results.append(
            {
                "states": states,
                "metadata": metadata,
                "track_id": tid,
                "location_id": location_id,
                "recording_id": recording_id,
                "start_frame_native": start_frame_native,
            }
        )
    return results
