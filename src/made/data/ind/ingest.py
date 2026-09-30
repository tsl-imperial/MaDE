"""Per-recording CSV ingestion for the inD dataset.

CRLF tolerance: Python's csv.DictReader with newline='' handles both LF and CRLF
transparently (per Python docs). We enforce this by always opening with newline=''.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from made.data.ind.constants import VALID_LOCATION_IDS


@dataclass(frozen=True)
class RecordingTables:
    """Parsed tables for a single inD recording."""

    recording_id: int
    location_id: int
    frame_rate: float
    track_rows: list[dict[str, str]]           # from XX_tracks.csv
    track_meta_rows: list[dict[str, str]]       # from XX_tracksMeta.csv
    recording_meta_row: dict[str, str]          # single row from XX_recordingMeta.csv


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load all rows from a CSV file.

    Opens with newline='' so csv.DictReader strips CRLF line endings correctly
    regardless of the file's actual line-ending convention (inD ships CRLF).
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


_REQUIRED_TRACK_COLUMNS: frozenset[str] = frozenset(
    [
        "recordingId",
        "trackId",
        "frame",
        "xCenter",
        "yCenter",
        "heading",
        "xVelocity",
        "yVelocity",
    ]
)

_REQUIRED_META_COLUMNS: frozenset[str] = frozenset(
    [
        "trackId",
        "width",
        "length",
        "class",
    ]
)

_REQUIRED_REC_COLUMNS: frozenset[str] = frozenset(
    [
        "recordingId",
        "locationId",
        "frameRate",
    ]
)


def _check_columns(rows: list[dict[str, str]], required: frozenset[str], label: str) -> None:
    if not rows:
        return
    present = frozenset(rows[0].keys())
    missing = required - present
    if missing:
        raise ValueError(f"{label}: missing required columns {sorted(missing)}")


def _load_recording(raw_dir: str | Path, recording_id: int) -> RecordingTables:
    """Parse all three CSV files for a single inD recording.

    Parameters
    ----------
    raw_dir:
        Path to the inD `data/` directory (the one containing the CSV files).
    recording_id:
        Integer recording ID, 0–32.

    Returns
    -------
    RecordingTables
        Parsed tables; does not filter or transform data.

    Raises
    ------
    FileNotFoundError
        If any of the three required CSV files is absent.
    ValueError
        If frameRate != 25, locationId is not in {1,2,3,4}, or a required column
        is missing.
    """
    raw_dir = Path(raw_dir)
    prefix = f"{recording_id:02d}"

    tracks_path = raw_dir / f"{prefix}_tracks.csv"
    meta_path = raw_dir / f"{prefix}_tracksMeta.csv"
    rec_path = raw_dir / f"{prefix}_recordingMeta.csv"

    for p in (tracks_path, meta_path, rec_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    track_rows = _load_csv_rows(tracks_path)
    track_meta_rows = _load_csv_rows(meta_path)
    recording_meta_rows = _load_csv_rows(rec_path)

    _check_columns(track_rows, _REQUIRED_TRACK_COLUMNS, f"recording {recording_id} tracks")
    _check_columns(track_meta_rows, _REQUIRED_META_COLUMNS, f"recording {recording_id} tracksMeta")
    _check_columns(
        recording_meta_rows, _REQUIRED_REC_COLUMNS, f"recording {recording_id} recordingMeta"
    )

    if not recording_meta_rows:
        raise ValueError(f"recording {recording_id}: recordingMeta.csv is empty")

    rec_row = recording_meta_rows[0]

    frame_rate = float(rec_row["frameRate"])
    if abs(frame_rate - 25.0) > 1e-6:
        raise ValueError(
            f"recording {recording_id}: expected frameRate=25, got {frame_rate}"
        )

    location_id = int(rec_row["locationId"])
    if location_id not in VALID_LOCATION_IDS:
        raise ValueError(
            f"recording {recording_id}: locationId={location_id} not in {sorted(VALID_LOCATION_IDS)}"
        )

    return RecordingTables(
        recording_id=recording_id,
        location_id=location_id,
        frame_rate=frame_rate,
        track_rows=track_rows,
        track_meta_rows=track_meta_rows,
        recording_meta_row=rec_row,
    )
