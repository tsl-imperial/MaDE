"""Constants for the inD preprocessing pipeline.

Data-exploration notes (Phase 0 acceptance):
- All 33 recordings (00–32) are at 25 Hz (frameRate=25.0 in recordingMeta.csv). Confirmed.
- Four locationId values map exactly as: 1=bendplatz, 2=frankenburg, 3=heckstrasse, 4=aseag.
  Confirmed by reading recordingMeta.csv for recs 0 (loc=4/aseag), 7 (loc=1/bendplatz),
  11 (loc=1/bendplatz/construction), 18 (loc=2/frankenburg), 30 (loc=3/heckstrasse).
- Bendplatz construction recordings are exactly {11, 12, 13, 14, 15, 16, 17}, matching
  the maps/README.txt and the parent spec.
- Four classes on inD: {pedestrian, bicycle, car, truck_bus}. Filtering to {car, truck_bus}
  retains exactly 8,233 raw tracks (before min-length filter). Per-recording counts:
  [324, 355, 341, 274, 177, 287, 340, 150, 284, 168, 190, 214, 268, 231, 226, 230, 280,
   262, 181, 156, 210, 218, 218, 205, 147, 227, 232, 203, 211, 228, 381, 377, 438]
  Plan §3 lists the same vector; the plan cited ~7,943 post-min-length-filter.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------
NATIVE_HZ: int = 25
TARGET_HZ: int = 5
DOWNSAMPLE_FACTOR: int = NATIVE_HZ // TARGET_HZ  # = 5
DELTA_T: float = 1.0 / TARGET_HZ  # = 0.2 s
DOWNSAMPLE_PHASE: int = 0  # keep frames where (relative_frame_index % DOWNSAMPLE_FACTOR == PHASE)

# ---------------------------------------------------------------------------
# Minimum trajectory length after downsample (Decision P6)
# ---------------------------------------------------------------------------
MIN_FRAMES_5HZ: int = 8

# ---------------------------------------------------------------------------
# Vehicle classes (Decision L1)
# ---------------------------------------------------------------------------
CLASS_VOCAB: list[str] = ["car", "truck_bus"]
VEHICLE_CLASSES: frozenset[str] = frozenset(CLASS_VOCAB)

# ---------------------------------------------------------------------------
# Location table (Decision P3 — vendor canonical IDs)
# 1=bendplatz, 2=frankenburg, 3=heckstrasse, 4=aseag
# ---------------------------------------------------------------------------
LOCATION_TABLE: dict[int, dict] = {
    1: {
        "name": "bendplatz",
        "recordings": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    },
    2: {
        "name": "frankenburg",
        "recordings": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    },
    3: {
        "name": "heckstrasse",
        "recordings": [30, 31, 32],
    },
    4: {
        "name": "aseag",
        "recordings": [0, 1, 2, 3, 4, 5, 6],
    },
}

# Recordings that use the bendplatz *construction* map variant (Decision P4)
CONSTRUCTION_RECORDINGS: frozenset[int] = frozenset([11, 12, 13, 14, 15, 16, 17])

# Reverse lookup: recording_id -> location_id
RECORDING_TO_LOCATION: dict[int, int] = {
    rec: loc_id
    for loc_id, info in LOCATION_TABLE.items()
    for rec in info["recordings"]
}

# Total number of recordings
NUM_RECORDINGS: int = 33  # 00–32

# All valid location IDs
VALID_LOCATION_IDS: frozenset[int] = frozenset(LOCATION_TABLE.keys())

# ---------------------------------------------------------------------------
# Lanelet2 map paths relative to the raw inD dataset root (Decision R5 — typo fallback)
# ---------------------------------------------------------------------------
# Key "1_construction" is for bendplatz construction variant
LANELET_MAP_PATHS: dict[str, str] = {
    "1": "maps/lanelets/01_bendplatz/location1.osm",
    # vendor typo: "constuction" (missing 'r')
    "1_construction": "maps/lanelets/01_bendplatz_constuction/location1_construction.osm",
    "2": "maps/lanelets/02_frankenburg/location2.osm",
    "3": "maps/lanelets/03_heckstrasse/location3.osm",
    "4": "maps/lanelets/04_aseag/location4.osm",
}

# Corrected spelling fallback (R5: if vendor fixes the typo in a future release)
LANELET_MAP_PATHS_FALLBACK: dict[str, str] = {
    "1_construction": "maps/lanelets/01_bendplatz_construction/location1_construction.osm",
}

# ---------------------------------------------------------------------------
# Default split seed (Decision P9/P10)
# ---------------------------------------------------------------------------
DEFAULT_SPLIT_SEED: int = 20260505

# ---------------------------------------------------------------------------
# Output format version
# ---------------------------------------------------------------------------
FORMAT_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Expected per-recording vehicle counts (before min-length filter), for
# soft manifest sanity checks
# ---------------------------------------------------------------------------
EXPECTED_VEHICLE_COUNTS: list[int] = [
    324, 355, 341, 274, 177, 287, 340,   # recs 0–6  (aseag)
    150, 284, 168, 190, 214, 268, 231,   # recs 7–13 (bendplatz)
    226, 230, 280, 262,                   # recs 14–17 (bendplatz)
    181, 156, 210, 218, 218, 205, 147,   # recs 18–24 (frankenburg)
    227, 232, 203, 211, 228,              # recs 25–29 (frankenburg)
    381, 377, 438,                        # recs 30–32 (heckstrasse)
]
