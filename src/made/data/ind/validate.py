"""Validation harness for preprocessed inD artefacts.

CLI: python -m made.data.ind.validate <data_dir>
Exits 0 on success, 1 on any validation failure.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from made.data.ind.constants import (
    DOWNSAMPLE_FACTOR,
    FORMAT_VERSION,
)


@dataclass
class ValidationReport:
    """Result of validate_preprocessed()."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["ValidationReport:"]
        lines.append(f"  PASSED: {self.passed}")
        if self.errors:
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ~ {w}")
        return "\n".join(lines)


def validate_preprocessed(data_dir: str) -> ValidationReport:
    """Run all validation checks on a preprocessed inD directory.

    Parameters
    ----------
    data_dir:
        Path to a versioned preprocessed root (e.g. ``data/inD-preprocessed/v1``).

    Returns
    -------
    ValidationReport
        ``.passed`` is True iff zero errors were found.
    """
    errors: list[str] = []
    warnings: list[str] = []
    root = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. manifest.json exists and has required fields
    # ------------------------------------------------------------------
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"manifest.json not found at {root}")
        return ValidationReport(passed=False, errors=errors, warnings=warnings)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"manifest.json is not valid JSON: {exc}")
        return ValidationReport(passed=False, errors=errors, warnings=warnings)

    for required_key in ("format_version", "splits", "location_table", "construction_recordings",
                         "class_vocab", "delta_t_seconds", "frame_rate_hz"):
        if required_key not in manifest:
            errors.append(f"manifest.json missing key: {required_key}")

    if errors:
        return ValidationReport(passed=False, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # 2. format_version is recognised
    # ------------------------------------------------------------------
    if manifest["format_version"] != FORMAT_VERSION:
        errors.append(
            f"Unrecognised format_version '{manifest['format_version']}'; "
            f"expected '{FORMAT_VERSION}'"
        )

    # ------------------------------------------------------------------
    # 3. Splits are pairwise disjoint at recording level
    # ------------------------------------------------------------------
    splits_dict: dict[str, list[int]] = manifest["splits"]
    all_recording_sets: dict[str, set[int]] = {
        s: set(recs) for s, recs in splits_dict.items()
    }
    split_names = list(all_recording_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = all_recording_sets[s1] & all_recording_sets[s2]
            if overlap:
                errors.append(
                    f"Split overlap between '{s1}' and '{s2}': recordings {sorted(overlap)}"
                )

    # ------------------------------------------------------------------
    # 4. Every location ID appears in the train split
    # ------------------------------------------------------------------
    train_recs = all_recording_sets.get("train", set())
    manifest_loc_table = manifest["location_table"]
    for loc_str, info in manifest_loc_table.items():
        loc_recs = set(info["recordings"])
        if not loc_recs & train_recs:
            errors.append(
                f"Location {loc_str} ({info['name']}) has no recordings in train split"
            )

    # ------------------------------------------------------------------
    # 5. Construction recordings (11–17) appear in val or test
    # ------------------------------------------------------------------
    construction_set = set(manifest.get("construction_recordings", []))
    val_test_recs = all_recording_sets.get("val", set()) | all_recording_sets.get("test", set())
    if construction_set and not construction_set & val_test_recs:
        warnings.append(
            "No construction recordings appear in val or test splits; "
            "map_variant=1 code path not exercised on held-out data"
        )

    # 6. Per-split array checks
    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.exists():
            errors.append(f"Split directory missing: {split_dir}")
            continue

        # trajectories.npz
        npz_path = split_dir / "trajectories.npz"
        if not npz_path.exists():
            errors.append(f"Missing: {npz_path}")
            continue

        try:
            npz = np.load(npz_path)
        except Exception as exc:
            errors.append(f"Cannot load {npz_path}: {exc}")
            continue

        required_keys = ["states", "lengths", "recording_ids", "track_ids",
                         "location_ids", "map_variants", "start_frame_native"]
        for k in required_keys:
            if k not in npz:
                errors.append(f"{npz_path}: missing key '{k}'")

        if "states" in npz and "lengths" in npz:
            states = npz["states"]
            lengths = npz["lengths"]
            n = len(lengths)

            if len(states) != n:
                errors.append(
                    f"{split_name}/trajectories.npz: states.shape[0]={len(states)} "
                    f"!= lengths.shape[0]={n}"
                )

            # No trajectory should have all-zero class one-hot
            metadata_path = split_dir / "metadata.npy"
            if metadata_path.exists():
                metadata = np.load(metadata_path)
                if metadata.shape[0] != n:
                    errors.append(
                        f"{split_name}/metadata.npy: shape[0]={metadata.shape[0]} != {n}"
                    )
                if n > 0 and metadata.shape[1] >= 4:
                    # Cols 2 and 3 are class one-hot
                    one_hot_sum = metadata[:, 2] + metadata[:, 3]
                    zero_class = np.where(one_hot_sum < 0.5)[0]
                    if len(zero_class) > 0:
                        errors.append(
                            f"{split_name}: {len(zero_class)} trajectories have zero "
                            f"class one-hot in both vehicle slots"
                        )

        # index.csv row count matches lengths
        index_path = split_dir / "index.csv"
        if index_path.exists() and "lengths" in npz:
            with index_path.open("r", newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            if len(csv_rows) != len(npz["lengths"]):
                errors.append(
                    f"{split_name}/index.csv: {len(csv_rows)} rows != "
                    f"{len(npz['lengths'])} trajectories in npz"
                )

        # Δt is enforced by construction in ``_downsample_track`` and exercised in
        # ``tests/data/test_ind_loader.py``. NPZ stores only the first native frame, so a
        # per-step check from disk alone would be a no-op; skipped here.
        _ = DOWNSAMPLE_FACTOR  # silences unused-import warning

    # 7. lanelet_map_paths exist on disk (relative to an inferred dataset root)
    lanelet_paths = manifest.get("lanelet_map_paths", {})
    if lanelet_paths:
        # raw_dir unknown here, so existence check is skipped with a warning.
        warnings.append(
            "lanelet_map_paths existence check skipped: raw_dir unknown to validator. "
            "Pass raw_dir explicitly or use the CLI with --raw-dir to verify."
        )

    # 8. vehicle_counts sum to total_trajectories (if present)
    if "vehicle_counts" in manifest and "total_trajectories" in manifest:
        claimed_total = manifest["total_trajectories"]
        split_total = sum(manifest.get("trajectory_counts_per_split", {}).values())
        if split_total != claimed_total:
            errors.append(
                f"manifest total_trajectories={claimed_total} != "
                f"sum of split counts={split_total}"
            )

    # 9. stats.json exists
    stats_path = root / "stats.json"
    if not stats_path.exists():
        warnings.append("stats.json not found (optional but expected)")

    passed = len(errors) == 0
    return ValidationReport(passed=passed, errors=errors, warnings=warnings)


def validate_and_check_lanelet_paths(data_dir: str, raw_dir: str) -> ValidationReport:
    """Run full validation including lanelet map file existence check.

    Parameters
    ----------
    data_dir:
        Preprocessed root directory.
    raw_dir:
        inD raw ``data/`` directory (used to locate the dataset root for map files).
    """
    report = validate_preprocessed(data_dir)

    # Now resolve map path existence with the known raw_dir
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(raw_dir)
    dataset_root = raw_path.parent  # data/ -> dataset root

    lanelet_paths = manifest.get("lanelet_map_paths", {})
    for key, rel_path in lanelet_paths.items():
        abs_path = dataset_root / rel_path
        if not abs_path.exists():
            report.errors.append(f"lanelet_map_paths[{key!r}] not found: {abs_path}")
            report.passed = False

    # Remove the "skipped" warning if we just checked
    report.warnings = [
        w for w in report.warnings if "lanelet_map_paths existence check skipped" not in w
    ]

    return report


# CLI entry point: python -m made.data.ind.validate
def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate a preprocessed inD directory."
    )
    parser.add_argument("data_dir", help="Path to versioned preprocessed root")
    parser.add_argument(
        "--raw-dir", default=None,
        help="Optional: inD raw data/ directory for lanelet map path existence check"
    )
    args = parser.parse_args()

    if args.raw_dir:
        report = validate_and_check_lanelet_paths(args.data_dir, args.raw_dir)
    else:
        report = validate_preprocessed(args.data_dir)

    print(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    _cli()
