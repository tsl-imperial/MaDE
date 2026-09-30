"""Recording-level stratified train/val/test split assignment.

Within each location, one recording is held out for val and one for test; the rest go
to train. heckstrasse has only 3 recordings, giving a 1/1/1 assignment.

Canonical assignment:
  train: [0,1,2,3,4, 7,8,9,10,11,12,13,14,15, 18,19,20,21,22,23,24,25,26,27, 30]
  val:   [5, 16, 28, 31]
  test:  [6, 17, 29, 32]

The canonical split is **hardcoded** in this module — hand-picked so that all 4
locations appear in every split, and construction recordings (11-17) appear in
train (11-15), val (16), and test (17), exercising the ``map_variant=1`` code path
on val/test.

The ``seed`` parameter on :func:`assign_splits` is a provenance label only with the
default ``LOCATION_TABLE`` (the canonical assignment is returned regardless of seed).
When ``location_table`` is overridden (e.g. ablations on a location subset), seed
drives a deterministic per-location shuffle. Do not regenerate this split without
updating downstream baselines.
"""

from __future__ import annotations

from made.data.ind.constants import LOCATION_TABLE


SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

# Canonical hand-picked splits. Documented above; do not modify without also
# updating any downstream baseline configs that pin the recording IDs.
_DEFAULT_TRAIN: tuple[int, ...] = (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30)
_DEFAULT_VAL: tuple[int, ...] = (5, 16, 28, 31)
_DEFAULT_TEST: tuple[int, ...] = (6, 17, 29, 32)

_DEFAULT_ASSIGNMENT: dict[int, str] = (
    {rec: "train" for rec in _DEFAULT_TRAIN}
    | {rec: "val" for rec in _DEFAULT_VAL}
    | {rec: "test" for rec in _DEFAULT_TEST}
)


def assign_splits(
    location_table: dict[int, dict] | None = None,
    seed: int = 20260505,
) -> dict[int, str]:
    """Assign each recording ID to ``'train'``, ``'val'``, or ``'test'``.

    With the default ``location_table`` (``LOCATION_TABLE`` from
    :mod:`made.data.ind.constants`), returns the canonical hand-picked splits verbatim;
    ``seed`` is a provenance tag only, it does not drive a shuffle.

    With an overridden ``location_table`` (e.g. an ablation on a subset of locations),
    falls back to a deterministic per-location shuffle keyed by ``seed``.

    Returns:
        Mapping ``recording_id`` -> split name.
    """
    if location_table is None:
        location_table = LOCATION_TABLE

    # Canonical hardcoded assignment — seed is decorative here.
    if location_table is LOCATION_TABLE:
        return dict(_DEFAULT_ASSIGNMENT)

    import random

    rng = random.Random(seed)
    assignment: dict[int, str] = {}

    for _loc_id, info in sorted(location_table.items()):
        recordings = sorted(info["recordings"])
        n = len(recordings)
        if n < 3:
            raise ValueError(
                f"Location {_loc_id} has only {n} recording(s); need at least 3 for "
                f"a 1-val / 1-test split."
            )
        # Shuffle then assign: last -> test, second-to-last -> val, rest -> train
        shuffled = recordings[:]
        rng.shuffle(shuffled)
        for rec in shuffled[:-2]:
            assignment[rec] = "train"
        assignment[shuffled[-2]] = "val"
        assignment[shuffled[-1]] = "test"

    return assignment


def get_default_splits() -> dict[int, str]:
    """Return the canonical P10 split assignment (seed=20260505)."""
    return dict(_DEFAULT_ASSIGNMENT)


def recording_ids_for_split(split: str, assignment: dict[int, str] | None = None) -> list[int]:
    """Return sorted list of recording IDs assigned to *split*."""
    if assignment is None:
        assignment = get_default_splits()
    return sorted(rec for rec, s in assignment.items() if s == split)
