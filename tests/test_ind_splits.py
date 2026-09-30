"""Phase 3 acceptance tests for made.data.ind.splits.

Covers:
- assign_splits() returns pairwise-disjoint train/val/test.
- Canonical P10 assignment matches the documented tuples verbatim.
- Seeded shuffle is deterministic and seed-sensitive when location_table is overridden.
"""

from __future__ import annotations

from made.data.ind.splits import (
    SPLIT_NAMES,
    _DEFAULT_TEST,
    _DEFAULT_TRAIN,
    _DEFAULT_VAL,
    assign_splits,
)


def test_splits_pairwise_disjoint() -> None:
    assignment = assign_splits()
    by_split: dict[str, set[int]] = {s: set() for s in SPLIT_NAMES}
    for rec_id, split in assignment.items():
        by_split[split].add(rec_id)
    for i, s1 in enumerate(SPLIT_NAMES):
        for s2 in SPLIT_NAMES[i + 1 :]:
            overlap = by_split[s1] & by_split[s2]
            assert not overlap, f"Splits '{s1}' and '{s2}' overlap: {overlap}"


def test_canonical_p10_train() -> None:
    assignment = assign_splits()
    train_ids = frozenset(rec for rec, s in assignment.items() if s == "train")
    assert train_ids == frozenset(_DEFAULT_TRAIN)


def test_canonical_p10_val() -> None:
    assignment = assign_splits()
    val_ids = frozenset(rec for rec, s in assignment.items() if s == "val")
    assert val_ids == frozenset(_DEFAULT_VAL)


def test_canonical_p10_test() -> None:
    assignment = assign_splits()
    test_ids = frozenset(rec for rec, s in assignment.items() if s == "test")
    assert test_ids == frozenset(_DEFAULT_TEST)


def test_seed_is_decorative_for_default_table() -> None:
    """Different seeds must return the same canonical P10 assignment."""
    a1 = assign_splits(seed=1)
    a2 = assign_splits(seed=99999)
    assert a1 == a2


def _mini_location_table() -> dict[int, dict]:
    """One location with 6 recordings — enough to get val+test holdouts."""
    return {99: {"name": "test_loc", "recordings": [10, 11, 12, 13, 14, 15]}}


def test_override_table_seed_deterministic() -> None:
    """Same seed with overridden location_table gives identical results on two calls."""
    table = _mini_location_table()
    a1 = assign_splits(location_table=table, seed=42)
    a2 = assign_splits(location_table=table, seed=42)
    assert a1 == a2


def test_override_table_different_seeds_differ() -> None:
    """Different seeds with overridden location_table yield different assignments."""
    table = _mini_location_table()
    a1 = assign_splits(location_table=table, seed=1)
    a2 = assign_splits(location_table=table, seed=2)
    # With 6 recordings and a per-location shuffle it's overwhelmingly likely they differ.
    assert a1 != a2


def test_override_table_all_recordings_assigned() -> None:
    """Every recording in the overridden table must appear in the result."""
    table = _mini_location_table()
    assignment = assign_splits(location_table=table, seed=7)
    expected_ids = frozenset(table[99]["recordings"])
    assert frozenset(assignment.keys()) == expected_ids


def test_override_table_splits_disjoint() -> None:
    table = _mini_location_table()
    assignment = assign_splits(location_table=table, seed=7)
    by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for rec_id, split in assignment.items():
        by_split[split].add(rec_id)
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])
