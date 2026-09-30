"""Tests for the E01 in-process runner (run_matrix.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sim.run_matrix import _resolve_pretrained_phase1_path


# ---------------------------------------------------------------------------
# M1 — seed substitution in pretrained_phase1_path
# ---------------------------------------------------------------------------


def test_run_e01_seed_substitutes_pretrained_phase1_path():
    """When the path contains 'seed0', it is rewritten to 'seed{N}' for the given seed."""
    path = "outputs/foo/seed0/checkpoints"
    result = _resolve_pretrained_phase1_path(path, seed=3)
    assert result == "outputs/foo/seed3/checkpoints"


def test_run_e01_seed_leaves_path_alone_if_no_seed_marker():
    """When the path has no 'seed0' token, it is returned unchanged for any seed."""
    path = "outputs/foo/checkpoints"
    assert _resolve_pretrained_phase1_path(path, seed=1) == path
    assert _resolve_pretrained_phase1_path(path, seed=5) == path


def test_run_e01_seed_no_pretrained_path_is_noop():
    """When pretrained_phase1_path is None, the helper returns None regardless of seed."""
    assert _resolve_pretrained_phase1_path(None, seed=0) is None
    assert _resolve_pretrained_phase1_path(None, seed=2) is None
