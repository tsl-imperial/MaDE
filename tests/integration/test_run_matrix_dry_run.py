# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Dry-run smoke for ``scripts/sim/run_matrix.py``.

Asserts that the in-process simulated-experiment driver enumerates the expected
TRAIN / EVALUATE / SKIP TRAIN lines for a small (system x variant x seed)
matrix.

Key facts about the dry-run output:
  - ``TRAIN:``      -- ``train.main_programmatic(...)`` call repr
  - ``EVALUATE:``   -- ``evaluate.main_programmatic(...)`` call repr
  - ``SKIP TRAIN:`` -- clamp config path (no training, parameterless)

CPU-only; JAX is imported by ``run_matrix.py`` itself but does not compile
any kernels in dry-run mode so each test finishes in < 3 s.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(
    *,
    systems: str,
    variants: str,
    seeds: str,
) -> str:
    """Invoke ``run_matrix.py --dry-run`` and return combined stdout.

    Args:
        systems: Comma-separated systems.
        variants: Comma-separated variants.
        seeds: Comma-separated seeds.

    Returns:
        Combined stdout of the dry run.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "sim" / "run_matrix.py"),
            "--dry-run",
            "--systems",
            systems,
            "--variants",
            variants,
            "--seeds",
            seeds,
            "--output-root",
            "/tmp/e01_test_output_root",
            "--config-dir",
            "configs/sim",
            "--data-dir",
            "data/generated",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_made_kinbicycle_two_seeds() -> None:
    """kinematic_bicycle / made × 2 seeds → 2 TRAIN + 2 EVALUATE lines."""
    out = _dry_run(systems="kinematic_bicycle", variants="made", seeds="0,1")
    train_lines = [line for line in out.splitlines() if line.startswith("TRAIN:")]
    eval_lines = [line for line in out.splitlines() if line.startswith("EVALUATE:")]

    assert len(train_lines) == 2, f"expected 2 TRAIN lines, got {len(train_lines)}: {out}"
    assert len(eval_lines) == 2, f"expected 2 EVALUATE lines, got {len(eval_lines)}: {out}"

    # Each TRAIN line references the config and output dir with the correct seed subdir.
    assert any("seed0" in line for line in train_lines), out
    assert any("seed1" in line for line in train_lines), out

    # Each EVALUATE line references the correct metrics.json path.
    assert any("seed0/metrics.json" in line for line in eval_lines), out
    assert any("seed1/metrics.json" in line for line in eval_lines), out


def test_clamp_skips_training() -> None:
    """clamp must emit SKIP TRAIN (not TRAIN) but still emit EVALUATE."""
    out = _dry_run(systems="kinematic_bicycle", variants="clamp", seeds="0")
    train_lines = [line for line in out.splitlines() if line.startswith("TRAIN:")]
    skip_lines = [line for line in out.splitlines() if line.startswith("SKIP TRAIN:")]
    eval_lines = [line for line in out.splitlines() if line.startswith("EVALUATE:")]

    assert len(train_lines) == 0, f"clamp must not emit TRAIN: {out}"
    assert len(skip_lines) == 1, f"expected 1 SKIP TRAIN for clamp, got {len(skip_lines)}: {out}"
    assert len(eval_lines) == 1, f"expected 1 EVALUATE for clamp, got {len(eval_lines)}: {out}"

    # The skip line references the config path.
    assert "kinbicycle_clamp.json" in skip_lines[0], skip_lines[0]


def test_multi_variant_count() -> None:
    """kinematic_bicycle / made+clamp+mlp × 1 seed → 2 TRAIN + 1 SKIP TRAIN + 3 EVALUATE."""
    out = _dry_run(systems="kinematic_bicycle", variants="made,clamp,mlp", seeds="0")
    train_lines = [line for line in out.splitlines() if line.startswith("TRAIN:")]
    skip_lines = [line for line in out.splitlines() if line.startswith("SKIP TRAIN:")]
    eval_lines = [line for line in out.splitlines() if line.startswith("EVALUATE:")]

    assert len(train_lines) == 2, f"expected 2 TRAIN, got {len(train_lines)}: {out}"
    assert len(skip_lines) == 1, f"expected 1 SKIP TRAIN (clamp), got {len(skip_lines)}: {out}"
    assert len(eval_lines) == 3, f"expected 3 EVALUATE, got {len(eval_lines)}: {out}"


def test_dynbicycle_canonical_emits_no_cross_eval() -> None:
    """Canonical dynamic_bicycle/made emits exactly one TRAIN + one EVALUATE,
    and zero cross-eval lines (the DB-N → DB-native cross-eval has been retired
    along with the -noise condition)."""
    out = _dry_run(systems="dynamic_bicycle", variants="made", seeds="0")
    train_lines = [line for line in out.splitlines() if line.startswith("TRAIN:")]
    eval_lines = [line for line in out.splitlines() if line.startswith("EVALUATE:")]
    cross_lines = [
        line for line in out.splitlines()
        if line.startswith("EVALUATE (cross")
    ]

    assert len(train_lines) == 1, f"expected 1 TRAIN line, got {len(train_lines)}: {out}"
    assert len(eval_lines) == 1, f"expected 1 EVALUATE line, got {len(eval_lines)}: {out}"
    assert cross_lines == [], (
        f"expected zero cross-eval lines for dynamic_bicycle/made: {out}"
    )


def test_seed_paths_are_isolated() -> None:
    """Each seed cell writes under .../seed<N>/, never sharing a directory."""
    out = _dry_run(systems="unicycle", variants="made", seeds="0,1,2")
    train_lines = [line for line in out.splitlines() if line.startswith("TRAIN:")]

    assert any("seed0" in line for line in train_lines), out
    assert any("seed1" in line for line in train_lines), out
    assert any("seed2" in line for line in train_lines), out

    # No two seeds share the same output_dir token in a single TRAIN line.
    for line in train_lines:
        seed_tokens = [tok for tok in ("seed0", "seed1", "seed2") if tok in line]
        assert len(seed_tokens) == 1, (
            f"TRAIN line references more than one seed dir: {line}"
        )
