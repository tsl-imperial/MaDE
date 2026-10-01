# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Smoke test: CheckpointManager must absolutise its directory on construction."""

from pathlib import Path
import os

from made.utils.checkpointing import CheckpointManager


def test_relative_path_is_absolutised(tmp_path: Path) -> None:
    """Verify relative path is absolutised."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        cm = CheckpointManager("relative_subdir")
        assert cm.directory.is_absolute(), (
            f"CheckpointManager.directory is not absolute: {cm.directory}"
        )
        assert cm.directory == tmp_path / "relative_subdir"
    finally:
        os.chdir(old_cwd)


def test_absolute_path_unchanged(tmp_path: Path) -> None:
    """Verify absolute path unchanged."""
    cm = CheckpointManager(str(tmp_path / "ckpt"))
    assert cm.directory.is_absolute()
    assert cm.directory == tmp_path / "ckpt"
