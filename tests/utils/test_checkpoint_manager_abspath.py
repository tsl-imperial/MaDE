"""Smoke test: CheckpointManager must absolutise its directory on construction."""

import os

from made.utils.checkpointing import CheckpointManager


def test_relative_path_is_absolutised(tmp_path):
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


def test_absolute_path_unchanged(tmp_path):
    cm = CheckpointManager(str(tmp_path / "ckpt"))
    assert cm.directory.is_absolute()
    assert cm.directory == tmp_path / "ckpt"
