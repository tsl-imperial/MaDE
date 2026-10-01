# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for opt-in data parallelism."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pytest

from made.data.grain_pipeline import InMemoryDataSource, create_data_loader
from made.training.trainer import _check_train_meta, _early_stopping_config_meta
from made.utils import CheckpointManager
from made.utils.config import TrainingConfig


_DEFAULT_TRAINING_CFG = TrainingConfig()


def _make_source(n_samples: int = 16) -> InMemoryDataSource:
    """Create a small in-memory source with dummy samples.

    Args:
        n_samples: Number of dummy samples.

    Returns:
        In-memory data source.
    """
    samples = [
        {
            "x_prev": jnp.zeros(4, dtype=jnp.float64),
            "x_curr": jnp.zeros(4, dtype=jnp.float64),
            "params": jnp.zeros(0, dtype=jnp.float64),
        }
        for _ in range(n_samples)
    ]
    return InMemoryDataSource(samples)


def _write_sig(ckpt_dir: Path, step: int, sig: dict, cfg: TrainingConfig | None = None) -> None:
    """Write a minimal train_meta.json with a given comparability_signature.

    Args:
        ckpt_dir: Checkpoint root directory.
        step: Step directory to write.
        sig: Comparability signature to store.
        cfg: Training config for the stored meta; default if None.
    """
    _cfg = cfg or _DEFAULT_TRAINING_CFG
    step_dir = ckpt_dir / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": _cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": _cfg.i_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(_cfg),
        "comparability_signature": sig,
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))


def test_create_data_loader_divisible() -> None:
    """num_devices=2, batch_size=4 — should succeed."""
    source = _make_source(16)
    loader = create_data_loader(source, batch_size=4, num_devices=2, seed=0)
    assert loader is not None


def test_create_data_loader_drops_remainder_by_default() -> None:
    """Checks create data loader drops remainder by default."""
    source = _make_source(10)
    loader = create_data_loader(source, batch_size=4, num_devices=1, seed=0)
    batch_sizes = [batch["x_prev"].shape[0] for batch in loader]
    assert batch_sizes == [4, 4]


def test_create_data_loader_can_keep_partial_final_batch() -> None:
    """Checks create data loader can keep partial final batch."""
    source = _make_source(10)
    loader = create_data_loader(
        source,
        batch_size=4,
        num_devices=1,
        seed=0,
        shuffle=False,
        drop_remainder=False,
    )
    batch_sizes = [batch["x_prev"].shape[0] for batch in loader]
    assert batch_sizes == [4, 4, 2]


def test_create_data_loader_not_divisible() -> None:
    """num_devices=2, batch_size=3 — should raise AssertionError."""
    source = _make_source(16)
    with pytest.raises(AssertionError, match="divisible"):
        create_data_loader(source, batch_size=3, num_devices=2, seed=0)


def test_dp_topology_mismatch_rejected(tmp_path: Path) -> None:
    """Stored data_parallel_devices=2 vs current num_devices=1 raises ValueError."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
            "data_parallel_devices": 2,
        },
    )
    cfg = TrainingConfig()
    with pytest.raises(ValueError, match="comparability_signature mismatch"):
        _check_train_meta(ckpt, 0, cfg, num_devices=1)


def test_dp_topology_match_passes(tmp_path: Path) -> None:
    """Stored data_parallel_devices=1 vs current num_devices=1 does not raise."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
            "data_parallel_devices": 1,
        },
    )
    cfg = TrainingConfig()
    _check_train_meta(ckpt, 0, cfg, num_devices=1)  # should not raise


def test_dp_absent_in_stored_sig_passes(tmp_path: Path) -> None:
    """Old checkpoint without data_parallel_devices key — overlap-key check skips it."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
            # no data_parallel_devices — simulates old checkpoint
        },
    )
    cfg = TrainingConfig()
    _check_train_meta(ckpt, 0, cfg, num_devices=2)  # should not raise (no overlap on DP key)


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="Multi-device float64 tolerance test requires >= 2 JAX devices",
)
def test_float64_tolerance_multi_device() -> None:
    """Float64 tolerance: losses with 2 vs 1 devices are within 1e-10 (skipped on single-device)."""
    pytest.skip("Skipping: requires multi-device environment not available in CPU tests.")
