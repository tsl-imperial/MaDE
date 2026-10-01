# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Underspecified dispatch E2E integration tests.

Tests that model.known_system != physics.true_system wires correctly through
the full train→evaluate pipeline.  Uses double_integrator (true) and unicycle
(known) because both systems share state_dim=4, control_dim=2, param_dim=0,
which allows MaDECell to process double-integrator data using unicycle physics
— the cleanest dimension-compatible underspecified proxy available.
"""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.models import MaDECell
from made.physics import build_system
from made.utils import CheckpointManager
from made.utils.checkpointing import TrainState
from made.utils.config import (
    CorrectorConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PhysicsConfig,
)

import pathlib
from scripts.sim.evaluate import main_programmatic

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_split(root: pathlib.Path, split: str, n: int, t: int, s: int, c: int) -> None:
    """Write all-zero `states.npy` and `controls.npy` for one split.

    Args:
        root: Dataset root directory.
        split: Split sub-directory name.
        n: Number of trajectories.
        t: Number of time steps.
        s: State dimension.
        c: Control dimension.
    """
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(split_dir / "states.npy"), np.zeros((n, t, s), dtype=np.float64))
    np.save(str(split_dir / "controls.npy"), np.zeros((n, t - 1, c), dtype=np.float64))


def test_kinbike_as_known_dynbike_as_true(tmp_path: "Path") -> None:
    """All 6 metric keys are finite for true=double_integrator, known=unicycle wiring."""
    state_dim, control_dim = 4, 2
    data_dir = tmp_path / "data"
    _write_split(data_dir, "test", n=4, t=8, s=state_dim, c=control_dim)

    known_physics, known_constraints = build_system("unicycle")
    model_cfg = ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16))
    corrector_cfg = CorrectorConfig(mode="enabled", train_steps=2)
    cell = MaDECell.from_config(
        known_physics, known_constraints, model_cfg, corrector_cfg, key=jax.random.key(0)
    )

    checkpoint_dir = str(tmp_path / "ckpt")
    ckpt_state = TrainState(
        model=cell, opt_state_I=None, opt_state_T=None, key=jax.random.key(1), step=0
    )
    CheckpointManager(checkpoint_dir).save(ckpt_state, 0)

    cfg = ExperimentConfig(
        physics=PhysicsConfig(true_system="double_integrator"),
        model=ModelConfig(
            known_system="unicycle",
            inverse_hidden=(16, 16),
            residual_hidden=(16, 16),
        ),
        corrector=corrector_cfg,
        data=DataConfig(perturbation_scale=0.05),
        evaluation=EvaluationConfig(perturbation_seed=42),
    )

    result = main_programmatic(
        cfg=cfg,
        checkpoint=checkpoint_dir,
        variant="made",
        test_data=str(data_dir),
        output=str(tmp_path / "metrics.json"),
    )

    metrics = result["metrics"]
    expected_keys = {
        "inequality_violation_rate",
        "inequality_violation_magnitude",
        "dynamics_violation_known",
        "dynamics_violation_learned",
        "dynamics_violation_true",
        "fidelity",
    }
    assert set(metrics.keys()) == expected_keys, (
        f"Missing metric keys: {expected_keys - set(metrics.keys())}"
    )
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"Metric {k} = {v} is not finite"
