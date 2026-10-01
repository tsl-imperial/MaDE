# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests that `generate_and_save` actually threads `control_sample_min/max` through to the
simulator.

Regression guard for D21: `generate_and_save` used to drop `data_config.control_sample_min/max`
on the floor, so the release generator's narrow dynamic-bicycle box (+-0.2 steering, +-1.5
accel) silently had no effect.
"""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from made.data.simulation_data import generate_and_save, load_split
from made.utils.config import DataConfig, PhysicsConfig
from pathlib import Path


def test_generate_and_save_respects_control_sample_bounds(tmp_path: Path) -> None:
    """Checks generate and save respects control sample bounds."""
    physics_config = PhysicsConfig(true_system="dynamic_bicycle", dt=0.1, true_params={})
    control_sample_min = (-0.2, -1.5)
    control_sample_max = (0.2, 1.5)
    data_config = DataConfig(
        num_trajectories_train=8,
        num_trajectories_val=4,
        num_trajectories_test=4,
        trajectory_length=8,
        control_sample_min=control_sample_min,
        control_sample_max=control_sample_max,
    )
    key = jax.random.key(0)

    output_dir = tmp_path / "db_narrow"
    generate_and_save(physics_config, data_config, str(output_dir), key)

    for split in ("train", "val", "test"):
        _, controls = load_split(str(output_dir), split)
        controls_np = np.asarray(controls)
        for dim, (lo, hi) in enumerate(zip(control_sample_min, control_sample_max)):
            assert controls_np[..., dim].min() >= lo
            assert controls_np[..., dim].max() <= hi


def test_generate_and_save_without_override_can_exceed_narrow_box(tmp_path: Path) -> None:
    # Without control_sample_min/max, dynamic-bicycle steering is drawn from the constraint
    # set's own (wider) bounds, so it must be able to exceed the narrow +-0.2 box above.
    """Checks generate and save without override can exceed narrow box."""
    physics_config = PhysicsConfig(true_system="dynamic_bicycle", dt=0.1, true_params={})
    data_config = DataConfig(
        num_trajectories_train=32,
        num_trajectories_val=4,
        num_trajectories_test=4,
        trajectory_length=8,
    )
    key = jax.random.key(0)

    output_dir = tmp_path / "db_wide"
    generate_and_save(physics_config, data_config, str(output_dir), key)

    _, train_controls = load_split(str(output_dir), "train")
    train_controls_np = np.asarray(train_controls)
    assert train_controls_np[..., 0].max() > 0.2 or train_controls_np[..., 0].min() < -0.2
