# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""CPU smoke tests for scripts/generate_data.py's dataset-size CLI knobs.

The override case runs the script as a subprocess against the cheapest
system (double_integrator) to validate the full CLI path end to end. The
no-override (default-size) case does NOT generate a 1024-trajectory
dataset — that would be slow for a "fast" test — and instead imports
``build_data_config`` directly to assert it reproduces the DataConfig
defaults byte-for-byte.
"""

from __future__ import annotations

from types import ModuleType
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "sim" / "generate_data.py"


def _env() -> dict:
    """Return the process environment pinned to the CPU JAX backend.

    Returns:
        Environment mapping for subprocess calls.
    """
    return {**os.environ, "JAX_PLATFORMS": "cpu"}


def _load_generate_data_module() -> ModuleType:
    """Import the data-generation script as a module by file path.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("generate_data", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_data_config_defaults_when_all_none() -> None:
    """No CLI overrides must reproduce the DataConfig defaults exactly."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    config = module.build_data_config(None, None, None, None)
    assert config == DataConfig()


def test_build_data_config_overrides_only_set_fields() -> None:
    """Verify build data config overrides only set fields."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    default = DataConfig()
    config = module.build_data_config(8, 4, 4, 28)
    assert config.num_trajectories_train == 8
    assert config.num_trajectories_val == 4
    assert config.num_trajectories_test == 4
    assert config.trajectory_length == 28
    # Untouched fields fall back to the default.
    assert config.perturbation_type == default.perturbation_type
    assert config.perturbation_scale == default.perturbation_scale
    assert config.noise_scale == default.noise_scale
    assert config.control_profile == default.control_profile
    assert config.control_tau == default.control_tau


def test_build_data_config_control_profile_defaults_when_none() -> None:
    """No --control-profile/--control-tau overrides must reproduce the
    DataConfig defaults exactly (byte-identical default-path generation)."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    config = module.build_data_config(None, None, None, None, None, None)
    assert config == DataConfig()


def test_build_data_config_control_profile_override() -> None:
    """Verify build data config control profile override."""
    module = _load_generate_data_module()

    config = module.build_data_config(None, None, None, None, "smooth_ou", 2.0)
    assert config.control_profile == "smooth_ou"
    assert config.control_tau == 2.0


def test_build_data_config_control_tau_override_only() -> None:
    """Verify build data config control tau override only."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    default = DataConfig()
    config = module.build_data_config(None, None, None, None, None, 3.0)
    assert config.control_profile == default.control_profile
    assert config.control_tau == 3.0


def test_build_data_config_min_speed_defaults_when_none() -> None:
    """No --min-speed override must reproduce the DataConfig default (None)."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    config = module.build_data_config(None, None, None, None, None, None, None)
    assert config == DataConfig()
    assert config.min_speed is None


def test_build_data_config_min_speed_override() -> None:
    """Verify build data config min speed override."""
    module = _load_generate_data_module()
    from made.utils.config import DataConfig

    default = DataConfig()
    config = module.build_data_config(None, None, None, None, None, None, 1.0)
    assert config.min_speed == 1.0
    # Untouched fields fall back to the default.
    assert config.control_profile == default.control_profile
    assert config.control_tau == default.control_tau


def test_cli_override_generates_requested_sizes(tmp_path: Path) -> None:
    """Verify cli override generates requested sizes."""
    output_dir = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--system",
            "double_integrator",
            "--output-dir",
            str(output_dir),
            "--num-train",
            "8",
            "--num-val",
            "4",
            "--num-test",
            "4",
            "--trajectory-length",
            "28",
        ],
        cwd=_REPO_ROOT,
        env={**_env(), "PYTHONPATH": str(_REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    system_dir = output_dir / "double_integrator"
    expected_sizes = {"train": 8, "val": 4, "test": 4}
    state_dim = None
    for split, num_trajectories in expected_sizes.items():
        split_dir = system_dir / split
        assert split_dir.is_dir(), f"missing split dir: {split_dir}"

        states = np.load(split_dir / "states.npy")
        controls = np.load(split_dir / "controls.npy")
        assert states.shape[0] == num_trajectories
        assert states.shape[1] == 28
        if state_dim is None:
            state_dim = states.shape[2]
        else:
            assert states.shape[2] == state_dim
        assert controls.shape[0] == num_trajectories
        # Controls span the transitions between states: one fewer step than states.
        assert controls.shape[1] == 28 - 1

    metadata = json.loads((system_dir / "metadata.json").read_text())
    data_config = metadata["data_config"]
    assert data_config["num_trajectories_train"] == 8
    assert data_config["num_trajectories_val"] == 4
    assert data_config["num_trajectories_test"] == 4
    assert data_config["trajectory_length"] == 28
