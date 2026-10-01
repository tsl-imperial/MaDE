# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Entry point for data generation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from made.utils.jax_setup import configure

configure()

from made.data import generate_and_save  # noqa: E402
from made.utils.config import DataConfig, PhysicsConfig  # noqa: E402
from made.utils.keys import init_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build_data_config(
    num_train: int | None,
    num_val: int | None,
    num_test: int | None,
    trajectory_length: int | None,
    control_profile: str | None = None,
    control_tau: float | None = None,
    min_speed: float | None = None,
    noise_scale: float | None = None,
    add_generation_noise: bool = False,
) -> DataConfig:
    """Construct a DataConfig, overriding only the fields that are not None.

    Arguments left as None fall back to the DataConfig default.

    Args:
        num_train: Number of training trajectories.
        num_val: Number of validation trajectories.
        num_test: Number of test trajectories.
        trajectory_length: Steps per trajectory.
        control_profile: Control sampling profile name.
        control_tau: Control profile time constant.
        min_speed: Minimum speed.
        noise_scale: Observation noise scale.
        add_generation_noise: Add noise to the generated data itself.

    Returns:
        The data configuration with the given overrides applied.
    """
    overrides = {
        "num_trajectories_train": num_train,
        "num_trajectories_val": num_val,
        "num_trajectories_test": num_test,
        "trajectory_length": trajectory_length,
        "control_profile": control_profile,
        "control_tau": control_tau,
        "min_speed": min_speed,
        "noise_scale": noise_scale,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if add_generation_noise:
        overrides["add_generation_noise"] = True
    return dataclasses.replace(DataConfig(), **overrides)


def apply_generation_config(data_config: DataConfig, system: str, generation_config_path: str) -> DataConfig:
    """Apply per-system control-sampling overrides recorded in a generation-config JSON.

    If ``system`` is a key of the JSON at ``generation_config_path``, its entries are
    applied as ``control_sample_min``/``control_sample_max`` overrides on ``data_config``.
    Systems not present in the JSON leave ``data_config`` unchanged.

    Args:
        data_config: The base data configuration.
        system: System name to look up in the JSON.
        generation_config_path: Path of the generation-config JSON.

    Returns:
        The configuration, with control-sampling overrides when ``system`` has an entry.

    Raises:
        FileNotFoundError: if ``generation_config_path`` does not exist.
    """
    path = Path(generation_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"generation config not found: {path}")
    with path.open(encoding="utf-8") as f:
        overrides_by_system = json.load(f)
    entry = overrides_by_system.get(system)
    if entry is None:
        return data_config
    return dataclasses.replace(
        data_config,
        control_sample_min=tuple(entry["control_sample_min"]),
        control_sample_max=tuple(entry["control_sample_max"]),
    )


def main() -> None:
    """Generate and save the simulated data for one system."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="double_integrator")
    parser.add_argument("--output-dir", default=str(ROOT / "data/generated"))
    parser.add_argument(
        "--generation-config",
        default=str(ROOT / "configs/sim/data_generation.json"),
        help=(
            "JSON file mapping system name to control-sampling overrides "
            "(control_sample_min/control_sample_max). Applied only to systems present as keys."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--trajectory-length", type=int, default=None)
    parser.add_argument(
        "--control-profile",
        choices=["iid_uniform", "smooth_ou"],
        default=None,
        help="Control sampling profile. Defaults to the DataConfig default (iid_uniform).",
    )
    parser.add_argument(
        "--control-tau",
        type=float,
        default=None,
        help="OU correlation time (seconds) for --control-profile smooth_ou.",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=None,
        help=(
            "Reject trajectories whose speed ever drops below this floor "
            "(dynamic-bicycle 6D layout only). Defaults to no floor."
        ),
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help=(
            "Observation-noise standard deviation. Applied at generation time only when "
            "--add-generation-noise is also passed; otherwise recorded in the config and "
            "consumed at load time."
        ),
    )
    parser.add_argument(
        "--add-generation-noise",
        action="store_true",
        help="Add zero-mean Gaussian observation noise to generated states on every split, "
             "at --noise-scale. Off by default.",
    )
    args = parser.parse_args()

    systems = (
        ["double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle"]
        if args.system == "all"
        else [args.system]
    )
    data_config = build_data_config(
        args.num_train,
        args.num_val,
        args.num_test,
        args.trajectory_length,
        args.control_profile,
        args.control_tau,
        args.min_speed,
        args.noise_scale,
        args.add_generation_noise,
    )
    print(f"effective DataConfig: {data_config}")
    keys = init_keys(args.seed)
    for system in systems:
        system_data_config = apply_generation_config(data_config, system, args.generation_config)
        config = PhysicsConfig(true_system=system)
        generate_and_save(config, system_data_config, f"{args.output_dir}/{system}", keys["data"])
        print(f"generated {system} data under {args.output_dir}/{system}")


if __name__ == "__main__":
    main()
