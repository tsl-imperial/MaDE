# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""In-process runner for the simulated single-agent experiment matrix."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from made.utils.jax_setup import configure

configure()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from made.utils import e01_condition, load_config  # noqa: E402

_SYSTEM_CONFIG_PREFIXES = {
    "double_integrator": ["di"],
    "unicycle": ["unicycle"],
    "kinematic_bicycle": ["kinbicycle"],
    "dynamic_bicycle": ["dynbicycle_underspecified"],
}
_VARIANTS = (
    "made",
    "made-no-residual",
    "made-no-corrector",
    "made-supervised-i",
    "made-fixed-i",
    "mlp",
    "clamp",
)


def _csv_or_default(value: str | None, default: tuple[str, ...] | list[str]) -> list[str]:
    """Parse a comma-separated option, with ``all`` or None meaning the default.

    Args:
        value: The option value.
        default: Names used when ``value`` is None or ``all``.

    Returns:
        The list of names.
    """
    if value is None or value == "all":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_paths(config_dir: Path, systems: list[str], variants: list[str]) -> list[Path]:
    """Config file paths for every system and variant combination.

    Args:
        config_dir: Directory of the config JSON files.
        systems: System names.
        variants: Variant names.

    Returns:
        The config paths in system, prefix, variant order.

    Raises:
        ValueError: If a system or variant is unknown.
    """
    paths: list[Path] = []
    for system in systems:
        if system not in _SYSTEM_CONFIG_PREFIXES:
            supported = sorted(_SYSTEM_CONFIG_PREFIXES)
            raise ValueError(f"Unknown system '{system}'. Supported: {supported}")
        for prefix in _SYSTEM_CONFIG_PREFIXES[system]:
            for variant in variants:
                if variant not in _VARIANTS:
                    raise ValueError(f"Unknown variant '{variant}'. Supported: {list(_VARIANTS)}")
                paths.append(config_dir / f"{prefix}_{variant}.json")
    return paths


def _wandb_run_name(system: str, condition: str, variant: str, seed: int) -> str:
    """Run name used for experiment tracking.

    Args:
        system: System name.
        condition: Condition name.
        variant: Variant name.
        seed: Random seed.

    Returns:
        The run name.
    """
    return f"e01-{system}-{condition}-{variant}-{seed}"


def _resolve_pretrained_phase1_path(path: str | None, seed: int) -> str | None:
    """Substitute the seed marker in a pretrained_phase1_path for multi-seed sweeps.

    If ``path`` contains the literal token ``seed0``, replaces it with ``seed{seed}``
    so each seed uses its own Phase-1 checkpoint rather than seed 0's checkpoint.

    If the path does not contain ``seed0`` (e.g. a user-supplied absolute path with
    no seed marker), the path is returned unchanged — a multi-seed sweep sharing a
    single source checkpoint is arguably intentional in that case.

    Args:
        path: The configured Phase-1 checkpoint path, or None.
        seed: Seed whose checkpoint to use.

    Returns:
        The path for this seed, or None when ``path`` is None.
    """
    if path is None:
        return None
    if "seed0" in path:
        return path.replace("seed0", f"seed{seed}")
    return path


def main() -> None:
    """Run the simulated-experiment matrix in-process."""
    parser = argparse.ArgumentParser(description="Run the simulated-experiment matrix in-process.")
    parser.add_argument("--config-dir", default=str(ROOT / "configs" / "sim"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "sim" / "runs"))
    parser.add_argument("--systems", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-steps-per-epoch", type=int, default=None)
    parser.add_argument("--metric-log-interval-steps", type=int, default=None)
    parser.add_argument("--data-parallel-devices", type=int, default=None)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    systems = _csv_or_default(args.systems, tuple(_SYSTEM_CONFIG_PREFIXES))
    variants = _csv_or_default(args.variants, _VARIANTS)
    seeds = [int(seed) for seed in _csv_or_default(args.seeds, ("0",))]

    for config_path in _config_paths(config_dir, systems, variants):
        cfg = load_config(str(config_path))
        for seed in seeds:
            condition = e01_condition(cfg)
            training_overrides: dict = {"seed": seed}
            resolved_path = _resolve_pretrained_phase1_path(
                cfg.training.pretrained_phase1_path, seed
            )
            if resolved_path != cfg.training.pretrained_phase1_path:
                training_overrides["pretrained_phase1_path"] = resolved_path
            if args.steps_per_epoch is not None:
                training_overrides["steps_per_epoch"] = args.steps_per_epoch
            if args.val_steps_per_epoch is not None:
                training_overrides["val_steps_per_epoch"] = args.val_steps_per_epoch
            if args.metric_log_interval_steps is not None:
                training_overrides["metric_log_interval_steps"] = args.metric_log_interval_steps
            seeded_cfg = replace(
                cfg,
                training=replace(cfg.training, **training_overrides),
                evaluation=replace(
                    cfg.evaluation,
                    perturbation_seed=cfg.evaluation.perturbation_seed + seed,
                ),
            )
            cell_dir = (
                Path(args.output_root)
                / seeded_cfg.physics.true_system
                / condition
                / seeded_cfg.experiment_name
                / f"seed{seed}"
            )
            test_data = str(Path(args.data_dir) / seeded_cfg.physics.true_system)
            metrics_path = str(cell_dir / "metrics.json")
            train_call = (
                "train.main_programmatic"
                f"(config={config_path}, output_dir={cell_dir}, data_dir={args.data_dir})"
            )
            eval_call = (
                "evaluate.main_programmatic"
                f"(variant={cfg.experiment_name}, test_data={test_data}, "
                f"output={metrics_path}, eval_regime={condition})"
            )

            if args.dry_run:
                if cfg.experiment_name == "clamp":
                    print(f"SKIP TRAIN: {config_path}")
                else:
                    print(f"TRAIN: {train_call}")
                print(f"EVALUATE: {eval_call}")
                continue

            from scripts.sim import evaluate, train

            if seeded_cfg.experiment_name == "clamp":
                train_result = {"checkpoint_dir": None, "variant": "clamp"}
            else:
                previous_wandb_name = os.environ.get("MADE_WANDB_RUN_NAME")
                os.environ["MADE_WANDB_RUN_NAME"] = _wandb_run_name(
                    seeded_cfg.physics.true_system,
                    condition,
                    seeded_cfg.experiment_name,
                    seed,
                )
                try:
                    train_result = train.main_programmatic(
                        seeded_cfg,
                        str(cell_dir),
                        data_dir=args.data_dir,
                        data_parallel_devices=args.data_parallel_devices,
                    )
                finally:
                    if previous_wandb_name is None:
                        os.environ.pop("MADE_WANDB_RUN_NAME", None)
                    else:
                        os.environ["MADE_WANDB_RUN_NAME"] = previous_wandb_name

            evaluate.main_programmatic(
                seeded_cfg,
                checkpoint=train_result["checkpoint_dir"],
                variant=train_result["variant"],
                test_data=test_data,
                output=metrics_path,
                eval_regime=condition,
            )


if __name__ == "__main__":
    main()
