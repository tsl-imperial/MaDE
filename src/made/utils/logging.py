"""Weights and Biases logging helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import wandb

from made.utils.config import ExperimentConfig

# When set, every log_metrics call also appends a JSON line to this file.
_local_metrics_path: str | None = None


def e01_condition(config: ExperimentConfig) -> str:
    """Return the E01 condition label used for W&B metadata and output paths.

    The condition is derived from ``model.known_system``: ``"underspecified"``
    when MaDE's known physics differs from the true data-generating physics,
    ``"fully-specified"`` otherwise. Training-time augmentation
    (``data.noise_scale``) and eval-time perturbation (``data.perturbation_type``)
    do not affect the label — the canonical underspecified DB row applies a
    small Gaussian augmentation by default and is evaluated under the same
    bound-violation protocol as every other cell.
    """
    return "underspecified" if config.model.known_system else "fully-specified"


def set_local_metrics_path(path: str | None) -> None:
    """Configure a local JSONL file to mirror all log_metrics calls."""
    global _local_metrics_path
    _local_metrics_path = path
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def init_logging(config: ExperimentConfig, project: str = "made") -> None:
    """Initialise a wandb run for the given experiment."""
    if wandb.run is not None:
        wandb.finish()

    system = config.physics.true_system
    variant = config.experiment_name
    seed = config.training.seed
    condition = e01_condition(config)

    run_name = os.environ.get(
        "MADE_WANDB_RUN_NAME",
        f"e01-{system}-{condition}-{variant}-{seed}",
    )

    config_dict = asdict(config)
    config_dict.update(
        e01_system=system,
        e01_variant=variant,
        e01_seed=seed,
        e01_condition=condition,
    )

    wandb.init(
        project=project,
        config=config_dict,
        name=run_name,
        group=f"{system}-{condition}-{variant}",
        tags=[
            f"system:{system}",
            f"variant:{variant}",
            f"seed:{seed}",
            f"condition:{condition}",
        ],
        job_type="train",
    )

    # Phase-2-only metrics start at different global steps per seed because
    # Phase 1 may early-stop at different epochs. Re-axis them on a per-phase
    # counter so wandb aggregations align across seeds.
    # Why: wandb's `step_metric` only takes effect for keys explicitly listed.
    # How to apply: any new Phase-2-only key must be registered here as well.
    wandb.define_metric("phase_step")
    wandb.define_metric("val/phase_step")
    wandb.define_metric("inequality_violation", step_metric="phase_step")
    wandb.define_metric("val/inequality_violation", step_metric="val/phase_step")


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """Log scalar metrics to the active wandb run and optionally to a local JSONL file."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)
    if _local_metrics_path is not None:
        with open(_local_metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **metrics}) + "\n")


def log_artifact(path: str, name: str, type: str = "model") -> None:
    """Log a file artifact to wandb."""
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(path)
    wandb.log_artifact(artifact)


def finish_logging() -> None:
    """Close the active wandb run."""
    wandb.finish()
