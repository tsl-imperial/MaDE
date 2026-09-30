"""Entry point for single-agent training."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from made.utils.jax_setup import configure

configure()

ROOT = Path(__file__).resolve().parents[2]

from made.baselines import (  # noqa: E402
    FABBaseline,
    MLPBaseline,
    train_fab_baseline,
    train_mlp_baseline,
)
from made.data import create_data_loader, create_data_source  # noqa: E402
from made.models import MaDECell  # noqa: E402
from made.physics import CompositeConstraints, build_system, build_system_for_model, resolve_params  # noqa: E402
from made.training import train  # noqa: E402
from made.utils import (  # noqa: E402
    CheckpointManager,
    ExperimentConfig,
    TrainState,
    init_logging,
    init_keys,
    save_config,
    set_local_metrics_path,
)

_MADE_VARIANTS = {
    "made",
    "made-no-residual",
    "made-no-corrector",
    "made-supervised-i",
    "made-fixed-i",
    "made-prior-only",
}


def _extract_constraint_bounds(constraints) -> dict:
    """Extract box-constraint bounds as serialisable lists.

    Handles CompositeConstraints by delegating to its first (BoxConstraints) layer.
    """
    box = constraints.constraints[0] if isinstance(constraints, CompositeConstraints) else constraints
    return {
        "state_min": np.asarray(box.state_min).tolist(),
        "state_max": np.asarray(box.state_max).tolist(),
        "control_min": np.asarray(box.control_min).tolist(),
        "control_max": np.asarray(box.control_max).tolist(),
    }


def _save_trained_model(model, checkpoint_dir: str, key: jax.Array, step: int = 1) -> None:
    CheckpointManager(checkpoint_dir).save(
        TrainState(
            model=model,
            opt_state_I=None,
            opt_state_T=None,
            key=key,
            step=step,
        ),
        step,
    )


def _known_param_overrides(cfg: ExperimentConfig) -> dict[str, float]:
    return cfg.model.known_params if cfg.model.known_system else cfg.physics.true_params


def _apply_known_params_to_source(source, known_params: jax.Array) -> None:
    for sample in source.samples:
        sample["params"] = known_params


def main_programmatic(
    cfg: ExperimentConfig,
    output_dir: str,
    data_dir: str = "data/generated",
    data_parallel_devices: int | None = None,
) -> dict:
    """Run training (or clamp marker creation) in-process and return result dict.

    Args:
        cfg: Fully resolved experiment configuration. ``cfg.experiment_name`` is used as
             the variant name. ``cfg.physics.true_system`` drives the data-loading path;
             ``cfg.model.known_system`` (falling back to ``cfg.physics.true_system``)
             drives ``MaDECell`` construction.
        output_dir: Directory under which artefacts are written.
        data_dir: Root directory for pre-generated datasets; true-system subdirectory is
                  appended automatically.
        data_parallel_devices: Number of devices for data-parallel training. None or 1
                               means single-device training.

    Returns:
        {"checkpoint_dir": str | None, "variant": str, "final_metrics": dict}.
        For clamp: checkpoint_dir is None, final_metrics is {}.
    """
    true_system_name = cfg.physics.true_system
    known_system_name = cfg.model.known_system or true_system_name
    variant_name = cfg.experiment_name

    true_physics, true_constraints = build_system(true_system_name)
    known_physics, known_constraints = build_system_for_model(true_system_name, known_system_name)

    known_param_overrides = _known_param_overrides(cfg)

    # Resolve parameter arrays and validate overrides eagerly.
    resolve_params(true_system_name, cfg.physics.true_params)
    known_params = resolve_params(known_system_name, known_param_overrides)

    data_path = f"{data_dir}/{true_system_name}"
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ clamp --
    if variant_name == "clamp":
        clamp_out = out_dir / "clamp_config.json"
        clamp_payload = {
            "system": true_system_name,
            "constraint_bounds": _extract_constraint_bounds(true_constraints),
        }
        clamp_out.write_text(json.dumps(clamp_payload, indent=2) + "\n", encoding="utf-8")
        return {"checkpoint_dir": None, "variant": "clamp", "final_metrics": {}}

    # -------------------------------------------------------------- seed / keys --
    seed = cfg.training.seed
    keys = init_keys(seed)

    # ---------------------------------------------------- data-parallel setup --
    n_dp = data_parallel_devices
    num_devices = n_dp if (n_dp is not None and n_dp > 1) else 1
    mesh = None
    if n_dp is not None and n_dp > 1:
        assert len(jax.local_devices()) >= n_dp, (
            f"Requested {n_dp} devices but only {len(jax.local_devices())} available."
        )
        mesh = jax.make_mesh((n_dp,), ("batch",))

    source = create_data_source(data_path, "train", cfg.data)
    val_source = create_data_source(data_path, "val", cfg.data)
    _apply_known_params_to_source(source, known_params)
    _apply_known_params_to_source(val_source, known_params)
    loader = create_data_loader(
        source,
        cfg.training.batch_size,
        num_devices=num_devices,
        seed=seed,
        noise_scale=cfg.data.noise_scale,
    )
    val_loader = create_data_loader(
        val_source,
        cfg.training.batch_size,
        num_devices=num_devices,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        noise_scale=cfg.data.noise_scale,
    )
    init_logging(cfg)

    # --------------------------------------------------------------- MaDE cell --
    if variant_name in _MADE_VARIANTS:
        model_cfg = cfg.model
        corrector_cfg = cfg.corrector
        training_cfg = cfg.training
        if variant_name == "made-no-residual":
            model_cfg = replace(model_cfg, residual="zero")
        elif variant_name == "made-no-corrector":
            corrector_cfg = replace(corrector_cfg, mode="disabled")
        elif variant_name == "made-supervised-i":
            training_cfg = replace(training_cfg, inverse_training="supervised_pretrain")
        elif variant_name == "made-fixed-i":
            model_cfg = replace(model_cfg, use_inverse_residual=False)
        train_cfg = replace(cfg, model=model_cfg, corrector=corrector_cfg, training=training_cfg)
        cell = MaDECell.from_config(
            known_physics,
            known_constraints,
            train_cfg.model,
            train_cfg.corrector,
            key=keys["init"],
            dt=train_cfg.physics.dt,
        )
        checkpoint_dir = str(out_dir / "checkpoints")
        ckpt_kwargs: dict = {}
        if train_cfg.training.checkpoint_save_interval is not None:
            ckpt_kwargs["save_interval"] = train_cfg.training.checkpoint_save_interval
        checkpoint_manager = CheckpointManager(checkpoint_dir, **ckpt_kwargs)
        set_local_metrics_path(str(out_dir / "metrics_log.jsonl"))
        trained = train(cell, loader, val_loader, train_cfg, checkpoint_manager, mesh=mesh)
        set_local_metrics_path(None)
        save_config(train_cfg, str(out_dir / "config.json"))
        return {"checkpoint_dir": checkpoint_dir, "variant": variant_name, "final_metrics": {}}

    # Inherit steps_per_epoch from the MaDE training config when the baseline doesn't set
    # it explicitly, so all methods sample the same number of transitions per step.
    # num_epochs is taken from each baseline's own config, decoupled from MaDE's phase count.
    _baseline_spe = cfg.training.steps_per_epoch  # None -> use all available batches

    # ------------------------------------------------------------ MLP baseline --
    if variant_name == "mlp":
        model = MLPBaseline(true_physics.state_dim, key=keys["init"])
        mlp_cfg = replace(
            cfg.mlp_baseline,
            steps_per_epoch=cfg.mlp_baseline.steps_per_epoch
            if cfg.mlp_baseline.steps_per_epoch is not None
            else _baseline_spe,
        )
        trained = train_mlp_baseline(model, loader, val_loader, mlp_cfg, key=keys["training"])
        checkpoint_dir = str(out_dir / "checkpoints")
        _save_trained_model(trained, checkpoint_dir, keys["training"])
        save_config(replace(cfg, mlp_baseline=mlp_cfg), str(out_dir / "config.json"))
        return {"checkpoint_dir": checkpoint_dir, "variant": variant_name, "final_metrics": {}}

    # ------------------------------------------------------------ FAB baseline --
    if variant_name == "fab":
        model = FABBaseline(true_physics.state_dim, key=keys["init"])
        fab_cfg = replace(
            cfg.fab_baseline,
            steps_per_epoch=cfg.fab_baseline.steps_per_epoch
            if cfg.fab_baseline.steps_per_epoch is not None
            else _baseline_spe,
        )
        trained = train_fab_baseline(model, loader, val_loader, fab_cfg, key=keys["training"])
        checkpoint_dir = str(out_dir / "checkpoints")
        _save_trained_model(trained, checkpoint_dir, keys["training"])
        save_config(replace(cfg, fab_baseline=fab_cfg), str(out_dir / "config.json"))
        return {"checkpoint_dir": checkpoint_dir, "variant": variant_name, "final_metrics": {}}

    raise ValueError(f"Unsupported variant '{variant_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent MaDE training.")
    parser.add_argument("--system", default=None, help="True physics system name.")
    parser.add_argument("--variant", default="made", help="Model variant to train.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "single_agent"))
    parser.add_argument(
        "--known-system",
        default=None,
        help="Physics system assumed known to the model (overrides model.known_system).",
    )
    parser.add_argument(
        "--true-params-json",
        default=None,
        help="JSON dict overriding physics.true_params, e.g. '{\"L\": 2.7}'.",
    )
    parser.add_argument(
        "--known-params-json",
        default=None,
        help="JSON dict overriding model.known_params, e.g. '{\"L\": 2.7}'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print resolved configuration and exit without training.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable MADE_PROFILE=1 timing output (train-step, validation, checkpoint wall-clock).",
    )
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=None,
        help="Number of devices for data-parallel training. Also reads DATA_PARALLEL_DEVICES env var as fallback.",
    )
    parser.add_argument(
        "--metric-log-interval-steps",
        type=int,
        default=None,
        help="Only sync/log train metrics every N steps; default config value preserves per-step logging.",
    )
    args = parser.parse_args()

    if args.profile:
        os.environ["MADE_PROFILE"] = "1"

    n_dp = args.data_parallel_devices
    if n_dp is None:
        _env = os.environ.get("DATA_PARALLEL_DEVICES")
        if _env:
            n_dp = int(_env)

    # Start from default config.
    experiment = ExperimentConfig(output_dir=args.output_dir)

    variant_name = args.variant

    # Use variant_name as experiment_name so main_programmatic can read it.
    experiment = replace(experiment, experiment_name=variant_name)

    # ----------------------------------------- apply --system / --known-system --
    if args.system is not None:
        experiment = replace(
            experiment,
            physics=replace(experiment.physics, true_system=args.system),
        )

    if args.known_system is not None:
        experiment = replace(
            experiment,
            model=replace(experiment.model, known_system=args.known_system),
        )

    # -------------------------------------------- apply JSON param overrides --
    if args.true_params_json is not None:
        true_params_override = json.loads(args.true_params_json)
        experiment = replace(
            experiment,
            physics=replace(experiment.physics, true_params=true_params_override),
        )

    if args.known_params_json is not None:
        known_params_override = json.loads(args.known_params_json)
        experiment = replace(
            experiment,
            model=replace(experiment.model, known_params=known_params_override),
        )

    # ----------------------------------------- apply seed from CLI --
    experiment = replace(
        experiment,
        training=replace(experiment.training, seed=args.seed),
    )
    if args.metric_log_interval_steps is not None:
        experiment = replace(
            experiment,
            training=replace(
                experiment.training,
                metric_log_interval_steps=args.metric_log_interval_steps,
            ),
        )

    true_system_name = experiment.physics.true_system
    known_system_name = experiment.model.known_system or true_system_name
    data_path = f"{args.data_dir}/{true_system_name}"
    out_dir = f"{args.output_dir}/{true_system_name}/{variant_name}"

    # ---------------------------------------------------------------- dry-run --
    if args.dry_run:
        print("DRY RUN:")
        print(f"  true_system: {true_system_name}")
        print(f"  known_system: {known_system_name}")
        print(f"  variant: {variant_name}")
        print(f"  data_path: {data_path}")
        print(f"  output_dir: {out_dir}")
        return

    main_programmatic(experiment, out_dir, data_dir=args.data_dir, data_parallel_devices=n_dp)


if __name__ == "__main__":
    main()
