"""Entry point for MaDE / baseline training on the inD dataset (Experiment 2).

Usage
-----
  # Full training (Phase 1 + Phase 2) from JSON config:
  python scripts/ind/train_made.py --config configs/ind/made.json

  # Smoke-train on stub data (no real inD dataset required):
  python scripts/ind/train_made.py --smoke --steps 10 --output-dir outputs/ind/smoke

  # Override config fields:
  python scripts/ind/train_made.py --config configs/ind/made.json \\
      --override training.num_epochs_phase1=2 \\
      --data-dir data/inD-preprocessed/v1

Dispatch
--------
The discrete ``variant`` field on :class:`ExperimentConfig` selects the
training path. Legal values are
``{made_phase1, made_phase2, made_no_residual, made_no_corrector,
mlp, fab, clamp}`` — see ``_LEGAL_VARIANTS``.
Unknown variants raise ``ValueError`` from ``_train_variant``.

Engineering rules:
  - jax_enable_x64 enabled before any JAX import.
  - eqx.filter_jit for all JIT-compiled functions.
  - Heun() + ConstantStepSize() for ODE integration.
  - Batch divisibility asserted before training.
  - All loss components logged separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from typing import Any
from pathlib import Path

import jax

from made.utils.jax_setup import configure

configure()

ROOT = Path(__file__).resolve().parents[2]

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from made.data.ind_data import (  # noqa: E402
    IND_LOCATION_ID_INDEX,
    IND_METADATA_DIM,
    IND_NUM_LOCATIONS,
    create_ind_data_source,
)
from made.data.grain_pipeline import (  # noqa: E402
    InMemoryDataSource,
    MetadataAttachTransform,
    TrajectoryWindowTransform,
    create_data_loader,
)
from made.evaluation.metrics import (  # noqa: E402
    EmpiricalEnvelope,
    estimate_empirical_envelope,
    kinematic_bicycle_inverse_controls,
)
from made.models import MaDEModel  # noqa: E402
from made.physics import (  # noqa: E402
    KinematicBicycleFieldData,
    _assert_xy_unbounded,
    drop_position_bounds,
    inD_physical_constraints,
    kinematic_bicycle_constraints,
)
from made.training import train  # noqa: E402
from made.utils import CheckpointManager, ExperimentConfig, save_config, set_local_metrics_path  # noqa: E402
from made.utils.config import (  # noqa: E402
    _LEGAL_VARIANTS,
    load_config,
    override_config,
)


# State / control dim for the kinematic bicycle (the only physics model used by inD).
_KB_STATE_DIM: int = 4
_KB_CONTROL_DIM: int = 2


def _build_data_loaders(
    data_dir: str,
    config: ExperimentConfig,
    *,
    use_stub: bool,
    num_devices: int,
    seed: int,
    stationary_filter_m: float | None = None,
) -> tuple[list, list, Any, Any]:
    """Build train and val data loaders for inD.

    Returns (train_loader, val_loader, train_states, train_lengths).
    train_states / train_lengths are the raw trajectory arrays needed for
    empirical-envelope estimation; they are NOT passed to the JAX training loop.
    """
    from made.data.window_dataset import filter_stationary_tracks

    window_transform = TrajectoryWindowTransform(window_size=2)
    attach_transform = MetadataAttachTransform()

    def _make_samples(states, metadata, lengths):
        samples = []

        for idx in range(states.shape[0]):
            traj = {
                "states": states[idx],
                "metadata": metadata[idx],
                "length": int(lengths[idx]),
                "params": jnp.array([], dtype=jnp.float64),
            }
            windows = window_transform.map(traj)
            samples.extend(attach_transform.map({**traj, "windows": windows}))
        return samples

    train_states, train_meta, train_lengths = create_ind_data_source(
        data_dir,
        "train",
        use_stub=use_stub,
        stub_num_trajectories=64,
        stub_trajectory_length=20,
        stub_seed=seed,
    )
    val_states, val_meta, val_lengths = create_ind_data_source(
        data_dir,
        "val",
        use_stub=use_stub,
        stub_num_trajectories=16,
        stub_trajectory_length=20,
        stub_seed=seed + 1,
    )

    # "Filtered" for MaDE training means TRACK-level -- drop a track whose end-to-end
    # displacement is at most the threshold, keep every transition pair of a surviving
    # track. Applied to train AND val, because the validation loss is what early
    # stopping reads and a split protocol between the two would make that signal meaningless.
    # Default None reproduces published behaviour exactly.
    if stationary_filter_m is not None:
        n_tr0, n_va0 = int(train_states.shape[0]), int(val_states.shape[0])
        train_states, train_lengths, train_meta, _ = filter_stationary_tracks(
            train_states, train_lengths, train_meta,
            min_displacement_m=stationary_filter_m)
        val_states, val_lengths, val_meta, _ = filter_stationary_tracks(
            val_states, val_lengths, val_meta,
            min_displacement_m=stationary_filter_m)
        print(
            f"[train_made_inD] track-level stationary filter at "
            f"{stationary_filter_m} m: train {n_tr0} -> {int(train_states.shape[0])} tracks, "
            f"val {n_va0} -> {int(val_states.shape[0])} tracks",
            flush=True,
        )

    train_samples = _make_samples(train_states, train_meta, train_lengths)
    val_samples = _make_samples(val_states, val_meta, val_lengths)
    print(
        f"[train_made_inD] transition pairs: train {len(train_samples)}, "
        f"val {len(val_samples)}",
        flush=True,
    )

    train_source = InMemoryDataSource(train_samples)
    val_source = InMemoryDataSource(val_samples)

    train_loader = create_data_loader(
        train_source,
        config.training.batch_size,
        num_devices=num_devices,
        shuffle=True,
        seed=seed,
        drop_remainder=True,
        noise_scale=config.data.noise_scale,
    )
    val_loader = create_data_loader(
        val_source,
        config.training.batch_size,
        num_devices=num_devices,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        noise_scale=0.0,
    )
    return train_loader, val_loader, train_states, train_lengths


# ---------------------------------------------------------------------------
# Empirical-envelope helpers.
# ---------------------------------------------------------------------------


def _compute_or_load_envelope(
    train_states: Any,
    train_lengths: Any,
    cache_path: Path,
    dt: float,
) -> EmpiricalEnvelope:
    """Return an EmpiricalEnvelope estimated from the inD training split.

    On the first call the envelope is computed from ``train_states`` /
    ``train_lengths`` and written to ``cache_path`` as JSON.  Subsequent calls
    load from the cache, avoiding redundant quantile computation across seeds.
    """
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return EmpiricalEnvelope(
            state_min=jnp.asarray(data["state_min"], dtype=jnp.float64),
            state_max=jnp.asarray(data["state_max"], dtype=jnp.float64),
            control_min=jnp.asarray(data["control_min"], dtype=jnp.float64),
            control_max=jnp.asarray(data["control_max"], dtype=jnp.float64),
        )
    controls, _ = kinematic_bicycle_inverse_controls(train_states, dt)
    envelope = estimate_empirical_envelope(train_states, controls, lengths=train_lengths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({
            "state_min": jnp.asarray(envelope.state_min).tolist(),
            "state_max": jnp.asarray(envelope.state_max).tolist(),
            "control_min": jnp.asarray(envelope.control_min).tolist(),
            "control_max": jnp.asarray(envelope.control_max).tolist(),
        }),
        encoding="utf-8",
    )
    return envelope


# ---------------------------------------------------------------------------
# Per-variant training adapters.
# ---------------------------------------------------------------------------


_MADE_VARIANT_NAMES = frozenset({"made_phase1", "made_phase2", "made_no_residual", "made_no_corrector"})


def _train_made(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: Path,
    seed: int,
    *,
    envelope: EmpiricalEnvelope | None = None,
) -> tuple[object, int]:
    """Train a MaDE cell. Phase 1, Phase 2, no_residual, and no_corrector all share this path.

    Phase-2 honours ``config.training.pretrained_phase1_path`` (already implemented
    in ``made.training.train``). The two ablations are configured purely via
    ``config.model.residual`` and ``config.corrector.mode`` — no code branch
    needed beyond passing them through.

    When ``envelope`` is provided (inD real-world data), the empirical-envelope
    constraint replaces ``kinematic_bicycle_constraints()`` so Phase-2 inequality
    penalties are calibrated to the actual coordinate range of the dataset.
    """
    config = replace(config, training=replace(config.training, is_field_data=True))
    os.environ.setdefault("MADE_DEBUG_ASSERT_XY", "1")
    # Field data: the stationary fallback split applies here.
    physics = KinematicBicycleFieldData()
    constraints = (
        inD_physical_constraints()
        if envelope is not None
        else drop_position_bounds(kinematic_bicycle_constraints())
    )
    _assert_xy_unbounded(constraints)
    model = MaDEModel.from_config(
        physics,
        constraints,
        config.model,
        config.corrector,
        key=jax.random.key(seed),
    )

    checkpoint_dir = str(output_dir / "checkpoints")
    ckpt_kwargs: dict = {}
    if config.training.checkpoint_save_interval is not None:
        ckpt_kwargs["save_interval"] = config.training.checkpoint_save_interval
    cm = CheckpointManager(checkpoint_dir, **ckpt_kwargs)
    trained = train(model, train_loader, val_loader, config, cm, constraints_factory=lambda: inD_physical_constraints())
    num_steps = config.training.steps_per_epoch or 0
    return trained, num_steps


def _train_fab(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: Path,
    seed: int,
) -> tuple[object, int]:
    """FAB latent-projection baseline."""
    from made.baselines.fab_baseline import FABBaseline, save_fab_checkpoint, train_fab_baseline

    init_key, train_key = jax.random.split(jax.random.key(seed + 1001), 2)
    model = FABBaseline(
        state_dim=_KB_STATE_DIM,
        key=init_key,
    )

    trained = train_fab_baseline(
        model,
        train_loader,
        val_loader,
        config.fab_baseline,
        key=train_key,
    )

    ckpt_dir = output_dir / "checkpoints"
    save_fab_checkpoint(trained, str(ckpt_dir))
    return trained, config.fab_baseline.steps_per_epoch or 0


def _train_mlp(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: Path,
    seed: int,
) -> tuple[object, int]:
    """Per-step MLP baseline."""
    from made.baselines.mlp_baseline import MLPBaseline, train_mlp_baseline

    init_key, train_key = jax.random.split(jax.random.key(seed + 2001), 2)
    model = MLPBaseline(
        state_dim=_KB_STATE_DIM,
        metadata_dim=config.model.metadata_dim,
        num_locations=config.model.num_locations,
        embedding_dim=config.model.embedding_dim,
        location_id_index=config.model.location_id_index,
        key=init_key,
    )

    trained = train_mlp_baseline(
        model,
        train_loader,
        val_loader,
        config.mlp_baseline,
        key=train_key,
    )

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(ckpt_dir / "mlp_model.pkl"), trained)
    return trained, config.mlp_baseline.steps_per_epoch or 0


def _train_clamp(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: Path,
    seed: int,
) -> tuple[None, int]:
    """Clamp baseline is parameterless; write a marker so the cell sentinel exists."""
    del config, train_loader, val_loader, seed
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / ".clamp").write_text("clamp baseline marker\n")
    return None, 0


# ---------------------------------------------------------------------------
# Finite-loss sentinel.
# ---------------------------------------------------------------------------


def _final_loss_for(variant: str, trained_model, val_loader) -> float:
    """Re-evaluate one val batch post-training to capture a final loss snapshot.

    Cheap (~0.5 s; one batch through one filter_jit'd loss). Convention-matched
    return of 0.0 for the parameterless clamp variant.
    """
    if variant == "clamp" or trained_model is None:
        return 0.0

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return float("nan")

    if variant in ("made_phase1", "made_phase2", "made_no_residual", "made_no_corrector"):
        # Cheapest forward pass through the corrected MaDE cell that produces
        # a real loss number — vmap over the batch and MSE against x_curr.
        x_prev, x_curr = batch["x_prev"], batch["x_curr"]
        metadata = batch.get("metadata")

        @eqx.filter_jit
        def _eval(model, x_prev, x_curr, metadata):
            def _per_sample(xp, xc, md):
                x_corr, _u = model(xp, xc, params=None, dt=0.2, metadata=md)
                return jnp.mean((x_corr - xc) ** 2)

            if metadata is None:
                return jnp.mean(jax.vmap(_per_sample, in_axes=(0, 0, None))(x_prev, x_curr, None))
            return jnp.mean(jax.vmap(_per_sample)(x_prev, x_curr, metadata))

        return float(_eval(trained_model, x_prev, x_curr, metadata))
    if variant == "fab":
        from made.baselines.fab_baseline import _fab_reconstruction_loss

        return float(eqx.filter_jit(_fab_reconstruction_loss)(trained_model, batch))
    if variant == "mlp":
        from made.baselines.mlp_baseline import _mlp_batch_loss

        return float(eqx.filter_jit(_mlp_batch_loss)(trained_model, batch))
    return float("nan")


def _early_stopping_summary(output_dir: Path) -> dict:
    """Per-phase `epochs_run`, `best_epoch` and `termination_reason`, read back from the run.

    Reconstructs a compact per-phase early-stopping summary (`epochs_run`,
    `best_epoch_in_phase`, `best_step`, `best_loss`, `termination_reason`) so the
    outcome of each training phase is visible without re-deriving it from configs
    and checkpoint metadata.

    Read from the last checkpoint's `train_meta.json` rather than threaded through the training
    loop, so the summary cannot disagree with the checkpoint it describes.
    """
    ck = output_dir / "checkpoints"
    if not ck.is_dir():
        return {"early_stopping": None, "note": "no checkpoints directory"}
    numeric = sorted((d for d in ck.iterdir() if d.is_dir() and d.name.isdigit()),
                     key=lambda p: int(p.name))
    per_phase: dict[str, dict] = {}
    reasons: list[str] = []
    for d in numeric:
        meta = d / "train_meta.json"
        if not meta.exists():
            continue
        try:
            tm = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in tm.get("early_stop_reasons") or []:
            if r not in reasons:
                reasons.append(r)
        for pk, st in (tm.get("early_stopping") or {}).items():
            hist = st.get("history") or []
            prev = per_phase.get(pk)
            if prev is None or len(hist) >= prev["epochs_run"]:
                bs = st.get("best_step")
                best_epoch = next(
                    (i for i, h in enumerate(hist) if h.get("step") == bs), None)
                per_phase[pk] = {
                    "epochs_run": len(hist),
                    "best_epoch_in_phase": best_epoch,
                    "best_step": bs,
                    "best_loss": st.get("best_loss"),
                }
    for pk, v in per_phase.items():
        v["termination_reason"] = next(
            (r for r in reasons if r.startswith(f"phase{pk}:")), "ceiling_or_not_recorded")
    return {"per_phase": per_phase, "all_reasons": reasons}


def _write_training_summary(
    output_dir: Path,
    variant: str,
    trained_model,
    val_loader,
    num_steps: int,
) -> None:
    """Write ``training_summary.json``.

    NaN-from-step-1 runs fail because ``final_loss`` is non-finite; downstream
    smoke tests assert ``math.isfinite(summary["final_loss"])``.

    Includes the per-phase early-stopping block (`epochs_run`, `best_epoch_in_phase`,
    `termination_reason`) so how each training phase ended is visible directly from
    this artefact.
    """
    final_loss = _final_loss_for(variant, trained_model, val_loader)
    summary = {
        "variant": variant,
        "final_loss": final_loss,
        "num_steps": int(num_steps),
        "final_loss_is_finite": bool(math.isfinite(final_loss)),
        **_early_stopping_summary(output_dir),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Variant dispatch.
# ---------------------------------------------------------------------------


_VARIANT_TRAINERS = {
    "made_phase1": _train_made,
    "made_phase2": _train_made,
    "made_no_residual": _train_made,
    "made_no_corrector": _train_made,
    "fab": _train_fab,
    "mlp": _train_mlp,
    "clamp": _train_clamp,
}


def _train_variant(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    output_dir: Path,
    seed: int,
    *,
    envelope: EmpiricalEnvelope | None = None,
) -> None:
    """Dispatch to the per-variant trainer based on ``config.variant``.

    Writes the per-cell ``training_summary.json`` artefact regardless of branch,
    so every variant has a uniform sentinel to inspect.
    """
    variant = config.variant
    if variant not in _LEGAL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant!r}; legal: {sorted(_LEGAL_VARIANTS)}"
        )

    trainer = _VARIANT_TRAINERS[variant]
    set_local_metrics_path(str(output_dir / "metrics_log.jsonl"))
    try:
        if variant in _MADE_VARIANT_NAMES and envelope is not None:
            trained, num_steps = trainer(config, train_loader, val_loader, output_dir, seed, envelope=envelope)
        else:
            trained, num_steps = trainer(config, train_loader, val_loader, output_dir, seed)
    finally:
        set_local_metrics_path(None)

    _write_training_summary(output_dir, variant, trained, val_loader, num_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaDE / baseline on inD (E02).")
    parser.add_argument("--config", default=None, help="Path to JSON experiment config.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "inD-preprocessed" / "v1"),
        help="Preprocessed inD root.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output_dir in config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use stub data (no real dataset).")
    parser.add_argument("--steps", type=int, default=None, help="Cap steps_per_epoch for smoke runs.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Dot-separated config overrides, e.g. training.num_epochs_phase1=2",
    )
    parser.add_argument(
        "--stationary-filter-m",
        type=float,
        default=None,
        help=(
            "Stationary filter threshold in metres. None (default) reproduces published "
            "behaviour byte-for-byte. 0.5 is the criterion used for the paper's results."
        ),
    )
    args = parser.parse_args()

    # --- Load or build config ---
    if args.config is not None:
        config = load_config(args.config)
    else:
        # Default inD config — MaDE Phase 2 path.
        config = ExperimentConfig(
            experiment_name="inD_made",
            output_dir=args.output_dir or "outputs/ind/made",
            variant="made_phase2",
        )
        config = replace(
            config,
            model=replace(
                config.model,
                use_metadata_encoder=True,
                metadata_dim=IND_METADATA_DIM,
                num_locations=IND_NUM_LOCATIONS,
                embedding_dim=8,
                location_id_index=IND_LOCATION_ID_INDEX,
            ),
            physics=replace(config.physics, dt=0.2, true_system="kinematic_bicycle"),
            training=replace(
                config.training,
                solver_train="heun",
                solver_eval="heun",
                num_epochs_phase1=30,
                num_epochs_phase2=30,
            ),
        )

    # Apply CLI overrides
    for ov in args.override:
        key, _, val_str = ov.partition("=")
        # Try numeric parse
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        config = override_config(config, {key: val})

    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.steps is not None:
        config = replace(
            config,
            training=replace(config.training, steps_per_epoch=args.steps),
        )

    # Smoke mode: shrink batch size so --steps N produces meaningful variety.
    if args.smoke:
        config = replace(
            config,
            training=replace(config.training, batch_size=32),
        )

    num_devices = max(jax.device_count(), 1)
    # Orbax's tensorstore backend requires absolute paths; resolve so that
    # `--output-dir outputs/...` from the runbook still produces a working
    # checkpoint write.
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.json"))

    print(f"[train_made_inD] variant={config.variant}  output_dir={output_dir}", file=sys.stderr)
    print(f"[train_made_inD] smoke={args.smoke}  num_devices={num_devices}", file=sys.stderr)
    print(f"[train_made_inD] metadata_dim={config.model.metadata_dim}  "
          f"num_locations={config.model.num_locations}", file=sys.stderr)

    # --- Build data loaders ---
    train_loader, val_loader, train_states, train_lengths = _build_data_loaders(
        args.data_dir,
        config,
        use_stub=args.smoke,
        num_devices=num_devices,
        seed=args.seed,
        stationary_filter_m=args.stationary_filter_m,
    )

    # Cap steps_per_epoch to available batches so --steps N never exceeds the loader length.
    if args.steps is not None:
        n_samples = len(train_loader.samples)
        available = max(n_samples // train_loader.batch_size, 1)
        capped = min(args.steps, available)
        if capped != config.training.steps_per_epoch:
            config = replace(
                config,
                training=replace(config.training, steps_per_epoch=capped),
            )

    # --- Compute empirical envelope from train split (cached across seeds) ---
    # Shared one directory above the per-variant output so all seeds reuse it.
    envelope_cache = output_dir.parent / "train_envelope.json"
    envelope = _compute_or_load_envelope(train_states, train_lengths, envelope_cache, config.physics.dt)
    print(
        f"[train_made_inD] envelope state_min={jnp.asarray(envelope.state_min).tolist()} "
        f"state_max={jnp.asarray(envelope.state_max).tolist()}",
        file=sys.stderr,
    )

    # --- Dispatch to per-variant trainer ---
    _train_variant(config, train_loader, val_loader, output_dir, args.seed, envelope=envelope)

    print(f"[train_made_inD] training complete (variant={config.variant})", file=sys.stderr)
    print(f"[train_made_inD] output_dir: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
