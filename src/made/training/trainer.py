"""Training loop and step functions for MaDE."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding, PartitionSpec
from tqdm.auto import tqdm

from made.models import MaDECell, MaDEModel
from made.training.losses import phase1_loss, phase2_loss, targeted_phase_loss
from made.training.sampling import sample_controls
from made.utils import CheckpointManager, ExperimentConfig, TrainState, TrainingConfig, log_metrics
from made.utils.config import DataConfig


MaDELike = MaDECell | MaDEModel
_SENTINEL = object()

# Fold-in constants keep train/val Phase-2 proposal-corruption PRNG streams disjoint from
# each other and from the batch-shuffle roots in train() (fold_in(key(seed), 0/1)). Never
# renumber: part of the reproducibility contract.
_PROPOSAL_KEY_STREAM_TRAIN: int = 1001
_PROPOSAL_KEY_STREAM_VAL: int = 1002

# Sub-streams folded into the per-(step, batch_index) proposal key for per-sample
# randomisation draws, distinct from the base key (which carries the Gaussian noise draw so
# `sampling == "fixed"` stays byte-identical to pre-randomisation runs). Never renumber.
_PROPOSAL_SUBSTREAM_SCALE: int = 1
_PROPOSAL_SUBSTREAM_CLEAN_MASK: int = 2

# Defined once at module level so the JIT cache persists across validation epochs.
_jit_val_phase1_loss = eqx.filter_jit(phase1_loss)
_jit_val_phase2_loss = eqx.filter_jit(phase2_loss)


def _rewrite_phase1_seed_meta(meta_path: Path, pretrained_path: str) -> bool:
    """Convert a Phase-1 source's train_meta.json into a Phase-2 curriculum-seed meta.

    Idempotent: no-op (returns False) if the meta already reflects a curriculum-seed
    state (signature.pretrained_phase1_path == str(pretrained_path) and
    resume.phase != 1). Otherwise applies four invariants and rewrites the meta.
    Returns True iff the rewrite was applied.

    ``_maybe_copy_pretrained_checkpoint`` rewrites a freshly-copied meta atomically, so a
    legitimate curriculum cell advances past ``resume.phase=1`` immediately and never
    legitimately reaches ``phase=1`` again. Observing ``pretrained_phase1_path`` matching but
    ``resume.phase=1`` therefore only happens on the stale-partial-seed migration path this
    function targets, so any ``phase=1`` meta is treated as a rewrite candidate.
    """
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    sig = meta.get("comparability_signature")
    stored_path = sig.get("pretrained_phase1_path") if isinstance(sig, dict) else None
    resume_block = meta.get("resume")
    resume_phase = resume_block.get("phase") if isinstance(resume_block, dict) else None
    already_curriculum = (stored_path == str(pretrained_path)) and (resume_phase != 1)
    if already_curriculum:
        return False

    # (a) Fix comparability_signature.pretrained_phase1_path.
    if "comparability_signature" in meta:
        meta["comparability_signature"]["pretrained_phase1_path"] = str(pretrained_path)
        # Phase-2 proposal corruption is a Phase-2-only knob, so the Phase-1 seed run
        # legitimately carries different values. Drop these keys so
        # _check_train_meta's overlap-intersection skips them; the curriculum cell writes
        # its own values on its first _write_train_meta call.
        for _p2_key in (
            "phase2_proposal_perturbation_type",
            "phase2_proposal_perturbation_scale",
            "phase2_proposal_perturbation_scale_per_dim",
            "phase2_proposal_perturbation_scale_sampling",
            "phase2_proposal_perturbation_scale_min_ratio",
            "phase2_proposal_perturbation_zero_fraction",
        ):
            meta["comparability_signature"].pop(_p2_key, None)
    # (b) Advance resume cursor to Phase 2, epoch 0.
    meta["resume"] = {
        "phase": 2,
        "epoch_in_phase": 0,
        "batch_offset": 0,
        "validation_completed": False,
    }
    # (c) Reset Phase-2 early-stopping state; preserve Phase-1 record.
    _es_default: dict[str, object] = {
        "best_loss": math.inf,
        "best_step": None,
        "wait": 0,
        "history": [],
        "reasons": [],
    }
    existing_es = meta.get("early_stopping")
    if not isinstance(existing_es, dict):
        existing_es = {}
    existing_es["2"] = _es_default
    meta["early_stopping"] = existing_es
    # (d) Reset active-phase early-stopping summary fields.
    meta["early_stop_reasons"] = []
    meta["early_stop_best_loss"] = math.inf
    meta["early_stop_best_step"] = None

    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return True


def _self_heal_curriculum_seed_meta(
    checkpoint_manager: CheckpointManager,
    pretrained_phase1_path: str | None,
) -> None:
    """Rewrite a stale Phase-1 source meta on disk into a curriculum-seed meta so
    _check_train_meta accepts the resume. Idempotent — no-op once the meta is already
    in the curriculum-seed state, on a non-curriculum config (pretrained_phase1_path is
    None), or on an empty checkpoints dir.
    """
    if pretrained_phase1_path is None:
        return
    ckpt_dir = checkpoint_manager.directory
    if not ckpt_dir.exists():
        return
    step_subdirs = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_subdirs:
        return
    highest = max(step_subdirs, key=lambda p: int(p.name))
    _rewrite_phase1_seed_meta(highest / "train_meta.json", pretrained_phase1_path)


def _maybe_copy_pretrained_checkpoint(
    pretrained_path: str | None,
    local_dir: str,
) -> None:
    """Seed the local checkpoint dir from a previous run's directory.

    Selects the highest-step subdirectory in the source whose ``train_meta.json`` records
    ``resume.phase == 1``, and copies that step dir (plus non-numeric top-level items)
    into ``local_dir``. Phase-2 step dirs are skipped: a standard MaDE run trains both
    phases contiguously in one ``checkpoints/`` tree, so the literal highest step is
    typically Phase-2 even when a Phase-1-end checkpoint is present.

    After copying, the destination's ``train_meta.json`` is rewritten via
    ``_rewrite_phase1_seed_meta`` so ``comparability_signature.pretrained_phase1_path``
    matches ``pretrained_path``, the resume cursor advances to ``phase=2,
    epoch_in_phase=0``, and Phase-2 early-stopping state resets.

    No-op when ``local_dir`` already has a numeric step subdir with ``state.pkl``
    (preserves partial runs). Raises ``FileNotFoundError`` if ``pretrained_path`` does not
    exist; ``ValueError`` if the source has no numeric step subdirs, the highest-step dir
    is missing ``train_meta.json``, or no step subdir records ``resume.phase == 1``.
    """
    if pretrained_path is None:
        return
    src = Path(pretrained_path)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"pretrained_phase1_path={pretrained_path!r} does not exist or is not a directory"
        )

    numeric_subdirs = sorted(
        [item for item in src.iterdir() if item.is_dir() and item.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not numeric_subdirs:
        raise ValueError(
            f"pretrained_phase1_path={pretrained_path!r} exists but contains no numeric step "
            "subdirectories. Run the source regime through Phase 1 before using it as a "
            "pretrained_phase1_path."
        )

    descending = list(reversed(numeric_subdirs))
    highest_meta_path = descending[0] / "train_meta.json"
    if not highest_meta_path.exists():
        raise ValueError(
            f"pretrained_phase1_path={pretrained_path!r}: highest step directory "
            f"{descending[0].name!r} is missing train_meta.json. The source checkpoint "
            "directory appears corrupted."
        )

    # Highest-numbered step subdir whose train_meta.json records resume.phase==1.
    # Phase-1 dirs are not pruned, so earlier numeric subdirs hold the Phase-1-end
    # checkpoint the curriculum needs even though the literal highest step is Phase 2.
    phase1_step_dir: Path | None = None
    highest_phase: int | None = None
    for candidate in descending:
        meta_path = candidate / "train_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        candidate_phase = meta.get("resume", {}).get("phase")
        if highest_phase is None:
            highest_phase = candidate_phase
        if candidate_phase == 1:
            phase1_step_dir = candidate
            break

    if phase1_step_dir is None:
        raise ValueError(
            f"pretrained_phase1_path={pretrained_path!r} contains no Phase-1 checkpoint "
            f"(highest step records phase={highest_phase}). "
            "Run the source regime through Phase 1 before using it as a "
            "pretrained_phase1_path."
        )

    local = Path(local_dir)
    # "has any existing step checkpoint" = there is a numeric subdir with state.pkl
    has_local_ckpt = local.exists() and any(
        item.is_dir() and item.name.isdigit() and (item / "state.pkl").exists()
        for item in local.iterdir()
    )
    if has_local_ckpt:
        return
    local.mkdir(parents=True, exist_ok=True)
    # Copy only the chosen Phase-1 step dir: other numeric step dirs would mislead
    # _load_resume_cursor into resuming Phase 2.
    dest = local / phase1_step_dir.name
    if not dest.exists():
        shutil.copytree(phase1_step_dir, dest)
    for item in src.iterdir():
        if item.is_dir() and item.name.isdigit():
            continue
        target = local / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    _rewrite_phase1_seed_meta(dest / "train_meta.json", str(pretrained_path))


class _CompiledTrainState(eqx.Module):
    """JIT-facing train state without Python-static step/phase metadata.

    ``TrainState.step`` and ``TrainState.phase`` stay host-owned checkpoint
    metadata. Passing them through ``eqx.filter_jit`` makes them static leaves,
    which can trigger a fresh compile for every global step.
    """

    model: MaDELike
    opt_state_I: optax.OptState
    opt_state_T: optax.OptState
    key: jax.Array


@dataclass
class _EarlyStoppingState:
    """Mutable Python-side state for validation-driven phase stopping."""

    best_loss: float = float("inf")
    best_step: int | None = None
    wait: int = 0
    # Epoch the best was found in, so the burn-in guard tests the BEST epoch rather than
    # the current one. `None` on a checkpoint written before this field existed; the guard
    # treats that as "not known to be inside burn-in" so a legacy resume behaves as before.
    best_epoch_in_phase: int | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "best_epoch_in_phase": self.best_epoch_in_phase,
            "wait": self.wait,
            "history": self.history,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_EarlyStoppingState":
        return cls(
            best_loss=float(payload.get("best_loss", float("inf"))),
            best_step=payload.get("best_step"),
            best_epoch_in_phase=(None if payload.get("best_epoch_in_phase") is None
                                 else int(payload["best_epoch_in_phase"])),
            wait=int(payload.get("wait", 0)),
            history=[
                {key: float(value) for key, value in metrics.items()}
                for metrics in payload.get("history", [])
            ],
            reasons=[str(reason) for reason in payload.get("reasons", [])],
        )


def _cell(model: MaDELike) -> MaDECell:
    return model.cell if isinstance(model, MaDEModel) else model


def _replace_cell(model: MaDELike, cell: MaDECell) -> MaDELike:
    if isinstance(model, MaDEModel):
        return eqx.tree_at(lambda made_model: made_model.cell, model, cell)
    return cell


def _to_compiled_state(state: TrainState) -> _CompiledTrainState:
    return _CompiledTrainState(
        model=state.model,
        opt_state_I=state.opt_state_I,
        opt_state_T=state.opt_state_T,
        key=state.key,
    )


def _to_train_state(compiled: _CompiledTrainState, *, step: int, phase: int) -> TrainState:
    return TrainState(
        model=compiled.model,
        opt_state_I=compiled.opt_state_I,
        opt_state_T=compiled.opt_state_T,
        key=compiled.key,
        step=int(step),
        phase=int(phase),
    )


def _t_trainable(model: MaDELike):
    cell = _cell(model)
    if isinstance(model, MaDEModel) and model.encoder is not None:
        return (cell.augmented_dynamics.residual, model.encoder)
    return (cell.augmented_dynamics.residual,)


def _replace_t_trainable(model: MaDELike, updated) -> MaDELike:
    cell = _cell(model)
    residual_updated = updated[0]
    augmented_updated = eqx.tree_at(
        lambda augmented: augmented.residual,
        cell.augmented_dynamics,
        residual_updated,
    )
    cell_updated = eqx.tree_at(lambda made_cell: made_cell.augmented_dynamics, cell, augmented_updated)
    model_updated = _replace_cell(model, cell_updated)
    if isinstance(model_updated, MaDEModel) and model_updated.encoder is not None:
        model_updated = eqx.tree_at(lambda made_model: made_model.encoder, model_updated, updated[1])
    return model_updated


def _make_lr_schedule(lr: float, warmup_steps: int) -> optax.ScalarOrSchedule:
    """Return a constant lr or a linear warmup schedule."""
    if warmup_steps > 0:
        return optax.linear_schedule(
            init_value=0.0, end_value=lr, transition_steps=warmup_steps
        )
    return lr


def _make_optimizers(
    config: TrainingConfig,
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Build I-side and T-side optimizers with optional clipping, zero_nans, warmup.

    optax.zero_nans is chained BEFORE clip_by_global_norm and adam: a single
    NaN/Inf gradient leaf would otherwise propagate through clip
    (NaN/NaN = NaN) and poison weights for the rest of training. zero_nans
    has empty state (no opt_state shape impact); zero_nans_enabled is
    recorded in train_meta.json for provenance only (not strict-equality).
    """
    lr_I = _make_lr_schedule(config.lr_I, config.warmup_steps)
    lr_T = _make_lr_schedule(config.lr_T, config.warmup_steps)

    def _build(lr, clip_norm):
        if clip_norm is not None:
            return optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(clip_norm),
                optax.adam(lr),
            )
        return optax.chain(optax.zero_nans(), optax.adam(lr))

    opt_I = _build(lr_I, config.i_side_grad_clip_norm)
    opt_T = _build(lr_T, config.t_side_grad_clip_norm)
    return opt_I, opt_T


def create_train_state(cell: MaDELike, config, key: jax.Array) -> TrainState:
    """Initialise the train state and optimiser slots."""
    opt_I, opt_T = _make_optimizers(config)
    inverse_params = eqx.filter(_cell(cell).inverse_dynamics, eqx.is_array)
    residual_params = eqx.filter(_t_trainable(cell), eqx.is_array)
    return TrainState(
        model=cell,
        opt_state_I=opt_I.init(inverse_params),
        opt_state_T=opt_T.init(residual_params),
        key=key,
        step=0,
        phase=1,
    )


def _migrate_opt_state_for_zero_nans(
    opt_state, new_opt: optax.GradientTransformation, params
):
    """Adapt a checkpoint's opt_state to the current optimizer chain.

    The chain gained a leading `optax.zero_nans()` transform, so pre-existing checkpoints
    save an opt_state tuple one element shorter than the current chain expects, and
    `chain.update` on a mismatched-length tuple raises. Detects the mismatch and prepends
    a fresh `zero_nans` state, preserving saved Adam moments and clip-norm state so resume
    stays momentum-faithful.
    """
    expected_state = new_opt.init(params)
    if not isinstance(expected_state, tuple) or not isinstance(opt_state, tuple):
        return opt_state
    if len(opt_state) == len(expected_state):
        return opt_state
    if len(opt_state) == len(expected_state) - 1:
        return (expected_state[0],) + opt_state
    return expected_state


def _batch_arrays(
    batch: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    if isinstance(batch, dict):
        return (
            batch["x_prev"],
            batch["x_curr"],
            batch.get("params"),
            batch.get("u_gt"),
            batch.get("metadata"),
        )
    x_prev, x_curr, params, *rest = batch
    u_gt = rest[0] if rest else None
    return x_prev, x_curr, params, u_gt, None


def _batch_size(batch: Any) -> int:
    x_prev, _, _, _, _ = _batch_arrays(batch)
    return int(x_prev.shape[0])


def _resolve_params(model: MaDELike, params: jax.Array | None, metadata: jax.Array | None) -> jax.Array:
    if isinstance(model, MaDEModel):
        return model.resolve_params(params, metadata)
    if params is None:
        raise ValueError("Batches for MaDECell training must include params.")
    return params


def _proposal_perturbation_key(seed: int, stream: int, *counters: int) -> jax.Array:
    """Derive the PRNG key for one Phase-2 proposal corruption draw.

    Depends only on the run's training seed, a stream constant
    (`_PROPOSAL_KEY_STREAM_TRAIN` / `_PROPOSAL_KEY_STREAM_VAL`) and host-side counters
    (global step, and batch index for validation) — never `TrainState.key` — so a
    checkpoint resume replays a bit-identical corruption sequence.
    """
    key = jax.random.fold_in(jax.random.key(seed), stream)
    for counter in counters:
        key = jax.random.fold_in(key, counter)
    return key


def _compute_x_proposal(
    x_curr: jax.Array,
    phase: int,
    training_config: TrainingConfig,
    cell: MaDECell,
    key: jax.Array | None = None,
) -> jax.Array:
    """Compute the proposal input x̃_t for the I-module.

    Returns x_curr unchanged unless in Phase 2 with a non-trivial
    `phase2_proposal_perturbation_type`. Two families:

    * ``bound_violation`` — deterministic, no PRNG: each component pushed outward from
      the box midpoint. ``key`` ignored.
    * ``gaussian`` — additive zero-mean Gaussian noise; ``key`` required.

    Magnitude source: if ``phase2_proposal_perturbation_scale_per_dim`` is ``None``
    (default), the scalar ``phase2_proposal_perturbation_scale`` is a fraction of the box
    range (`scale * (state_max - state_min)`); ±inf range entries (unbounded dims like x,y
    on inD/field-data) give zero perturbation there. If the per-dimension tuple is set, it
    overrides the scalar with absolute magnitudes in state units (sigma for gaussian, push
    distance for bound_violation; 0.0 means never perturbed); no box range is consulted, so
    infinite-bound dims can be corrupted.

    On the gaussian path the resolved magnitude is the upper-bound sigma_max per
    dimension. If ``phase2_proposal_perturbation_scale_sampling`` is not ``"fixed"``
    and/or ``phase2_proposal_perturbation_zero_fraction`` is nonzero, the realised sigma
    also varies per batch element and dimension (see `TrainingConfig`). sigma_max == 0.0
    stays exactly unperturbed under every mode; a clean-drawn sample is bit-identical on
    every dimension. Both knobs are gaussian-only; ``bound_violation`` raises if either is
    set.
    """
    if phase != 2:
        return x_curr
    p_type = training_config.phase2_proposal_perturbation_type
    if p_type == "none":
        return x_curr
    if p_type not in ("bound_violation", "gaussian"):
        raise ValueError(f"Unknown phase2_proposal_perturbation_type: {p_type!r}")

    per_dim = training_config.phase2_proposal_perturbation_scale_per_dim
    if per_dim is not None:
        state_dim = int(x_curr.shape[-1])
        if len(per_dim) != state_dim:
            raise ValueError(
                "phase2_proposal_perturbation_scale_per_dim length "
                f"{len(per_dim)} does not match state dim {state_dim}"
            )
        magnitude = jnp.asarray(per_dim, dtype=x_curr.dtype)
    else:
        constraints = cell.constraints
        # ±inf range entries give zero perturbation there; use per-dim absolute scales
        # for nonzero perturbation on unbounded dims.
        range_ = constraints.state_max - constraints.state_min
        safe_range = jnp.where(jnp.isfinite(range_), range_, 0.0)
        magnitude = training_config.phase2_proposal_perturbation_scale * safe_range

    sampling = training_config.phase2_proposal_perturbation_scale_sampling
    min_ratio = training_config.phase2_proposal_perturbation_scale_min_ratio
    zero_fraction = training_config.phase2_proposal_perturbation_zero_fraction
    randomised = sampling != "fixed" or zero_fraction != 0.0

    if p_type == "gaussian":
        if key is None:
            raise ValueError(
                "phase2_proposal_perturbation_type='gaussian' requires a PRNG key; "
                "_compute_x_proposal was called with key=None."
            )
        # Drawn directly from `key`, before any fold_in, so the base noise stream stays
        # byte-identical when randomisation is off. Per-sample draws use disjoint
        # fold_in sub-streams.
        noise = jax.random.normal(key, x_curr.shape, dtype=x_curr.dtype)
        if not randomised:
            return x_curr + magnitude * noise

        if sampling == "fixed":
            multiplier = jnp.ones_like(noise)
        else:
            scale_key = jax.random.fold_in(key, _PROPOSAL_SUBSTREAM_SCALE)
            if sampling == "uniform":
                multiplier = jax.random.uniform(
                    scale_key,
                    x_curr.shape,
                    dtype=x_curr.dtype,
                    minval=min_ratio,
                    maxval=1.0,
                )
            elif sampling == "loguniform":
                # log m ~ U(log min_ratio, 0); min_ratio > 0 enforced by
                # TrainingConfig.__post_init__.
                log_multiplier = jax.random.uniform(
                    scale_key,
                    x_curr.shape,
                    dtype=x_curr.dtype,
                    minval=math.log(min_ratio),
                    maxval=0.0,
                )
                multiplier = jnp.exp(log_multiplier)
            else:
                raise ValueError(
                    f"Unknown phase2_proposal_perturbation_scale_sampling: {sampling!r}"
                )
        # sigma_max == 0.0 keeps this product exactly 0.0 under every sampling mode.
        perturbed = x_curr + (magnitude * multiplier) * noise
        if zero_fraction == 0.0:
            return perturbed

        # Whole-sample clean mask (leading axes = batch): a clean sample is untouched
        # on every dimension.
        mask_key = jax.random.fold_in(key, _PROPOSAL_SUBSTREAM_CLEAN_MASK)
        keep_clean = jax.random.bernoulli(
            mask_key, zero_fraction, x_curr.shape[:-1] + (1,)
        )
        return jnp.where(keep_clean, x_curr, perturbed)

    if randomised:
        # Per-sample randomisation knobs are gaussian-only; ignoring them here would
        # silently hand a caller deterministic corruption despite the config requesting
        # randomisation.
        raise ValueError(
            "phase2_proposal_perturbation_scale_sampling / "
            "phase2_proposal_perturbation_zero_fraction apply only to "
            "phase2_proposal_perturbation_type='gaussian'; got type="
            f"{p_type!r}, sampling={sampling!r}, zero_fraction={zero_fraction!r}."
        )

    constraints = cell.constraints
    midpoint = 0.5 * (constraints.state_min + constraints.state_max)
    if per_dim is not None:
        # +/-inf bounds make midpoint NaN, and `x >= NaN` is False, pushing every sample
        # in the -1 direction. Fall back to 0.0 so "outward" is well defined on unbounded
        # dims. Confined to the per-dim branch: the scalar path's magnitude is already
        # 0.0 there, so it stays byte-identical.
        midpoint = jnp.where(jnp.isfinite(midpoint), midpoint, 0.0)
    sign = jnp.where(x_curr >= midpoint, 1.0, -1.0)
    return x_curr + sign * magnitude


def _sample_batch_controls(
    model: MaDELike,
    batch: Any,
    config,
    key: jax.Array,
    step: int | jax.Array,
    total_steps: int,
) -> tuple[jax.Array, jax.Array, bool]:
    x_prev, x_curr, params, u_gt, metadata = _batch_arrays(batch)
    params = _resolve_params(model, params, metadata)
    if config.inverse_training == "supervised_pretrain":
        if u_gt is None:
            raise ValueError("inverse_training='supervised_pretrain' requires batch['u_gt'].")
        return u_gt, key, True
    if config.inverse_training != "cycle":
        raise ValueError("inverse_training must be 'cycle' or 'supervised_pretrain'.")

    keys = jax.random.split(key, x_prev.shape[0] + 1)
    next_key = keys[0]
    sample_keys = keys[1:]
    controls = jax.vmap(
        lambda xp, xc, p, k: sample_controls(
            config.control_sampling,
            _cell(model).constraints,
            _cell(model).inverse_dynamics,
            xp,
            xc,
            p,
            step,
            total_steps,
            k,
        )
    )(x_prev, x_curr, params, sample_keys)
    return controls, next_key, False


def _metrics_to_float(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _phase_metric_context(*, phase: int, phase_step: int) -> dict[str, float]:
    """Return phase-local axis fields for metrics whose global-step origin varies."""
    return {
        "phase": float(phase),
        "phase_step": float(phase_step),
    }


def _validation_metric_context(*, phase: int, phase_step: int) -> dict[str, float]:
    context = _phase_metric_context(phase=phase, phase_step=phase_step)
    return {f"val/{key}": value for key, value in context.items()}


def _effective_steps_per_epoch(
    requested: int | None,
    available_batches: int,
    *,
    name: str,
) -> int:
    n_batches = max(available_batches, 1)
    if requested is None:
        return n_batches
    if requested > n_batches:
        raise ValueError(f"{name}={requested} exceeds available batches={n_batches}")
    return requested


def _epoch_batches(
    batches: list[Any],
    steps_per_epoch: int,
    key_root: jax.Array,
    epoch_index: int,
    *,
    capped: bool,
) -> list[Any]:
    if not batches or not capped:
        return batches
    perm = jax.random.permutation(jax.random.fold_in(key_root, epoch_index), max(len(batches), 1))
    indices = np.asarray(perm[:steps_per_epoch])
    return [batches[int(index)] for index in indices]


def _write_train_meta(
    cm: CheckpointManager,
    step: int,
    cfg: TrainingConfig,
    extra: dict[str, Any] | None = None,
    dp_overrides: dict | None = None,
    data_cfg: DataConfig | None = None,
) -> None:
    _sig: dict = {
        "validation_interval_epochs": cfg.validation_interval_epochs,
        "checkpoint_save_interval": cfg.checkpoint_save_interval,
        "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        "pretrained_phase1_path": cfg.pretrained_phase1_path,
        "phase2_proposal_perturbation_type": cfg.phase2_proposal_perturbation_type,
        "phase2_proposal_perturbation_scale": cfg.phase2_proposal_perturbation_scale,
        "phase2_proposal_perturbation_scale_per_dim": (
            None
            if cfg.phase2_proposal_perturbation_scale_per_dim is None
            else list(cfg.phase2_proposal_perturbation_scale_per_dim)
        ),
        "phase2_proposal_perturbation_scale_sampling": (
            cfg.phase2_proposal_perturbation_scale_sampling
        ),
        "phase2_proposal_perturbation_scale_min_ratio": (
            cfg.phase2_proposal_perturbation_scale_min_ratio
        ),
        "phase2_proposal_perturbation_zero_fraction": (
            cfg.phase2_proposal_perturbation_zero_fraction
        ),
        "isolate_ineq_gradient": cfg.isolate_ineq_gradient,
    }
    if data_cfg is not None:
        _sig["noise_scale"] = data_cfg.noise_scale
    if dp_overrides is not None:
        _sig.update(dp_overrides)
    meta = {
        "steps_per_epoch": cfg.steps_per_epoch,
        "val_steps_per_epoch": cfg.val_steps_per_epoch,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": cfg.i_side_grad_clip_norm,
        "warmup_steps": cfg.warmup_steps,
        "zero_nans_enabled": cfg.zero_nans_enabled,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": _sig,
    }
    if extra is not None:
        meta.update(extra)
    meta_path = Path(cm.directory) / str(step) / "train_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _check_train_meta(
    cm: CheckpointManager,
    step: int,
    cfg: TrainingConfig,
    num_devices: int = 1,
    data_cfg: DataConfig | None = None,
) -> None:
    meta = _read_train_meta(cm, step)
    if meta is None:
        return
    if meta.get("steps_per_epoch") != cfg.steps_per_epoch:
        raise ValueError(
            f"Checkpoint steps_per_epoch={meta.get('steps_per_epoch')} "
            f"!= config steps_per_epoch={cfg.steps_per_epoch}. "
            "steps_per_epoch must be stable across resumes."
        )
    if meta.get("val_steps_per_epoch") != cfg.val_steps_per_epoch:
        raise ValueError(
            f"Checkpoint val_steps_per_epoch={meta.get('val_steps_per_epoch')} "
            f"!= config val_steps_per_epoch={cfg.val_steps_per_epoch}. "
            "val_steps_per_epoch must be stable across resumes."
        )
    if (
        meta.get("t_side_grad_clip_norm", _SENTINEL) is not _SENTINEL
        and meta.get("t_side_grad_clip_norm") != cfg.t_side_grad_clip_norm
    ):
        raise ValueError(
            f"Checkpoint t_side_grad_clip_norm={meta.get('t_side_grad_clip_norm')} "
            f"!= config t_side_grad_clip_norm={cfg.t_side_grad_clip_norm}. "
            "t_side_grad_clip_norm must be stable across resumes."
        )
    if (
        meta.get("i_side_grad_clip_norm", _SENTINEL) is not _SENTINEL
        and meta.get("i_side_grad_clip_norm") != cfg.i_side_grad_clip_norm
    ):
        raise ValueError(
            f"Checkpoint i_side_grad_clip_norm={meta.get('i_side_grad_clip_norm')} "
            f"!= config i_side_grad_clip_norm={cfg.i_side_grad_clip_norm}. "
            "i_side_grad_clip_norm must be stable across resumes."
        )
    if (
        meta.get("warmup_steps", _SENTINEL) is not _SENTINEL
        and meta.get("warmup_steps") != cfg.warmup_steps
    ):
        raise ValueError(
            f"Checkpoint warmup_steps={meta.get('warmup_steps')} "
            f"!= config warmup_steps={cfg.warmup_steps}. "
            "warmup_steps must be stable across resumes."
        )
    if meta.get("early_stopping_config") != _early_stopping_config_meta(cfg):
        raise ValueError("early stopping config must be stable across resumes.")
    stored_sig = meta.get("comparability_signature")
    current_sig: dict = {
        "validation_interval_epochs": cfg.validation_interval_epochs,
        "checkpoint_save_interval": cfg.checkpoint_save_interval,
        "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        "data_parallel_devices": num_devices,
        "pretrained_phase1_path": cfg.pretrained_phase1_path,
        "phase2_proposal_perturbation_type": cfg.phase2_proposal_perturbation_type,
        "phase2_proposal_perturbation_scale": cfg.phase2_proposal_perturbation_scale,
        "phase2_proposal_perturbation_scale_per_dim": (
            None
            if cfg.phase2_proposal_perturbation_scale_per_dim is None
            else list(cfg.phase2_proposal_perturbation_scale_per_dim)
        ),
        "phase2_proposal_perturbation_scale_sampling": (
            cfg.phase2_proposal_perturbation_scale_sampling
        ),
        "phase2_proposal_perturbation_scale_min_ratio": (
            cfg.phase2_proposal_perturbation_scale_min_ratio
        ),
        "phase2_proposal_perturbation_zero_fraction": (
            cfg.phase2_proposal_perturbation_zero_fraction
        ),
        "isolate_ineq_gradient": cfg.isolate_ineq_gradient,
    }
    if data_cfg is not None:
        current_sig["noise_scale"] = data_cfg.noise_scale
    if stored_sig is not None:
        overlap_keys = set(stored_sig.keys()) & set(current_sig.keys())
        for k in overlap_keys:
            if stored_sig[k] != current_sig[k]:
                raise ValueError(
                    f"comparability_signature mismatch on key '{k}': "
                    f"stored={stored_sig[k]!r}, current={current_sig[k]!r}. "
                    "Start a fresh run or match the original training config."
                )


def _read_train_meta(cm: CheckpointManager, step: int) -> dict[str, Any] | None:
    meta_path = Path(cm.directory) / str(step) / "train_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _early_stopping_config_meta(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "enabled": cfg.early_stopping_enabled,
        "min_epochs": cfg.early_stopping_min_epochs,
        "patience": cfg.early_stopping_patience,
        "min_delta": cfg.early_stopping_min_delta,
        "physical_exact": cfg.early_stopping_physical_exact,
        "physical_saturation": cfg.early_stopping_physical_saturation,
        "forward_tol": cfg.early_stopping_forward_tol,
        "inverse_tol": cfg.early_stopping_inverse_tol,
        "minimum_norm_tol": cfg.early_stopping_minimum_norm_tol,
        "delta_i_norm_tol": cfg.early_stopping_delta_i_norm_tol,
        "saturation_window": cfg.early_stopping_saturation_window,
        "saturation_delta": cfg.early_stopping_saturation_delta,
    }


def _progress_output() -> tuple[TextIO, bool, TextIO | None]:
    if sys.stderr.isatty():
        return sys.stderr, False, None
    try:
        tty = Path("/dev/tty").open("w", encoding="utf-8")
    except OSError:
        return sys.stderr, True, None
    return tty, False, tty


def _phase_epoch_progress(
    step: int,
    phase1_steps: int,
    train_steps_per_epoch: int,
) -> tuple[int, int, int]:
    """Return completed epochs and current-epoch batch offset from a step checkpoint."""
    if step < phase1_steps:
        return step // train_steps_per_epoch, 0, step % train_steps_per_epoch
    phase2_steps = step - phase1_steps
    return (
        phase1_steps // train_steps_per_epoch,
        phase2_steps // train_steps_per_epoch,
        phase2_steps % train_steps_per_epoch,
    )


def _weighted_mean_metric_dict(
    weighted_metric_dicts: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    if not weighted_metric_dicts:
        return {"loss": float("inf")}
    total_weight = sum(weight for weight, _ in weighted_metric_dicts)
    if total_weight <= 0:
        return {"loss": float("inf")}
    keys = weighted_metric_dicts[0][1].keys()
    return {
        key: float(
            sum(weight * metrics[key] for weight, metrics in weighted_metric_dicts)
            / total_weight
        )
        for key in keys
    }


def _metrics_below(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
    return all(metrics.get(name, float("inf")) <= threshold for name, threshold in thresholds.items())


def _physical_exact_reason(cfg: TrainingConfig, metrics: dict[str, float]) -> str | None:
    if not cfg.early_stopping_physical_exact:
        return None
    thresholds = {
        "forward_consistency": cfg.early_stopping_forward_tol,
        "inverse_consistency": cfg.early_stopping_inverse_tol,
        "minimum_norm": cfg.early_stopping_minimum_norm_tol,
        "delta_i_norm": cfg.early_stopping_delta_i_norm_tol,
    }
    if _metrics_below(metrics, thresholds):
        return "physical_exact"
    return None


def _physical_saturation_reason(
    cfg: TrainingConfig,
    history: list[dict[str, float]],
) -> str | None:
    if not cfg.early_stopping_physical_saturation:
        return None
    window_size = cfg.early_stopping_saturation_window
    if len(history) < window_size:
        return None
    window = history[-window_size:]
    near_zero = {
        "forward_consistency": cfg.early_stopping_forward_tol,
        "inverse_consistency": cfg.early_stopping_inverse_tol,
    }
    if not all(_metrics_below(metrics, near_zero) for metrics in window):
        return None
    stable_names = ("minimum_norm", "delta_i_norm")
    for name in stable_names:
        values = np.asarray([metrics.get(name, float("inf")) for metrics in window], dtype=float)
        if not np.all(np.isfinite(values)):
            return None
        if float(np.max(values) - np.min(values)) > cfg.early_stopping_saturation_delta:
            return None
    return "physical_saturation"


def _early_stop_reason(
    cfg: TrainingConfig,
    state: _EarlyStoppingState,
    metrics: dict[str, float],
    *,
    phase: int,
    epoch_in_phase: int,
) -> str | None:
    if not cfg.early_stopping_enabled:
        return None

    loss = metrics.get("loss", float("inf"))

    # No best is tracked inside the burn-in window: `best_loss`/`best_step`/
    # `best_epoch_in_phase` do not update while `epoch_in_phase + 1 <
    # early_stopping_min_epochs`, so a best inside burn-in can never exist and patience
    # counts only from the first post-window epoch. Deliberate cost: the saved checkpoint
    # is the best from end-of-burn-in onward, not the global best, so a run whose
    # validation genuinely peaks inside burn-in saves a worse model than it reached.
    # `wait` self-resets at the first post-window epoch (best_loss still infinite there),
    # so it needs no explicit reset.
    in_burn_in = epoch_in_phase + 1 < cfg.early_stopping_min_epochs
    if not in_burn_in:
        if loss < state.best_loss - cfg.early_stopping_min_delta:
            state.best_loss = loss
            state.best_step = int(metrics["step"])
            state.best_epoch_in_phase = epoch_in_phase
            state.wait = 0
        else:
            state.wait += 1

    # Indices stay equal to epoch_in_phase (appended every epoch, burn-in included), so
    # best_epoch can be recovered by indexing history against best_step.
    state.history.append(metrics)

    # Two burn-in guards: the current epoch (don't stop before the minimum), and the best
    # epoch (a best found inside the window is not a converged best, so patience counted
    # from it would stop a run that has barely started). Stopping never fires while the
    # best sits inside burn-in; patience restarts from the first best found outside the
    # window, or the run goes to its ceiling if none is ever found.
    if in_burn_in:
        return None

    # Assertion that never fires under the no-best-inside-burn-in rule above: if it did,
    # the burn-in logic is broken. `best_epoch_in_phase is None` (e.g. a legacy checkpoint)
    # is read as "not known to be inside burn-in" so a legacy resume behaves as before.
    if state.best_epoch_in_phase is not None and (
        state.best_epoch_in_phase + 1 < cfg.early_stopping_min_epochs
    ):
        return None

    reason = _physical_exact_reason(cfg, metrics)
    if reason is None:
        reason = _physical_saturation_reason(cfg, state.history)
    if reason is None and cfg.early_stopping_patience is not None:
        if state.wait >= cfg.early_stopping_patience:
            reason = "validation_patience"
    if reason is None:
        return None
    return f"phase{phase}:{reason}:step{int(metrics['step'])}"


def _early_stopping_meta(stopper: _EarlyStoppingState) -> dict[str, Any]:
    return {
        "early_stop_reasons": stopper.reasons,
        "early_stop_best_loss": stopper.best_loss,
        "early_stop_best_step": stopper.best_step,
        "early_stop_best_epoch_in_phase": stopper.best_epoch_in_phase,
        "early_stop_epochs_run_in_phase": len(stopper.history),
    }


def _phase_stoppers_meta(phase_stoppers: dict[int, _EarlyStoppingState]) -> dict[str, Any]:
    return {
        "early_stopping": {
            str(phase): stopper.to_dict() for phase, stopper in phase_stoppers.items()
        }
    }


def _load_phase_stoppers(cm: CheckpointManager, step: int) -> dict[int, _EarlyStoppingState]:
    meta = _read_train_meta(cm, step)
    if meta is None:
        return {1: _EarlyStoppingState(), 2: _EarlyStoppingState()}
    payload = meta.get("early_stopping", {})
    return {
        1: _EarlyStoppingState.from_dict(payload.get("1", {})),
        2: _EarlyStoppingState.from_dict(payload.get("2", {})),
    }


def _checkpoint_meta(
    phase_stoppers: dict[int, _EarlyStoppingState],
    *,
    active_phase: int,
    epoch_in_phase: int,
    batch_offset: int,
    validation_completed: bool,
) -> dict[str, Any]:
    all_reasons = phase_stoppers[1].reasons + phase_stoppers[2].reasons
    meta = _early_stopping_meta(phase_stoppers[active_phase])
    meta["early_stop_reasons"] = all_reasons
    meta.update(_phase_stoppers_meta(phase_stoppers))
    meta["resume"] = {
        "phase": active_phase,
        "epoch_in_phase": epoch_in_phase,
        "batch_offset": batch_offset,
        "validation_completed": validation_completed,
    }
    return meta


def _load_resume_cursor(
    cm: CheckpointManager,
    step: int,
    phase1_steps: int,
    train_steps_per_epoch: int,
) -> tuple[int, int, int, bool]:
    meta = _read_train_meta(cm, step)
    if meta is not None:
        resume = meta.get("resume")
        if resume is not None:
            return (
                int(resume.get("phase", 1)),
                int(resume.get("epoch_in_phase", 0)),
                int(resume.get("batch_offset", 0)),
                bool(resume.get("validation_completed", False)),
            )

    phase1_epochs, phase2_epochs, batch_offset = _phase_epoch_progress(
        step,
        phase1_steps,
        train_steps_per_epoch,
    )
    if step < phase1_steps:
        return 1, phase1_epochs, batch_offset, False
    return 2, phase2_epochs, batch_offset, False


@eqx.filter_jit
def _train_step_phase1_i(
    state: _CompiledTrainState,
    batch: Any,
    config,
    dt: float,
    total_steps: int,
    opt_I: optax.GradientTransformation,
    opt_T: optax.GradientTransformation,
    step_scalar: jax.Array,
    x_proposal: jax.Array,
) -> tuple[_CompiledTrainState, dict[str, jax.Array]]:
    return _train_step_impl(
        state, batch, config, dt, total_steps, opt_I, opt_T,
        phase=1, target="I", step=step_scalar, x_proposal=x_proposal,
    )


@eqx.filter_jit
def _train_step_phase1_t(
    state: _CompiledTrainState,
    batch: Any,
    config,
    dt: float,
    total_steps: int,
    opt_I: optax.GradientTransformation,
    opt_T: optax.GradientTransformation,
    step_scalar: jax.Array,
    x_proposal: jax.Array,
) -> tuple[_CompiledTrainState, dict[str, jax.Array]]:
    return _train_step_impl(
        state, batch, config, dt, total_steps, opt_I, opt_T,
        phase=1, target="T", step=step_scalar, x_proposal=x_proposal,
    )


@eqx.filter_jit
def _train_step_phase2_i(
    state: _CompiledTrainState,
    batch: Any,
    config,
    dt: float,
    total_steps: int,
    opt_I: optax.GradientTransformation,
    opt_T: optax.GradientTransformation,
    step_scalar: jax.Array,
    x_proposal: jax.Array,
) -> tuple[_CompiledTrainState, dict[str, jax.Array]]:
    return _train_step_impl(
        state, batch, config, dt, total_steps, opt_I, opt_T,
        phase=2, target="I", step=step_scalar, x_proposal=x_proposal,
    )


@eqx.filter_jit
def _train_step_phase2_t(
    state: _CompiledTrainState,
    batch: Any,
    config,
    dt: float,
    total_steps: int,
    opt_I: optax.GradientTransformation,
    opt_T: optax.GradientTransformation,
    step_scalar: jax.Array,
    x_proposal: jax.Array,
) -> tuple[_CompiledTrainState, dict[str, jax.Array]]:
    return _train_step_impl(
        state, batch, config, dt, total_steps, opt_I, opt_T,
        phase=2, target="T", step=step_scalar, x_proposal=x_proposal,
    )


def _train_step_impl(
    state: _CompiledTrainState,
    batch: Any,
    config,
    dt: float,
    total_steps: int,
    opt_I: optax.GradientTransformation,
    opt_T: optax.GradientTransformation,
    *,
    phase: int,
    target: str,
    step: jax.Array,
    x_proposal: jax.Array,
) -> tuple[_CompiledTrainState, dict[str, jax.Array]]:
    x_prev, x_curr, params, _, metadata = _batch_arrays(batch)
    params = _resolve_params(state.model, params, metadata)
    key, subkey = jax.random.split(state.key)
    u_sampled, key, used_supervised = _sample_batch_controls(
        state.model,
        batch,
        config,
        subkey,
        step,
        total_steps,
    )

    def _loss_fn(model: MaDELike) -> tuple[jax.Array, dict[str, jax.Array]]:
        return targeted_phase_loss(
            model,
            x_prev,
            x_curr,
            params,
            dt,
            u_sampled,
            config,
            phase=phase,
            target=target,
            x_proposal=x_proposal,
        )

    (_, metrics), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(state.model)
    metrics = dict(metrics)
    metrics["control_supervised"] = jnp.asarray(float(used_supervised))
    metrics["update_target_I"] = jnp.asarray(float(target == "I"))

    if target == "I":
        inverse_grads = _cell(grads).inverse_dynamics
        inverse_params = eqx.filter(_cell(state.model).inverse_dynamics, eqx.is_array)
        updates, opt_state_I = opt_I.update(inverse_grads, state.opt_state_I, inverse_params)
        inverse_updated = eqx.apply_updates(_cell(state.model).inverse_dynamics, updates)
        cell_updated = eqx.tree_at(
            lambda made_cell: made_cell.inverse_dynamics,
            _cell(state.model),
            inverse_updated,
        )
        new_model = _replace_cell(state.model, cell_updated)
        new_state = _CompiledTrainState(
            model=new_model,
            opt_state_I=opt_state_I,
            opt_state_T=state.opt_state_T,
            key=key,
        )
        return new_state, metrics

    residual_grads = eqx.filter(_t_trainable(grads), eqx.is_array)
    residual_params = eqx.filter(_t_trainable(state.model), eqx.is_array)
    updates, opt_state_T = opt_T.update(residual_grads, state.opt_state_T, residual_params)
    residual_updated = eqx.apply_updates(_t_trainable(state.model), updates)
    new_model = _replace_t_trainable(state.model, residual_updated)
    new_state = _CompiledTrainState(
        model=new_model,
        opt_state_I=state.opt_state_I,
        opt_state_T=opt_state_T,
        key=key,
    )
    return new_state, metrics


def _validation_epoch(
    state: TrainState,
    batches: list[Any],
    config: ExperimentConfig,
    total_steps: int,
    *,
    phase: int,
    phase_step: int,
) -> tuple[TrainState, dict[str, float]]:
    training_config = config.training
    epoch_metrics = []
    for batch_index, batch in enumerate(batches):
        batch_weight = _batch_size(batch)
        x_prev, x_curr, params, _, metadata = _batch_arrays(batch)
        params = _resolve_params(state.model, params, metadata)
        key, subkey = jax.random.split(state.key)
        u_sampled, key, used_supervised = _sample_batch_controls(
            state.model,
            batch,
            training_config,
            subkey,
            state.step,
            total_steps,
        )
        state = TrainState(
            model=state.model,
            opt_state_I=state.opt_state_I,
            opt_state_T=state.opt_state_T,
            key=key,
            step=state.step,
            phase=state.phase,
        )
        proposal_key = None
        if training_config.phase2_proposal_perturbation_type == "gaussian":
            proposal_key = _proposal_perturbation_key(
                training_config.seed,
                _PROPOSAL_KEY_STREAM_VAL,
                int(state.step),
                batch_index,
            )
        x_proposal = _compute_x_proposal(
            x_curr, phase, training_config, _cell(state.model), proposal_key
        )
        if phase == 1:
            val_loss, val_metrics = _jit_val_phase1_loss(
                state.model,
                x_prev,
                x_curr,
                params,
                config.physics.dt,
                u_sampled,
                training_config,
                x_proposal=x_proposal,
            )
        else:
            val_loss, val_metrics = _jit_val_phase2_loss(
                state.model,
                x_prev,
                x_curr,
                params,
                config.physics.dt,
                u_sampled,
                training_config,
                x_proposal=x_proposal,
            )
        val_metrics = dict(val_metrics)
        val_metrics["loss"] = val_loss
        val_metrics["control_supervised"] = jnp.asarray(float(used_supervised))
        metrics_float = _metrics_to_float(val_metrics)
        metrics_float["step"] = float(state.step)
        epoch_metrics.append((batch_weight, metrics_float))
        val_log_metrics = {f"val/{k}": v for k, v in metrics_float.items()}
        val_log_metrics.update(_validation_metric_context(phase=phase, phase_step=phase_step))
        log_metrics(val_log_metrics, step=state.step)
    return state, _weighted_mean_metric_dict(epoch_metrics)


def train(
    cell: MaDELike,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    *,
    mesh=None,
    constraints_factory: Callable[[], "BoxConstraints"] | None = None,
) -> MaDELike:
    """Train MaDE across Phase 1 and Phase 2.

    Args:
        constraints_factory: Optional zero-arg builder for a fresh BoxConstraints. When
            given, the cell's constraints are forced to constraints_factory() right after
            checkpoint restore (or fresh state creation) -- mandatory on the inD/field-data
            path to prevent stale finite-x,y bounds resurfacing via
            eqx.tree_deserialise_leaves. E01 callers omit this kwarg.
    """
    training_config = config.training
    num_devices = mesh.shape[0] if mesh is not None else 1
    dp_overrides = {
        "data_parallel_devices": num_devices,
        "mesh_shape": str(mesh.shape) if mesh is not None else None,
        "strict_comparable": True,
        "effective_global_batch_size": training_config.batch_size,
    }
    opt_I, opt_T = _make_optimizers(training_config)
    _maybe_copy_pretrained_checkpoint(
        training_config.pretrained_phase1_path,
        str(checkpoint_manager.directory),
    )
    _self_heal_curriculum_seed_meta(
        checkpoint_manager, training_config.pretrained_phase1_path
    )
    state = checkpoint_manager.restore()
    if (
        state is not None
        and constraints_factory is None
        and training_config.is_field_data
    ):
        raise ValueError(
            "Phase-2 resume detected on a field-data (inD) config but no "
            "constraints_factory was provided. eqx.tree_deserialise_leaves "
            "would resurrect stale x,y bounds from the checkpoint and silently "
            "re-trigger the divergence bug. Pass constraints_factory=lambda: "
            "inD_physical_constraints() from the inD script."
        )
    if state is None:
        state = create_train_state(cell, training_config, jax.random.key(training_config.seed))
    else:
        _check_train_meta(
            checkpoint_manager, state.step, training_config, num_devices, data_cfg=config.data
        )
        # Older checkpoints saved opt_state with a 2-transform chain (clip_by_global_norm
        # + adam); the current chain has 3 (zero_nans + clip + adam). Prepend a fresh
        # zero_nans state on resume.
        inverse_params = eqx.filter(_cell(state.model).inverse_dynamics, eqx.is_array)
        residual_params = eqx.filter(_t_trainable(state.model), eqx.is_array)
        migrated_opt_I = _migrate_opt_state_for_zero_nans(
            state.opt_state_I, opt_I, inverse_params
        )
        migrated_opt_T = _migrate_opt_state_for_zero_nans(
            state.opt_state_T, opt_T, residual_params
        )
        if migrated_opt_I is not state.opt_state_I or migrated_opt_T is not state.opt_state_T:
            state = TrainState(
                model=state.model,
                opt_state_I=migrated_opt_I,
                opt_state_T=migrated_opt_T,
                key=state.key,
                step=state.step,
                phase=state.phase,
            )
    if constraints_factory is not None:
        from made.physics import _assert_xy_unbounded
        new_constraints = constraints_factory()
        _assert_xy_unbounded(new_constraints)
        cell_obj = _cell(state.model)
        new_cell = eqx.tree_at(lambda c: c.constraints, cell_obj, new_constraints)
        if cell_obj is state.model:
            new_model = new_cell
        else:
            new_model = eqx.tree_at(lambda m: m.cell, state.model, new_cell)
        state = TrainState(
            model=new_model,
            opt_state_I=state.opt_state_I,
            opt_state_T=state.opt_state_T,
            key=state.key,
            step=state.step,
            phase=state.phase,
        )
    host_step_int = int(state.step)
    host_phase_int = int(state.phase)
    if mesh is not None:
        replication_sharding = NamedSharding(mesh, PartitionSpec())
        state = eqx.filter_shard(state, replication_sharding)
    compiled_state = _to_compiled_state(state)

    train_batches = list(train_loader)
    val_batches = list(val_loader)
    train_steps_per_epoch = _effective_steps_per_epoch(
        training_config.steps_per_epoch,
        len(train_batches),
        name="steps_per_epoch",
    )
    total_epochs = training_config.num_epochs_phase1 + training_config.num_epochs_phase2
    total_steps = int(max(total_epochs * train_steps_per_epoch, 1))
    phase1_steps = int(training_config.num_epochs_phase1 * train_steps_per_epoch)
    train_key_root = jax.random.fold_in(jax.random.key(training_config.seed), 0)
    val_key_root = jax.random.fold_in(jax.random.key(training_config.seed), 1)
    resume_phase, resume_epoch_in_phase, resumed_batch_offset, validation_completed = (
        _load_resume_cursor(
            checkpoint_manager,
            host_step_int,
            phase1_steps,
            train_steps_per_epoch,
        )
    )
    completed_phase1_epochs, completed_phase2_epochs, _ = _phase_epoch_progress(
        host_step_int,
        phase1_steps,
        train_steps_per_epoch,
    )
    phase_stoppers = _load_phase_stoppers(checkpoint_manager, host_step_int)

    _profile = os.environ.get("MADE_PROFILE", "") == "1"
    _t_train_total = 0.0
    _t_val_total = 0.0
    _t_ckpt_total = 0.0
    _n_train_steps = 0

    progress_file, progress_disabled, progress_owned_file = _progress_output()
    progress = tqdm(
        total=total_steps,
        initial=min(host_step_int, total_steps),
        desc=f"train {config.experiment_name}",
        unit="step",
        disable=progress_disabled,
        file=progress_file,
        dynamic_ncols=True,
    )
    last_epoch_in_phase = 0
    # What the phase-2 boundary did, so a reader need not infer it from a step number.
    # None when the option is off.
    _phase1_best_restored: dict | None = None
    try:
        phase_specs = (
            (1, training_config.num_epochs_phase1, 0),
            (2, training_config.num_epochs_phase2, training_config.num_epochs_phase1),
        )
        for phase, phase_epochs, epoch_offset in phase_specs:
            if phase < resume_phase:
                start_epoch = phase_epochs
            elif phase > resume_phase:
                start_epoch = 0
            else:
                start_epoch = resume_epoch_in_phase + int(validation_completed)
            if phase == 1 and resume_phase == 1:
                start_epoch = max(start_epoch, completed_phase1_epochs)
            if phase == 2 and resume_phase == 2:
                start_epoch = max(start_epoch, completed_phase2_epochs)

            # Phase 2 otherwise inherits `compiled_state` as left by phase 1 -- its FINAL
            # model. If early stopping picked an earlier epoch, phase 2 would start from a
            # model the run judged worse than its best; the only other restore runs after
            # this loop and applies to the final phase alone.
            #
            # Guarded on `start_epoch == 0` so this fires only when phase 2 genuinely
            # begins -- resuming a partially trained phase 2 must not rewind to phase 1's
            # best and silently discard completed phase-2 epochs.
            if (
                phase == 2
                and training_config.restore_phase1_best_before_phase2
                and start_epoch == 0
            ):
                best1 = phase_stoppers[1].best_step
                if best1 is None:
                    raise ValueError(
                        "restore_phase1_best_before_phase2 is set, but phase 1 recorded no "
                        "best step. That happens when early stopping was disabled or never "
                        "updated, so there is no 'phase-1 best' to restore and starting phase 2 "
                        "would silently use the final model instead. Set "
                        "early_stopping_enabled, or turn this option off."
                    )
                if best1 != host_step_int:
                    # Captured before the restore below reassigns `state`: `from_step` is the
                    # step phase 2 would have started from (phase 1's final), which
                    # distinguishes a restore from a no-op.
                    phase1_final_step = int(host_step_int)
                    restored = checkpoint_manager.restore(best1)
                    if restored is None:
                        raise ValueError(
                            f"restore_phase1_best_before_phase2 is set and phase 1's best step "
                            f"is {best1}, but no checkpoint for that step could be restored "
                            f"from {checkpoint_manager.directory}. Refusing to begin phase 2 "
                            f"from the phase-1 final model, which is what this option exists to "
                            f"prevent."
                        )
                    state = TrainState(
                        model=restored.model,
                        opt_state_I=restored.opt_state_I,
                        opt_state_T=restored.opt_state_T,
                        key=restored.key,
                        step=restored.step,
                        phase=2,
                    )
                    host_step_int = int(state.step)
                    host_phase_int = 2
                    compiled_state = _to_compiled_state(state)
                    _phase1_best_restored = {"restored": True, "best_step": int(best1),
                                             "from_step": phase1_final_step}
                else:
                    _phase1_best_restored = {"restored": False, "best_step": int(best1),
                                             "reason": "phase-1 best IS phase-1 final"}
            for epoch_in_phase in range(start_epoch, phase_epochs):
                last_epoch_in_phase = epoch_in_phase
                host_phase_int = phase
                state = _to_train_state(compiled_state, step=host_step_int, phase=host_phase_int)
                global_epoch = epoch_offset + epoch_in_phase
                epoch_train_batches = _epoch_batches(
                    train_batches,
                    train_steps_per_epoch,
                    train_key_root,
                    global_epoch,
                    capped=training_config.steps_per_epoch is not None,
                )
                batch_offset = 0
                if phase == resume_phase and epoch_in_phase == resume_epoch_in_phase and not validation_completed:
                    batch_offset = resumed_batch_offset
                epoch_train_batches = epoch_train_batches[batch_offset:]
                for batch_index, batch in enumerate(epoch_train_batches, start=batch_offset):
                    if _profile:
                        _t0 = time.perf_counter()
                    # Cross-device reduction order may differ; losses are float64-close but not bitwise identical.
                    if mesh is not None:
                        batch = jax.device_put(batch, NamedSharding(mesh, PartitionSpec("batch")))
                    update_target = (
                        "I"
                        if (host_step_int // training_config.alternation_period) % 2 == 0
                        else "T"
                    )
                    step_scalar = jnp.asarray(host_step_int)
                    x_prev_batch, x_curr_batch, _, _, _ = _batch_arrays(batch)
                    proposal_key = None
                    if training_config.phase2_proposal_perturbation_type == "gaussian":
                        proposal_key = _proposal_perturbation_key(
                            training_config.seed,
                            _PROPOSAL_KEY_STREAM_TRAIN,
                            host_step_int,
                        )
                    x_proposal_batch = _compute_x_proposal(
                        x_curr_batch,
                        phase,
                        training_config,
                        _cell(compiled_state.model),
                        proposal_key,
                    )
                    if phase == 1 and update_target == "I":
                        compiled_state, metrics = _train_step_phase1_i(
                            compiled_state,
                            batch,
                            training_config,
                            config.physics.dt,
                            total_steps,
                            opt_I,
                            opt_T,
                            step_scalar,
                            x_proposal_batch,
                        )
                    elif phase == 1:
                        compiled_state, metrics = _train_step_phase1_t(
                            compiled_state,
                            batch,
                            training_config,
                            config.physics.dt,
                            total_steps,
                            opt_I,
                            opt_T,
                            step_scalar,
                            x_proposal_batch,
                        )
                    elif update_target == "I":
                        compiled_state, metrics = _train_step_phase2_i(
                            compiled_state,
                            batch,
                            training_config,
                            config.physics.dt,
                            total_steps,
                            opt_I,
                            opt_T,
                            step_scalar,
                            x_proposal_batch,
                        )
                    else:
                        compiled_state, metrics = _train_step_phase2_t(
                            compiled_state,
                            batch,
                            training_config,
                            config.physics.dt,
                            total_steps,
                            opt_I,
                            opt_T,
                            step_scalar,
                            x_proposal_batch,
                        )
                    host_step_int += 1
                    host_phase_int = phase
                    state = _to_train_state(
                        compiled_state,
                        step=host_step_int,
                        phase=host_phase_int,
                    )
                    if _profile:
                        _t_train_total += time.perf_counter() - _t0
                        _n_train_steps += 1
                    should_log_train = (
                        host_step_int % training_config.metric_log_interval_steps == 0
                        or host_step_int == total_steps
                        or host_step_int % checkpoint_manager.save_interval == 0
                    )
                    if should_log_train:
                        metrics_float = _metrics_to_float(metrics)
                        phase_step = epoch_in_phase * train_steps_per_epoch + batch_index + 1
                        metrics_float.update(
                            _phase_metric_context(phase=phase, phase_step=phase_step)
                        )
                        log_metrics(metrics_float, step=host_step_int)
                    progress.set_postfix(
                        phase=phase,
                        target=update_target,
                        refresh=False,
                    )
                    progress.update(1)
                    if host_step_int % checkpoint_manager.save_interval == 0:
                        if _profile:
                            _t0 = time.perf_counter()
                        checkpoint_manager.save(state, host_step_int)
                        if _profile:
                            _t_ckpt_total += time.perf_counter() - _t0
                        _write_train_meta(
                            checkpoint_manager,
                            host_step_int,
                            training_config,
                            _checkpoint_meta(
                                phase_stoppers,
                                active_phase=phase,
                                epoch_in_phase=epoch_in_phase,
                                batch_offset=batch_index + 1,
                                validation_completed=False,
                            ),
                            dp_overrides=dp_overrides,
                            data_cfg=config.data,
                        )

                _should_validate = (
                    (epoch_in_phase + 1) % training_config.validation_interval_epochs == 0
                    or (epoch_in_phase + 1) == phase_epochs
                )
                if _should_validate:
                    epoch_val_batches = _epoch_batches(
                        val_batches,
                        len(val_batches),
                        val_key_root,
                        global_epoch,
                        capped=False,
                    )
                    if mesh is not None:
                        validation_sharding = NamedSharding(mesh, PartitionSpec("batch"))
                        epoch_val_batches = [
                            jax.device_put(b, validation_sharding)
                            if _batch_size(b) % num_devices == 0
                            else b
                            for b in epoch_val_batches
                        ]
                    if _profile:
                        _t0 = time.perf_counter()
                    state, val_metrics = _validation_epoch(
                        state,
                        epoch_val_batches,
                        config,
                        total_steps,
                        phase=phase,
                        phase_step=(epoch_in_phase + 1) * train_steps_per_epoch,
                    )
                    host_step_int = int(state.step)
                    host_phase_int = int(state.phase)
                    compiled_state = _to_compiled_state(state)
                    if _profile:
                        _t_val_total += time.perf_counter() - _t0
                    stopper = phase_stoppers[phase]
                    previous_best_loss = stopper.best_loss
                    stop_reason = _early_stop_reason(
                        training_config,
                        stopper,
                        val_metrics,
                        phase=phase,
                        epoch_in_phase=epoch_in_phase,
                    )
                    _should_save_best = (
                        (epoch_in_phase + 1) % training_config.best_checkpoint_interval_epochs == 0
                        or (epoch_in_phase + 1) == phase_epochs
                    )
                    if stopper.best_loss < previous_best_loss and _should_save_best:
                        if _profile:
                            _t0 = time.perf_counter()
                        checkpoint_manager.save(state, host_step_int)
                        if _profile:
                            _t_ckpt_total += time.perf_counter() - _t0
                        _write_train_meta(
                            checkpoint_manager,
                            host_step_int,
                            training_config,
                            _checkpoint_meta(
                                phase_stoppers,
                                active_phase=phase,
                                epoch_in_phase=epoch_in_phase,
                                batch_offset=train_steps_per_epoch,
                                validation_completed=True,
                            ),
                            dp_overrides=dp_overrides,
                            data_cfg=config.data,
                        )
                    if stop_reason is not None:
                        stopper.reasons.append(stop_reason)
                        log_metrics(
                            {
                                "early_stop/phase": float(phase),
                                "early_stop/step": float(host_step_int),
                                "early_stop/phase_step": float(
                                    (epoch_in_phase + 1) * train_steps_per_epoch
                                ),
                            },
                            step=host_step_int,
                        )
                        break
    finally:
        progress.close()
        if progress_owned_file is not None:
            progress_owned_file.close()

    state = _to_train_state(compiled_state, step=host_step_int, phase=host_phase_int)
    final_stopper = phase_stoppers[state.phase]
    if final_stopper.reasons and final_stopper.best_step is not None and final_stopper.best_step != host_step_int:
        restored_state = checkpoint_manager.restore(final_stopper.best_step)
        if restored_state is not None:
            state = TrainState(
                model=restored_state.model,
                opt_state_I=restored_state.opt_state_I,
                opt_state_T=restored_state.opt_state_T,
                key=restored_state.key,
                step=restored_state.step,
                phase=state.phase,
            )
            host_step_int = int(state.step)
            host_phase_int = int(state.phase)
            compiled_state = _to_compiled_state(state)

    if _profile:
        _t0 = time.perf_counter()
    state = _to_train_state(compiled_state, step=host_step_int, phase=host_phase_int)
    checkpoint_manager.save(state, host_step_int)
    if _profile:
        _t_ckpt_total += time.perf_counter() - _t0
    _final_meta = _checkpoint_meta(
        phase_stoppers,
        active_phase=state.phase,
        epoch_in_phase=last_epoch_in_phase,
        batch_offset=train_steps_per_epoch,
        validation_completed=True,
    )
    # Record what happened at the phase-2 boundary, so a reader can tell a run
    # that started phase 2 from the phase-1 best from one that did not, without reconstructing
    # it from step numbers.
    _final_meta["restore_phase1_best_before_phase2"] = (
        training_config.restore_phase1_best_before_phase2
    )
    _final_meta["phase1_best_restore"] = _phase1_best_restored
    _write_train_meta(
        checkpoint_manager,
        host_step_int,
        training_config,
        _final_meta,
        dp_overrides=dp_overrides,
        data_cfg=config.data,
    )
    if _profile:
        _mean_train = _t_train_total / max(_n_train_steps, 1)
        print(
            f"[MADE_PROFILE] train_step_mean={_mean_train:.4f}s"
            f"  validation_total={_t_val_total:.4f}s"
            f"  checkpoint_total={_t_ckpt_total:.4f}s"
            f"  n_train_steps={_n_train_steps}",
            file=__import__("sys").stderr,
        )
    return state.model
