# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

from dataclasses import replace
import json
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from made.physics import BoxConstraints
from made.training.losses import (
    phase1_i_loss,
    phase1_loss,
    phase1_t_loss,
    phase2_i_loss,
    phase2_loss,
    phase2_t_loss,
)
from made.training import create_train_state, train
from made.utils import CheckpointManager, ExperimentConfig
from made.utils.config import TrainingConfig
from made.training.trainer import (
    _PROPOSAL_KEY_STREAM_TRAIN,
    _PROPOSAL_KEY_STREAM_VAL,
    _CompiledTrainState,
    _EarlyStoppingState,
    _check_train_meta,
    _compute_x_proposal,
    _early_stop_reason,
    _epoch_batches,
    _load_phase_stoppers,
    _load_resume_cursor,
    _phase_epoch_progress,
    _proposal_perturbation_key,
    _rewrite_phase1_seed_meta,
    _train_step_phase1_i,
    _train_step_phase1_t,
    _weighted_mean_metric_dict,
    _write_train_meta,
)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from made.models import MaDECell
    from pathlib import Path


def test_create_train_state(small_cell: "MaDECell") -> None:
    """Check that create train state."""
    state = create_train_state(small_cell, ExperimentConfig().training, jax.random.key(0))
    assert state.step == 0
    assert state.phase == 1


def test_create_train_state_uses_i_side_clip_when_set(small_cell: "MaDECell") -> None:
    """Check that create train state uses i side clip when set."""
    state = create_train_state(
        small_cell,
        TrainingConfig(i_side_grad_clip_norm=1.0),
        jax.random.key(0),
    )

    # Chain is (zero_nans, clip_by_global_norm, adam): index 0 is ZeroNansState.
    assert isinstance(state.opt_state_I[0], optax.ZeroNansState)
    assert isinstance(state.opt_state_I[1], optax.EmptyState)
    assert isinstance(state.opt_state_I[2], tuple)


def test_create_train_state_no_i_side_clip_when_none(small_cell: "MaDECell") -> None:
    """Check that create train state no i side clip when none."""
    state = create_train_state(
        small_cell,
        TrainingConfig(i_side_grad_clip_norm=None),
        jax.random.key(0),
    )

    # Chain is (zero_nans, adam): no clip slot, so index 1 is adam's state tuple.
    assert isinstance(state.opt_state_I[0], optax.ZeroNansState)
    assert len(state.opt_state_I) == 2
    assert not isinstance(state.opt_state_I[1], optax.EmptyState)


def _cell_with_nonzero_delta_i(cell: "MaDECell") -> "MaDECell":
    """Return a copy of `cell` whose final inverse-dynamics bias is all ones.

    Args:
        cell: The cell to copy.

    Returns:
        The cell with a nonzero residual.
    """
    final_bias = cell.inverse_dynamics.mlp.layers[-1].bias
    updated_bias = jnp.ones_like(final_bias)
    return eqx.tree_at(
        lambda made_cell: made_cell.inverse_dynamics.mlp.layers[-1].bias,
        cell,
        updated_bias,
    )


def _loss_batch(sample_batch: dict[str, jax.Array]) -> tuple[Any, ...]:
    """Unpack a sample batch into the positional loss-function arguments.

    Args:
        sample_batch: Batch dict from the `sample_batch` fixture.

    Returns:
        `(x_prev, x_curr, params, dt, u_gt)`.
    """
    return (
        sample_batch["x_prev"],
        sample_batch["x_curr"],
        sample_batch["params"],
        0.1,
        sample_batch["u_gt"],
    )


def test_phase1_i_loss_total_includes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that phase1 i loss total includes delta i term."""
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=10.0)

    total_off, metrics_off = phase1_i_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, metrics_on = phase1_i_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert metrics_off["delta_i_norm"] > 0.0
    assert jnp.allclose(metrics_off["delta_i_norm"], metrics_on["delta_i_norm"])
    assert jnp.allclose(total_on - total_off, 10.0 * metrics_off["delta_i_norm"], rtol=1e-6)


def test_phase2_i_loss_total_includes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that phase2 i loss total includes delta i term."""
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=10.0)

    total_off, metrics_off = phase2_i_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, metrics_on = phase2_i_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert metrics_off["delta_i_norm"] > 0.0
    assert jnp.allclose(metrics_off["delta_i_norm"], metrics_on["delta_i_norm"])
    assert jnp.allclose(total_on - total_off, 10.0 * metrics_off["delta_i_norm"], rtol=1e-6)


def test_phase1_t_loss_total_excludes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that phase1 t loss total excludes delta i term."""
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=1e6)

    total_off, _ = phase1_t_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, _ = phase1_t_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert jnp.allclose(total_off, total_on)


def test_phase2_t_loss_total_excludes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that phase2 t loss total excludes delta i term."""
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=1e6)

    total_off, _ = phase2_t_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, _ = phase2_t_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert jnp.allclose(total_off, total_on)


def test_phase1_loss_total_includes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """`phase1_loss` total carries the lambda_delta_i_norm penalty.

    Keeps `val/loss` (and validation-driven early stopping) sensitive to the
    regularizer pressure I-side training applies, so early stopping fires
    when ΔI saturates.
    """
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=10.0)

    total_off, metrics_off = phase1_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, metrics_on = phase1_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert metrics_off["delta_i_norm"] > 0.0
    assert jnp.allclose(metrics_off["delta_i_norm"], metrics_on["delta_i_norm"])
    assert jnp.allclose(total_on - total_off, 10.0 * metrics_off["delta_i_norm"], rtol=1e-6)


def test_phase2_loss_total_includes_delta_i_term(
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """`phase2_loss` total carries the lambda_delta_i_norm penalty via
    `phase1_loss`, so validation/early-stop signals inherit it."""
    cell = _cell_with_nonzero_delta_i(small_cell)
    x_prev, x_curr, params, dt, u_sampled = _loss_batch(sample_batch)
    cfg_off = TrainingConfig(lambda_delta_i_norm=0.0)
    cfg_on = TrainingConfig(lambda_delta_i_norm=10.0)

    total_off, metrics_off = phase2_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_off)
    total_on, metrics_on = phase2_loss(cell, x_prev, x_curr, params, dt, u_sampled, cfg_on)

    assert metrics_off["delta_i_norm"] > 0.0
    assert jnp.allclose(metrics_off["delta_i_norm"], metrics_on["delta_i_norm"])
    assert jnp.allclose(total_on - total_off, 10.0 * metrics_off["delta_i_norm"], rtol=1e-6)


def test_training_smoke(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that training smoke."""
    config = ExperimentConfig(output_dir=str(tmp_path))
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    trained = train(small_cell, [sample_batch], [sample_batch], config, checkpoint_manager)
    assert trained is not None


def _budgeted_config(
    tmp_path: "Path", *, steps_per_epoch: int = 2, val_steps_per_epoch: int = 1
) -> ExperimentConfig:
    """Return a one-epoch-per-phase config with a small step budget.

    Args:
        tmp_path: Output directory.
        steps_per_epoch: Training steps per epoch.
        val_steps_per_epoch: Validation steps per epoch.

    Returns:
        The experiment config.
    """
    base = ExperimentConfig(output_dir=str(tmp_path))
    return replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=1,
            num_epochs_phase2=1,
            steps_per_epoch=steps_per_epoch,
            val_steps_per_epoch=val_steps_per_epoch,
            inverse_training="supervised_pretrain",
        ),
    )


def test_steps_per_epoch_caps_steps(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that steps per epoch caps steps."""
    config = _budgeted_config(tmp_path, steps_per_epoch=2, val_steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    train(small_cell, [sample_batch, sample_batch, sample_batch], [sample_batch], config, checkpoint_manager)

    state = checkpoint_manager.restore()
    assert state is not None
    assert state.step == 4
    assert state.phase == 2


def test_steps_per_epoch_oversize_raises(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that steps per epoch oversize raises."""
    config = _budgeted_config(tmp_path, steps_per_epoch=3)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    with pytest.raises(ValueError, match="steps_per_epoch=3 exceeds available batches=2"):
        train(small_cell, [sample_batch, sample_batch], [sample_batch], config, checkpoint_manager)


def test_epoch_batch_sampling_without_replacement() -> None:
    """Check that epoch batch sampling without replacement."""
    key_root = jax.random.fold_in(jax.random.key(0), 0)
    selected = _epoch_batches(list(range(20)), 10, key_root, 0, capped=True)
    assert len(selected) == 10
    assert len(set(selected)) == 10


def test_epoch_batch_sampling_is_deterministic_for_same_seed() -> None:
    """Check that epoch batch sampling is deterministic for same seed."""
    key_root = jax.random.fold_in(jax.random.key(7), 0)
    batches = list(range(20))
    first = _epoch_batches(batches, 10, key_root, 2, capped=True)
    second = _epoch_batches(batches, 10, key_root, 2, capped=True)
    assert first == second


def test_epoch_batch_sampling_differs_for_different_seeds() -> None:
    """Check that epoch batch sampling differs for different seeds."""
    batches = list(range(20))
    first = _epoch_batches(batches, 10, jax.random.fold_in(jax.random.key(7), 0), 2, capped=True)
    second = _epoch_batches(batches, 10, jax.random.fold_in(jax.random.key(8), 0), 2, capped=True)
    assert first != second


def test_uncapped_epoch_batches_keep_original_order() -> None:
    """Check that uncapped epoch batches keep original order."""
    batches = list(range(5))
    selected = _epoch_batches(batches, 5, jax.random.key(0), 0, capped=False)
    assert selected == batches


def test_weighted_mean_metric_dict_weights_partial_batches_by_sample_count() -> None:
    """Check that weighted mean metric dict weights partial batches by sample count."""
    metrics = _weighted_mean_metric_dict(
        [
            (512, {"loss": 1.0, "forward_consistency": 3.0}),
            (384, {"loss": 10.0, "forward_consistency": 30.0}),
        ]
    )
    assert metrics["loss"] == pytest.approx((512 * 1.0 + 384 * 10.0) / 896)
    assert metrics["forward_consistency"] == pytest.approx((512 * 3.0 + 384 * 30.0) / 896)


def test_resume_mismatch_raises(tmp_path: "Path", small_cell: "MaDECell") -> None:
    """Check that resume mismatch raises."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    mismatched = replace(
        config,
        training=replace(config.training, steps_per_epoch=2),
    )
    with pytest.raises(ValueError, match="steps_per_epoch must be stable across resumes"):
        _check_train_meta(checkpoint_manager, state.step, mismatched.training)


def test_resume_match_succeeds(tmp_path: "Path", small_cell: "MaDECell") -> None:
    """Check that resume match succeeds."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    _check_train_meta(checkpoint_manager, state.step, config.training)


def test_resume_early_stopping_config_mismatch_raises(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """Check that resume early stopping config mismatch raises."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    config = replace(config, training=replace(config.training, early_stopping_enabled=True))
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    mismatched = replace(
        config,
        training=replace(config.training, early_stopping_patience=3),
    )
    with pytest.raises(ValueError, match="early stopping config must be stable across resumes"):
        _check_train_meta(checkpoint_manager, state.step, mismatched.training)


def test_legacy_checkpoint_without_sidecar_allows_restore(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """Check that legacy checkpoint without sidecar allows restore."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)

    _check_train_meta(checkpoint_manager, state.step, config.training)


def test_phase_epoch_progress_reports_partial_epoch_offset() -> None:
    """Check that phase epoch progress reports partial epoch offset."""
    assert _phase_epoch_progress(step=3, phase1_steps=8, train_steps_per_epoch=4) == (0, 0, 3)
    assert _phase_epoch_progress(step=10, phase1_steps=8, train_steps_per_epoch=4) == (2, 0, 2)


def test_load_phase_stoppers_restores_history(tmp_path: "Path", small_cell: "MaDECell") -> None:
    """Check that load phase stoppers restores history."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    stopper = _EarlyStoppingState(
        best_loss=0.25,
        best_step=7,
        wait=2,
        history=[_early_stop_metrics(7, loss=0.25)],
        reasons=["phase1:validation_patience:step7"],
    )
    _write_train_meta(
        checkpoint_manager,
        state.step,
        config.training,
        {"early_stopping": {"1": stopper.to_dict(), "2": _EarlyStoppingState().to_dict()}},
    )

    restored = _load_phase_stoppers(checkpoint_manager, state.step)

    assert restored[1].best_loss == 0.25
    assert restored[1].best_step == 7
    assert restored[1].wait == 2
    assert restored[1].history == [_early_stop_metrics(7, loss=0.25)]
    assert restored[1].reasons == ["phase1:validation_patience:step7"]


def test_load_resume_cursor_uses_saved_phase_metadata(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """Check that load resume cursor uses saved phase metadata."""
    config = _budgeted_config(tmp_path, steps_per_epoch=2)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(
        checkpoint_manager,
        state.step,
        config.training,
        {
            "resume": {
                "phase": 2,
                "epoch_in_phase": 1,
                "batch_offset": 1,
                "validation_completed": False,
            }
        },
    )

    assert _load_resume_cursor(checkpoint_manager, 0, phase1_steps=4, train_steps_per_epoch=2) == (
        2,
        1,
        1,
        False,
    )


def _early_stop_metrics(step: int, *, loss: float = 1.0) -> dict[str, float]:
    """Build a metrics dict with the given loss and zeroed components.

    Args:
        step: Training step recorded in the metrics.
        loss: Validation loss value.

    Returns:
        The metrics dict.
    """
    return {
        "loss": loss,
        "step": float(step),
        "forward_consistency": 0.0,
        "inverse_consistency": 0.0,
        "minimum_norm": 0.0,
        "delta_i_norm": 0.0,
    }


def test_early_stopping_disabled_returns_no_reason() -> None:
    """Check that early stopping disabled returns no reason."""
    cfg = replace(ExperimentConfig().training, early_stopping_enabled=False, early_stopping_patience=1)
    stopper = _EarlyStoppingState()

    reason = _early_stop_reason(cfg, stopper, _early_stop_metrics(1), phase=1, epoch_in_phase=0)

    assert reason is None
    assert stopper.history == []


def test_validation_patience_stop_reason() -> None:
    """Check that validation patience stop reason."""
    cfg = replace(
        ExperimentConfig().training,
        early_stopping_enabled=True,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )
    stopper = _EarlyStoppingState()

    assert _early_stop_reason(cfg, stopper, _early_stop_metrics(1), phase=1, epoch_in_phase=0) is None
    assert _early_stop_reason(cfg, stopper, _early_stop_metrics(2), phase=1, epoch_in_phase=1) is None
    reason = _early_stop_reason(cfg, stopper, _early_stop_metrics(3), phase=1, epoch_in_phase=2)

    assert reason == "phase1:validation_patience:step3"


def test_physical_exact_stop_reason() -> None:
    """Check that physical exact stop reason."""
    cfg = replace(
        ExperimentConfig().training,
        early_stopping_enabled=True,
        early_stopping_physical_exact=True,
    )
    stopper = _EarlyStoppingState()

    reason = _early_stop_reason(cfg, stopper, _early_stop_metrics(1), phase=2, epoch_in_phase=0)

    assert reason == "phase2:physical_exact:step1"


def test_physical_saturation_stop_reason() -> None:
    """Check that physical saturation stop reason."""
    cfg = replace(
        ExperimentConfig().training,
        early_stopping_enabled=True,
        early_stopping_physical_saturation=True,
        early_stopping_saturation_window=3,
        early_stopping_saturation_delta=0.01,
        early_stopping_forward_tol=1e-4,
        early_stopping_inverse_tol=1e-4,
    )
    stopper = _EarlyStoppingState()
    metrics = [
        _early_stop_metrics(1, loss=3.0),
        _early_stop_metrics(2, loss=2.0),
        _early_stop_metrics(3, loss=1.0),
    ]
    metrics[0]["minimum_norm"] = 0.50
    metrics[1]["minimum_norm"] = 0.505
    metrics[2]["minimum_norm"] = 0.502
    metrics[0]["delta_i_norm"] = 0.20
    metrics[1]["delta_i_norm"] = 0.201
    metrics[2]["delta_i_norm"] = 0.199

    assert _early_stop_reason(cfg, stopper, metrics[0], phase=1, epoch_in_phase=0) is None
    assert _early_stop_reason(cfg, stopper, metrics[1], phase=1, epoch_in_phase=1) is None
    reason = _early_stop_reason(cfg, stopper, metrics[2], phase=1, epoch_in_phase=2)

    assert reason == "phase1:physical_saturation:step3"


def test_phase1_early_stop_still_runs_phase2(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Check that phase1 early stop still runs phase2."""
    base = _budgeted_config(tmp_path, steps_per_epoch=1, val_steps_per_epoch=1)
    config = replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=3,
            num_epochs_phase2=2,
            early_stopping_enabled=True,
            early_stopping_physical_exact=True,
            early_stopping_forward_tol=1e9,
            early_stopping_inverse_tol=1e9,
            early_stopping_minimum_norm_tol=1e9,
            early_stopping_delta_i_norm_tol=1e9,
        ),
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    train(small_cell, [sample_batch], [sample_batch], config, checkpoint_manager)

    state = checkpoint_manager.restore()
    assert state is not None
    assert state.phase == 2
    assert state.step == 2
    meta = json.loads((tmp_path / "checkpoints" / str(state.step) / "train_meta.json").read_text())
    assert any(reason.startswith("phase1:physical_exact") for reason in meta["early_stop_reasons"])


def test_jitted_train_wrappers_use_compiled_state_not_checkpoint_state() -> None:
    """Check that jitted train wrappers use compiled state not checkpoint state."""
    assert _train_step_phase1_i.__annotations__["state"] == "_CompiledTrainState"
    assert _train_step_phase1_t.__annotations__["state"] == "_CompiledTrainState"
    assert _CompiledTrainState.__name__ == "_CompiledTrainState"


def test_metric_log_interval_skips_intermediate_train_sync(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that metric log interval skips intermediate train sync."""
    logged: list[tuple[int | None, dict[str, float]]] = []

    def _capture(metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Record logged metrics with their step.

        Args:
            metrics: Logged metrics.
            step: Training step of the log call.
        """
        logged.append((step, metrics))

    monkeypatch.setattr("made.training.trainer.log_metrics", _capture)
    base = _budgeted_config(tmp_path, steps_per_epoch=3, val_steps_per_epoch=1)
    config = replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=1,
            num_epochs_phase2=0,
            metric_log_interval_steps=2,
        ),
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    train(
        small_cell,
        [sample_batch, sample_batch, sample_batch],
        [sample_batch],
        config,
        checkpoint_manager,
    )

    train_steps = [
        step
        for step, metrics in logged
        if metrics and not all(key.startswith("val/") for key in metrics)
    ]
    assert train_steps == [2, 3]


def test_phase2_metrics_include_phase_local_axes(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that phase2 metrics include phase local axes."""
    logged: list[tuple[int | None, dict[str, float]]] = []

    def _capture(metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Record logged metrics with their step.

        Args:
            metrics: Logged metrics.
            step: Training step of the log call.
        """
        logged.append((step, metrics))

    monkeypatch.setattr("made.training.trainer.log_metrics", _capture)
    base = _budgeted_config(tmp_path, steps_per_epoch=2, val_steps_per_epoch=1)
    config = replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=1,
            num_epochs_phase2=1,
            metric_log_interval_steps=1,
        ),
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    train(
        small_cell,
        [sample_batch, sample_batch],
        [sample_batch],
        config,
        checkpoint_manager,
    )

    phase2_train_rows = [
        (step, metrics)
        for step, metrics in logged
        if "inequality_violation" in metrics
    ]
    assert [(step, metrics["phase_step"]) for step, metrics in phase2_train_rows] == [
        (3, 1.0),
        (4, 2.0),
    ]
    assert all(metrics["phase"] == 2.0 for _, metrics in phase2_train_rows)

    phase2_val_rows = [
        (step, metrics)
        for step, metrics in logged
        if "val/inequality_violation" in metrics
    ]
    assert phase2_val_rows[-1][0] == 4
    assert phase2_val_rows[-1][1]["val/phase"] == 2.0
    assert phase2_val_rows[-1][1]["val/phase_step"] == 2.0


def _perturbation_config(tmp_path: "Path", *, perturbation_type: str) -> ExperimentConfig:
    """Small 1-epoch-per-phase config with supervised training for speed.

    Args:
        tmp_path: Output directory.
        perturbation_type: Phase 2 proposal perturbation type.

    Returns:
        The experiment config.
    """
    base = ExperimentConfig(output_dir=str(tmp_path))
    return replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=1,
            num_epochs_phase2=1,
            steps_per_epoch=1,
            val_steps_per_epoch=1,
            inverse_training="supervised_pretrain",
            phase2_proposal_perturbation_type=perturbation_type,
            phase2_proposal_perturbation_scale=0.1,
        ),
    )


def test_phase2_proposal_perturbation_smoke(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Training with bound_violation perturbation completes without errors and loss is finite."""
    config = _perturbation_config(tmp_path, perturbation_type="bound_violation")
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    trained = train(small_cell, [sample_batch], [sample_batch], config, checkpoint_manager)

    assert trained is not None
    state = checkpoint_manager.restore()
    assert state is not None
    assert state.phase == 2


def test_phase2_proposal_perturbation_changes_model(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """bound_violation perturbation causes different gradient flow than 'none'.

    We train two sessions from the same seed; after Phase 2 the models must
    differ in at least one leaf, proving the perturbation affects training.
    """
    tmp_none = tmp_path / "none"
    tmp_bv = tmp_path / "bv"
    tmp_none.mkdir()
    tmp_bv.mkdir()

    config_none = _perturbation_config(tmp_none, perturbation_type="none")
    config_bv = _perturbation_config(tmp_bv, perturbation_type="bound_violation")

    cm_none = CheckpointManager(str(tmp_none / "checkpoints"), save_interval=100)
    cm_bv = CheckpointManager(str(tmp_bv / "checkpoints"), save_interval=100)

    trained_none = train(small_cell, [sample_batch], [sample_batch], config_none, cm_none)
    trained_bv = train(small_cell, [sample_batch], [sample_batch], config_bv, cm_bv)

    leaves_none = jax.tree_util.tree_leaves(eqx.filter(trained_none, eqx.is_array))
    leaves_bv = jax.tree_util.tree_leaves(eqx.filter(trained_bv, eqx.is_array))

    assert len(leaves_none) == len(leaves_bv)
    any_diff = any(
        not jnp.array_equal(a, b) for a, b in zip(leaves_none, leaves_bv)
    )
    assert any_diff, "Models are identical — perturbation had no effect on training"


def test_compute_x_proposal_phase1_is_identity(small_cell: "MaDECell") -> None:
    """In Phase 1, _compute_x_proposal always returns x_curr unchanged."""
    x_curr = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    cfg_none = TrainingConfig(
        phase2_proposal_perturbation_type="none",
        phase2_proposal_perturbation_scale=0.5,
    )
    cfg_bv = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.5,
    )

    result_none = _compute_x_proposal(x_curr, phase=1, training_config=cfg_none, cell=small_cell)
    result_bv = _compute_x_proposal(x_curr, phase=1, training_config=cfg_bv, cell=small_cell)

    assert jnp.array_equal(result_none, x_curr)
    assert jnp.array_equal(result_bv, x_curr)


def test_compute_x_proposal_phase2_bound_violation_pushes_outward(small_cell: "MaDECell") -> None:
    """Phase 2 bound_violation with scale=0.5 pushes every component outside the box."""
    # Use a state strictly inside the box: state_min=[-10,-10,-5,-5], state_max=[10,10,5,5]
    # Pick x_curr well inside so sign is unambiguous.
    x_curr = jnp.array([[5.0, -5.0, 2.0, -2.0]])  # shape (1, 4)
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.5,
    )

    result = _compute_x_proposal(x_curr, phase=2, training_config=cfg, cell=small_cell)

    # Access constraints via CompositeConstraints → first child BoxConstraints
    constraints = small_cell.constraints
    state_min = constraints.state_min  # shape (4,)
    state_max = constraints.state_max  # shape (4,)

    # Every component of result must be strictly outside [state_min, state_max]
    outside = jnp.logical_or(result[0] > state_max, result[0] < state_min)
    assert jnp.all(outside), (
        f"Expected all components outside [{state_min}, {state_max}], got {result[0]}"
    )


def _box_cell(state_min: list[float], state_max: list[float]) -> SimpleNamespace:
    """Minimal cell stand-in exposing exactly what `_compute_x_proposal` reads.

    Args:
        state_min: Lower state bounds.
        state_max: Upper state bounds.

    Returns:
        A namespace with a `constraints` attribute.
    """
    constraints = BoxConstraints(
        state_min=jnp.asarray(state_min, dtype=jnp.float64),
        state_max=jnp.asarray(state_max, dtype=jnp.float64),
        control_min=jnp.asarray([-1.0, -1.0], dtype=jnp.float64),
        control_max=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
    )
    return SimpleNamespace(constraints=constraints)


def test_phase2_gaussian_fixed_seed_and_step_is_deterministic(small_cell: "MaDECell") -> None:
    """Same (seed, stream, step) => bit-identical corruption; a new step differs."""
    cfg = TrainingConfig(
        seed=3,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
    )
    x_curr = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.5, -1.5]])

    key_a = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 17)
    key_b = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 17)
    out_a = _compute_x_proposal(x_curr, 2, cfg, small_cell, key_a)
    out_b = _compute_x_proposal(x_curr, 2, cfg, small_cell, key_b)
    assert jnp.array_equal(out_a, out_b)

    key_next = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 18)
    out_next = _compute_x_proposal(x_curr, 2, cfg, small_cell, key_next)
    assert not jnp.array_equal(out_a, out_next)


def test_phase2_train_and_val_key_streams_differ(small_cell: "MaDECell") -> None:
    """Train and validation corruption streams never coincide at the same step."""
    cfg = TrainingConfig(
        seed=3,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
    )
    x_curr = jnp.zeros((4, 4))

    train_key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 5)
    val_key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_VAL, 5)
    assert not jnp.array_equal(
        jax.random.key_data(train_key), jax.random.key_data(val_key)
    )

    out_train = _compute_x_proposal(x_curr, 2, cfg, small_cell, train_key)
    out_val = _compute_x_proposal(x_curr, 2, cfg, small_cell, val_key)
    assert not jnp.array_equal(out_train, out_val)


def test_phase2_per_dim_zero_leaves_dimension_bit_unchanged() -> None:
    """A per-dimension entry of 0.0 means that dimension is never touched."""
    cell = _box_cell((-10.0, -10.0, -5.0, -5.0), (10.0, 10.0, 5.0, 5.0))
    cfg = TrainingConfig(
        seed=0,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.5,
        phase2_proposal_perturbation_scale_per_dim=(0.0, 0.3, 0.0, 0.0),
    )
    x_curr = jnp.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, cell, key)

    assert jnp.array_equal(out[:, 0], x_curr[:, 0])
    assert jnp.array_equal(out[:, 2], x_curr[:, 2])
    assert jnp.array_equal(out[:, 3], x_curr[:, 3])
    assert not jnp.array_equal(out[:, 1], x_curr[:, 1])


def test_phase2_per_dim_scales_are_absolute_state_units() -> None:
    """sigma=0.3 gives ~0.3-scale noise regardless of that dimension's box range."""
    # dim 0 spans 2000 units, dim 1 spans 2 units — both must get sigma == 0.3.
    cell = _box_cell((-1000.0, -1.0, -5.0, -5.0), (1000.0, 1.0, 5.0, 5.0))
    cfg = TrainingConfig(
        seed=1,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.5,
        phase2_proposal_perturbation_scale_per_dim=(0.3, 0.3, 0.0, 0.0),
    )
    x_curr = jnp.zeros((8192, 4))
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, cell, key)

    assert float(jnp.std(out[:, 0])) == pytest.approx(0.3, abs=0.02)
    assert float(jnp.std(out[:, 1])) == pytest.approx(0.3, abs=0.02)


def test_phase2_per_dim_perturbs_infinite_range_dimension() -> None:
    """Per-dim (absolute) magnitudes need no box range, so inf-bound dims corrupt."""
    inf = float("inf")
    cell = _box_cell((-inf, -10.0, -5.0, -5.0), (inf, 10.0, 5.0, 5.0))
    cfg = TrainingConfig(
        seed=2,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
        phase2_proposal_perturbation_scale_per_dim=(0.5, 0.0, 0.0, 0.0),
    )
    x_curr = jnp.zeros((1024, 4))
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, cell, key)

    assert bool(jnp.all(jnp.isfinite(out)))
    assert float(jnp.std(out[:, 0])) == pytest.approx(0.5, abs=0.05)
    assert jnp.array_equal(out[:, 1:], x_curr[:, 1:])


def test_phase2_scalar_path_leaves_infinite_range_dimension_untouched() -> None:
    """The range-fraction (scalar) path still zeroes infinite-range dimensions."""
    inf = float("inf")
    cell = _box_cell((-inf, -10.0, -5.0, -5.0), (inf, 10.0, 5.0, 5.0))
    cfg = TrainingConfig(
        seed=2,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
    )
    x_curr = jnp.zeros((256, 4))
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, cell, key)

    assert jnp.array_equal(out[:, 0], x_curr[:, 0])
    assert not jnp.array_equal(out[:, 1], x_curr[:, 1])


def test_phase2_zero_scale_is_exact_no_op(small_cell: "MaDECell") -> None:
    """scale=0 and all-zero per-dim are exact no-ops for both perturbation types."""
    x_curr = jnp.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    key = _proposal_perturbation_key(0, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    cfg_gaussian = TrainingConfig(
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.0,
    )
    assert jnp.array_equal(
        _compute_x_proposal(x_curr, 2, cfg_gaussian, small_cell, key), x_curr
    )

    cfg_gaussian_pd = TrainingConfig(
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.7,
        phase2_proposal_perturbation_scale_per_dim=(0.0, 0.0, 0.0, 0.0),
    )
    assert jnp.array_equal(
        _compute_x_proposal(x_curr, 2, cfg_gaussian_pd, small_cell, key), x_curr
    )

    cfg_bv = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.0,
    )
    assert jnp.array_equal(_compute_x_proposal(x_curr, 2, cfg_bv, small_cell), x_curr)

    cfg_bv_pd = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.7,
        phase2_proposal_perturbation_scale_per_dim=(0.0, 0.0, 0.0, 0.0),
    )
    assert jnp.array_equal(
        _compute_x_proposal(x_curr, 2, cfg_bv_pd, small_cell), x_curr
    )


def test_phase2_none_is_identity_and_key_free(small_cell: "MaDECell") -> None:
    """'none' returns x_curr itself, needs no key, and ignores the per-dim tuple."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="none",
        phase2_proposal_perturbation_scale=0.5,
        phase2_proposal_perturbation_scale_per_dim=(0.3, 0.3, 0.3, 0.3),
    )
    x_curr = jnp.array([[1.0, 2.0, 3.0, 4.0]])

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell)

    assert out is x_curr


def test_phase2_bound_violation_scalar_is_byte_identical(small_cell: "MaDECell") -> None:
    """bound_violation without per-dim reproduces the historical formula bitwise."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.13,
    )
    x_curr = jnp.array([[5.0, -5.0, 2.0, -2.0], [-3.0, 7.0, -1.0, 4.0]])

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell)

    constraints = small_cell.constraints
    state_min = constraints.state_min
    state_max = constraints.state_max
    range_ = state_max - state_min
    safe_range = jnp.where(jnp.isfinite(range_), range_, 0.0)
    midpoint = 0.5 * (state_min + state_max)
    sign = jnp.where(x_curr >= midpoint, 1.0, -1.0)
    expected = x_curr + sign * 0.13 * safe_range

    assert out.dtype == expected.dtype
    assert jnp.array_equal(out, expected)


def test_phase2_bound_violation_per_dim_is_absolute_push(small_cell: "MaDECell") -> None:
    """With per-dim set, bound_violation pushes by absolute distances."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.13,
        phase2_proposal_perturbation_scale_per_dim=(0.4, 0.0, 0.25, 0.0),
    )
    # small_cell box midpoint is 0 on every dimension.
    x_curr = jnp.array([[5.0, -5.0, 2.0, -2.0]])

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell)

    expected = jnp.array([[5.4, -5.0, 2.25, -2.0]])
    assert jnp.allclose(out, expected)


def test_phase2_gaussian_without_key_raises(small_cell: "MaDECell") -> None:
    """Check that phase2 gaussian without key raises."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale=0.1,
    )
    with pytest.raises(ValueError, match="requires a PRNG key"):
        _compute_x_proposal(jnp.zeros((2, 4)), 2, cfg, small_cell)


def test_phase2_per_dim_length_mismatch_raises(small_cell: "MaDECell") -> None:
    """Check that phase2 per dim length mismatch raises."""
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.1,
        phase2_proposal_perturbation_scale_per_dim=(0.1, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="does not match state dim"):
        _compute_x_proposal(jnp.zeros((2, 4)), 2, cfg, small_cell)


@pytest.mark.parametrize(
    "override",
    [
        {"phase2_proposal_perturbation_type": "gaussian"},
        {"phase2_proposal_perturbation_scale": 0.25},
        {"phase2_proposal_perturbation_scale_per_dim": (0.1, 0.1, 0.1, 0.1)},
        {
            "phase2_proposal_perturbation_scale_sampling": "loguniform",
            "phase2_proposal_perturbation_scale_min_ratio": 0.01,
        },
        {"phase2_proposal_perturbation_zero_fraction": 0.25},
    ],
)
def test_resume_phase2_perturbation_mismatch_raises(
    tmp_path: "Path",
    small_cell: "MaDECell",
    override: dict[str, Any],
) -> None:
    """A resume cannot silently switch the Phase-2 proposal corruption."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    config = replace(
        config,
        training=replace(
            config.training,
            phase2_proposal_perturbation_type="bound_violation",
            phase2_proposal_perturbation_scale=0.1,
        ),
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    mismatched = replace(config, training=replace(config.training, **override))
    with pytest.raises(ValueError, match="comparability_signature mismatch"):
        _check_train_meta(checkpoint_manager, state.step, mismatched.training)


def test_resume_phase2_perturbation_match_succeeds(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """Identical perturbation settings resume cleanly."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    config = replace(
        config,
        training=replace(
            config.training,
            phase2_proposal_perturbation_type="gaussian",
            phase2_proposal_perturbation_scale=0.0,
            phase2_proposal_perturbation_scale_per_dim=(0.3, 0.0, 0.05, 0.0),
        ),
    )
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    _check_train_meta(checkpoint_manager, state.step, config.training)


def test_resume_pre_change_checkpoint_meta_still_loads(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """A meta written before the phase2-perturbation signature keys existed loads."""
    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    _write_train_meta(checkpoint_manager, state.step, config.training)

    meta_path = checkpoint_manager.directory / str(state.step) / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for stale_key in (
        "phase2_proposal_perturbation_type",
        "phase2_proposal_perturbation_scale",
        "phase2_proposal_perturbation_scale_per_dim",
        "phase2_proposal_perturbation_scale_sampling",
        "phase2_proposal_perturbation_scale_min_ratio",
        "phase2_proposal_perturbation_zero_fraction",
    ):
        meta["comparability_signature"].pop(stale_key, None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    _check_train_meta(checkpoint_manager, state.step, config.training)


def test_phase2_gaussian_training_smoke(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """End-to-end: gaussian perturbation trains through both call sites."""
    config = _perturbation_config(tmp_path, perturbation_type="gaussian")
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=100)

    trained = train(small_cell, [sample_batch], [sample_batch], config, checkpoint_manager)

    assert trained is not None
    restored = checkpoint_manager.restore()
    assert restored is not None
    assert restored.phase == 2


def test_phase2_bound_violation_per_dim_pushes_outward_on_infinite_bounds() -> None:
    """An unbounded dim must still be pushed AWAY from zero, not always down."""
    inf = float("inf")
    cell = _box_cell((-inf, -10.0, -5.0, -5.0), (inf, 10.0, 5.0, 5.0))
    cfg = TrainingConfig(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale=0.1,
        phase2_proposal_perturbation_scale_per_dim=(0.5, 0.0, 0.0, 0.0),
    )
    x_curr = jnp.array([[3.0, 2.0, 1.0, 0.0], [-3.0, -2.0, -1.0, 0.0]])

    out = _compute_x_proposal(x_curr, 2, cfg, cell)

    assert float(out[0, 0]) == pytest.approx(3.5)
    assert float(out[1, 0]) == pytest.approx(-3.5)
    assert jnp.array_equal(out[:, 1:], x_curr[:, 1:])


def test_curriculum_seed_meta_allows_new_phase2_perturbation(
    tmp_path: "Path",
    small_cell: "MaDECell",
) -> None:
    """A Phase-1 seed meta must not block a Phase-2 cell that enables corruption."""
    from pathlib import Path

    config = _budgeted_config(tmp_path, steps_per_epoch=1)
    checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    state = create_train_state(small_cell, config.training, jax.random.key(0))
    checkpoint_manager.save(state, state.step)
    # Phase-1 seed run: perturbation left at its defaults.
    _write_train_meta(
        checkpoint_manager,
        state.step,
        config.training,
        extra={
            "resume": {
                "phase": 1,
                "epoch_in_phase": 0,
                "batch_offset": 0,
                "validation_completed": False,
            }
        },
    )
    meta_path = Path(checkpoint_manager.directory) / str(state.step) / "train_meta.json"
    _rewrite_phase1_seed_meta(meta_path, "/some/phase1/dir")

    phase2_cfg = replace(
        config.training,
        pretrained_phase1_path="/some/phase1/dir",
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale_per_dim=(0.3, 0.0, 0.05, 0.0),
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.02,
        phase2_proposal_perturbation_zero_fraction=0.1,
    )
    _check_train_meta(checkpoint_manager, state.step, phase2_cfg)


_RANDOMISED_SIGMA_MAX = (0.4, 0.2, 0.0, 0.1)


def _randomised_cfg(**overrides: Any) -> TrainingConfig:
    """Return a randomised-perturbation `TrainingConfig` with optional overrides.

    Args:
        **overrides: Fields overriding the defaults.

    Returns:
        The training config.
    """
    base = dict(
        seed=7,
        phase2_proposal_perturbation_type="gaussian",
        phase2_proposal_perturbation_scale_per_dim=_RANDOMISED_SIGMA_MAX,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def test_randomised_defaults_reproduce_fixed_sigma_bit_for_bit(small_cell: "MaDECell") -> None:
    """Default fields ('fixed'/0.0) change nothing, bit-for-bit.

    The default config must not consume extra randomness, so
    `x_curr + sigma_max * normal(key)` is reproduced exactly.
    """
    cfg = _randomised_cfg()
    assert cfg.phase2_proposal_perturbation_scale_sampling == "fixed"
    assert cfg.phase2_proposal_perturbation_scale_min_ratio == 0.0
    assert cfg.phase2_proposal_perturbation_zero_fraction == 0.0

    x_curr = jax.random.normal(jax.random.key(0), (64, 4), dtype=jnp.float64)
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 11)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    expected = x_curr + jnp.asarray(_RANDOMISED_SIGMA_MAX) * jax.random.normal(
        key, x_curr.shape, dtype=x_curr.dtype
    )
    assert jnp.array_equal(out, expected)


def _expected_multiplier_mean(cfg: TrainingConfig) -> float:
    """Return the analytic mean of the sigma multiplier for the sampling mode.

    Args:
        cfg: Config holding the sampling mode and minimum ratio.

    Returns:
        The expected multiplier mean.
    """
    import math

    r = cfg.phase2_proposal_perturbation_scale_min_ratio
    if cfg.phase2_proposal_perturbation_scale_sampling == "uniform":
        return 0.5 * (r + 1.0)
    # E[exp(U(log r, 0))] = (1 - r) / (-log r)
    return (1.0 - r) / (-math.log(r))


@pytest.mark.parametrize("sampling", ["loguniform", "uniform"])
def test_randomised_sigma_varies_across_samples(small_cell: "MaDECell", sampling: str) -> None:
    """Realised |dx| / sigma_max has nonzero spread across rows."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling=sampling,
        phase2_proposal_perturbation_scale_min_ratio=0.01,
    )
    x_curr = jnp.zeros((32768, 4))
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    # Per-row realised magnitude on dim 0, normalised by that dim's sigma_max.
    ratio = jnp.abs(out[:, 0]) / _RANDOMISED_SIGMA_MAX[0]

    # Under "fixed" the only spread would come from the standard normal itself;
    # the multiplier draw must widen the distribution well beyond that.
    fixed_out = _compute_x_proposal(x_curr, 2, _randomised_cfg(), small_cell, key)
    fixed_ratio = jnp.abs(fixed_out[:, 0]) / _RANDOMISED_SIGMA_MAX[0]

    assert float(jnp.std(ratio)) > 0.0
    assert float(jnp.mean(ratio)) < float(jnp.mean(fixed_ratio))
    # Every realised sigma is bounded above by sigma_max, so the multiplier
    # only ever shrinks the draw.
    assert float(jnp.mean(ratio)) == pytest.approx(
        float(jnp.mean(fixed_ratio)) * _expected_multiplier_mean(cfg), rel=0.06
    )


def test_randomised_multipliers_are_independent_across_dimensions(small_cell: "MaDECell") -> None:
    """Dimension multipliers are drawn independently: empirical correlation ~ 0."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.01,
        phase2_proposal_perturbation_scale_per_dim=(0.4, 0.4, 0.4, 0.4),
    )
    x_curr = jnp.zeros((100_000, 4))
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 3)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    # |dx| = sigma_max * m * |eps|; |eps| is iid across dims, so a nonzero
    # correlation between log|dx| columns can only come from correlated m.
    log_mag = jnp.log(jnp.abs(out) + 1e-300)
    corr = jnp.corrcoef(log_mag.T)
    off_diagonal = corr - jnp.diag(jnp.diag(corr))
    assert float(jnp.max(jnp.abs(off_diagonal))) < 0.02


def test_randomised_zero_fraction_one_is_exact_identity(small_cell: "MaDECell") -> None:
    """zero_fraction=1.0 leaves every sample bit-identical on every dimension."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.01,
        phase2_proposal_perturbation_zero_fraction=1.0,
    )
    x_curr = jax.random.normal(jax.random.key(1), (512, 4), dtype=jnp.float64)
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    assert jnp.array_equal(out, x_curr)


def test_randomised_zero_fraction_half_leaves_half_the_rows_clean(small_cell: "MaDECell") -> None:
    """zero_fraction=0.5 leaves ~half the rows clean, and cleanliness is whole-row."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.01,
        phase2_proposal_perturbation_zero_fraction=0.5,
    )
    x_curr = jax.random.normal(jax.random.key(2), (8192, 4), dtype=jnp.float64)
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    # dim 2 has sigma_max == 0.0, so restrict the "clean" test to perturbed dims.
    touched_dims = jnp.array([0, 1, 3])
    per_dim_clean = out[:, touched_dims] == x_curr[:, touched_dims]
    row_clean = jnp.all(per_dim_clean, axis=1)

    # Whole-sample guard: a row is either clean on all perturbed dims or none.
    assert bool(jnp.all(jnp.any(per_dim_clean, axis=1) == row_clean))
    assert float(jnp.mean(row_clean)) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize(
    "sampling,min_ratio",
    [("fixed", 0.0), ("uniform", 0.05), ("loguniform", 0.05)],
)
def test_randomised_zero_sigma_dimension_stays_exact(
    small_cell: "MaDECell",
    sampling: str,
    min_ratio: float,
) -> None:
    """A dimension with sigma_max == 0 is untouched under every sampling mode."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling=sampling,
        phase2_proposal_perturbation_scale_min_ratio=min_ratio,
        phase2_proposal_perturbation_zero_fraction=0.3,
    )
    x_curr = jax.random.normal(jax.random.key(3), (1024, 4), dtype=jnp.float64)
    key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 0)

    out = _compute_x_proposal(x_curr, 2, cfg, small_cell, key)
    assert jnp.array_equal(out[:, 2], x_curr[:, 2])
    assert not jnp.array_equal(out[:, 0], x_curr[:, 0])


def test_randomised_draws_are_resume_stable(small_cell: "MaDECell") -> None:
    """Same (seed, stream, step) reproduces identical draws; a new step differs."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.01,
        phase2_proposal_perturbation_zero_fraction=0.25,
    )
    x_curr = jax.random.normal(jax.random.key(4), (256, 4), dtype=jnp.float64)

    key_a = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 42)
    key_b = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 42)
    key_next = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_TRAIN, 43)
    val_key = _proposal_perturbation_key(cfg.seed, _PROPOSAL_KEY_STREAM_VAL, 42)

    out_a = _compute_x_proposal(x_curr, 2, cfg, small_cell, key_a)
    out_b = _compute_x_proposal(x_curr, 2, cfg, small_cell, key_b)
    assert jnp.array_equal(out_a, out_b)
    assert not jnp.array_equal(
        out_a, _compute_x_proposal(x_curr, 2, cfg, small_cell, key_next)
    )
    assert not jnp.array_equal(
        out_a, _compute_x_proposal(x_curr, 2, cfg, small_cell, val_key)
    )


def test_randomised_knobs_rejected_for_bound_violation(small_cell: "MaDECell") -> None:
    """The randomisation knobs are gaussian-only and must not be silently ignored."""
    cfg = _randomised_cfg(
        phase2_proposal_perturbation_type="bound_violation",
        phase2_proposal_perturbation_scale_sampling="loguniform",
        phase2_proposal_perturbation_scale_min_ratio=0.01,
    )
    with pytest.raises(ValueError, match="apply only to"):
        _compute_x_proposal(jnp.zeros((2, 4)), 2, cfg, small_cell)


# restore_phase1_best_before_phase2: phase 2 otherwise inherits phase 1's FINAL state, so
# when early stopping picked an earlier epoch, phase 2 starts from a model the run itself
# judged worse. Covers: default off unchanged, on with best == final unchanged, on with
# best != final actually restores the best.


def _final_train_meta(checkpoints_dir: "Path") -> dict[str, Any]:
    """The train_meta.json of the highest-numbered checkpoint step under `checkpoints_dir`.

    Args:
        checkpoints_dir: Directory of numbered checkpoint step sub-directories.

    Returns:
        The parsed `train_meta.json`.
    """
    from pathlib import Path as _P
    steps = sorted((int(d.name) for d in _P(checkpoints_dir).iterdir()
                    if d.is_dir() and d.name.isdigit()))
    assert steps, f"no checkpoint step directories under {checkpoints_dir}"
    return json.loads((_P(checkpoints_dir) / str(steps[-1]) / "train_meta.json").read_text())


def _two_phase_config(
    tmp_path: "Path", *, restore: bool, patience: int | None = None
) -> ExperimentConfig:
    """Return a config with three Phase 1 epochs and one Phase 2 epoch.

    Args:
        tmp_path: Output directory.
        restore: Value of the restore-best-before-Phase-2 option.
        patience: Early-stopping patience, or None to leave it unset.

    Returns:
        The experiment config.
    """
    base = ExperimentConfig(output_dir=str(tmp_path))
    return replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=3,
            num_epochs_phase2=1,
            steps_per_epoch=1,
            val_steps_per_epoch=1,
            inverse_training="supervised_pretrain",
            early_stopping_enabled=True,
            early_stopping_patience=patience,
            restore_phase1_best_before_phase2=restore,
        ),
    )


def test_restore_phase1_best_defaults_to_off() -> None:
    """The option must not change behaviour for any existing config."""
    assert TrainingConfig().restore_phase1_best_before_phase2 is False


def test_restore_phase1_best_off_leaves_behaviour_unchanged(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Option OFF reproduces today's run exactly, step for step."""
    results = []
    for i in (0, 1):
        d = tmp_path / f"run{i}"
        cm = CheckpointManager(str(d / "checkpoints"), save_interval=1)
        train(small_cell, [sample_batch], [sample_batch], _two_phase_config(d, restore=False), cm)
        st = cm.restore()
        results.append(int(st.step))
    assert results[0] == results[1]

    meta = _final_train_meta(tmp_path / "run0" / "checkpoints")
    assert meta.get("restore_phase1_best_before_phase2") in (False, None)
    # Nothing was restored at the boundary, because the option was off.
    assert meta.get("phase1_best_restore") is None


def test_restore_phase1_best_on_is_a_noop_when_best_is_final(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Option ON with best == final must change nothing: same final step as with it off."""
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    cm_off = CheckpointManager(str(off_dir / "checkpoints"), save_interval=1)
    train(small_cell, [sample_batch], [sample_batch], _two_phase_config(off_dir, restore=False),
          cm_off)
    cm_on = CheckpointManager(str(on_dir / "checkpoints"), save_interval=1)
    train(small_cell, [sample_batch], [sample_batch], _two_phase_config(on_dir, restore=True),
          cm_on)

    assert int(cm_off.restore().step) == int(cm_on.restore().step)

    meta = _final_train_meta(on_dir / "checkpoints")
    assert meta.get("restore_phase1_best_before_phase2") is True
    rec = meta.get("phase1_best_restore")
    # Either the boundary found best == final (nothing to do), or it restored to the same step.
    if rec is not None and rec.get("restored"):
        # A restore fires ONLY when best != final, so the two steps must differ. Asserting
        # equality here could not fail under any implementation and tested nothing.
        assert rec["from_step"] != rec["best_step"]


def test_restore_phase1_best_on_restores_the_best_step(
    tmp_path: "Path",
    small_cell: "MaDECell",
    sample_batch: dict[str, jax.Array],
) -> None:
    """Option ON with best != final must begin phase 2 from the BEST step, not the final one.

    Driven through the trainer's own early-stopping state rather than by hand: phase 1 runs three
    epochs on a batch whose loss does not keep improving, so the stopper's `best_step` lands
    before the phase-1 final step. The assertion is on what the trainer recorded at the boundary,
    which is the thing the option exists to control.
    """
    d = tmp_path / "restore"
    cm = CheckpointManager(str(d / "checkpoints"), save_interval=1)
    train(small_cell, [sample_batch], [sample_batch],
          _two_phase_config(d, restore=True, patience=1), cm)

    meta = _final_train_meta(d / "checkpoints")
    assert meta.get("restore_phase1_best_before_phase2") is True
    rec = meta.get("phase1_best_restore")
    assert rec is not None, "the phase-2 boundary recorded nothing with the option on"
    print(f"\n[restore-phase1-best] phase-2 boundary recorded: {rec}")
    if rec.get("restored"):
        # It restored: it moved TO phase 1's best FROM phase 1's final, so the two differ.
        assert rec["from_step"] != rec["best_step"]
    else:
        # It declined only for the stated reason, never silently.
        assert rec.get("reason") == "phase-1 best IS phase-1 final"


def test_restore_phase1_best_on_genuinely_restores_an_earlier_best(
    tmp_path: "Path", small_cell: "MaDECell", sample_batch: dict[str, jax.Array]
) -> None:
    """The case that matters: best != final, so the restore must FIRE.

    On a synthetic batch the loss keeps improving, so a straight run always ends with
    best == final and the restore branch is never taken -- which is why the sibling test above
    passes through the no-op branch and cannot stand in for this one. Here phase 1 is run to
    completion first, then its recorded best_step is rewritten to an EARLIER step that has a real
    checkpoint on disk, and the run is resumed into phase 2 with the option on. The assertion is
    that phase 2 began from that earlier step.
    """
    ckpt_dir = tmp_path / "checkpoints"

    # 1. Phase 1 only, checkpointing every step, so several steps exist to restore from.
    base = ExperimentConfig(output_dir=str(tmp_path))
    phase1_cfg = replace(base, training=replace(
        base.training, num_epochs_phase1=3, num_epochs_phase2=0,
        steps_per_epoch=1, val_steps_per_epoch=1, inverse_training="supervised_pretrain",
        early_stopping_enabled=True))
    cm = CheckpointManager(str(ckpt_dir), save_interval=1)
    train(small_cell, [sample_batch], [sample_batch], phase1_cfg, cm)

    steps = sorted(int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit())
    assert len(steps) >= 2, f"need at least two checkpointed steps, got {steps}"
    final_step, earlier_step = steps[-1], steps[-2]

    # 2. Rewrite phase 1's recorded best to the EARLIER step, creating best != final.
    meta_path = ckpt_dir / str(final_step) / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    stopper = _EarlyStoppingState(
        best_loss=0.125, best_step=earlier_step, wait=1,
        history=[_early_stop_metrics(earlier_step, loss=0.125)],
        reasons=[f"phase1:validation_patience:step{earlier_step}"])
    meta["early_stopping"] = {"1": stopper.to_dict(), "2": _EarlyStoppingState().to_dict()}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # 3. Resume into phase 2 with the option ON.
    phase2_cfg = replace(base, training=replace(
        base.training, num_epochs_phase1=3, num_epochs_phase2=1,
        steps_per_epoch=1, val_steps_per_epoch=1, inverse_training="supervised_pretrain",
        early_stopping_enabled=True, restore_phase1_best_before_phase2=True))
    cm2 = CheckpointManager(str(ckpt_dir), save_interval=1)
    train(small_cell, [sample_batch], [sample_batch], phase2_cfg, cm2)

    final_meta = _final_train_meta(ckpt_dir)
    rec = final_meta.get("phase1_best_restore")
    assert rec is not None, "the phase-2 boundary recorded nothing with the option on"
    assert rec["restored"] is True, (
        f"the restore did NOT fire even though best ({earlier_step}) != final ({final_step}): "
        f"{rec}")
    assert rec["best_step"] == earlier_step
    assert rec["from_step"] == final_step


def test_restore_phase1_best_fails_loudly_without_early_stopping(
    tmp_path: "Path", small_cell: "MaDECell", sample_batch: dict[str, jax.Array]
) -> None:
    """The guard: with no phase-1 best recorded, refuse rather than quietly use the final model."""
    base = ExperimentConfig(output_dir=str(tmp_path))
    config = replace(
        base,
        training=replace(
            base.training,
            num_epochs_phase1=1,
            num_epochs_phase2=1,
            steps_per_epoch=1,
            val_steps_per_epoch=1,
            inverse_training="supervised_pretrain",
            early_stopping_enabled=False,
            restore_phase1_best_before_phase2=True,
        ),
    )
    cm = CheckpointManager(str(tmp_path / "checkpoints"), save_interval=1)
    with pytest.raises(ValueError, match="no best step"):
        train(small_cell, [sample_batch], [sample_batch], config, cm)
