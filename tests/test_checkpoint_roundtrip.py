"""Checkpoint save/restore round-trip tests (US-019)."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import builtins
import importlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from made.models import MaDECell, MaDEModel
from made.physics import build_system
from made.utils import CheckpointManager
from made.utils.checkpointing import TrainState
from made.utils.config import CorrectorConfig, ModelConfig


def _make_small_cell(key: jax.Array) -> MaDECell:
    physics, constraints = build_system("double_integrator")
    model_cfg = ModelConfig(inverse_hidden=(16, 16), residual_hidden=(16, 16))
    corrector_cfg = CorrectorConfig(mode="enabled")
    return MaDECell.from_config(physics, constraints, model_cfg, corrector_cfg, key=key)


def _leaves(module: eqx.Module) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))


def _all_close(a: list[jax.Array], b: list[jax.Array]) -> bool:
    if len(a) != len(b):
        return False
    return all(jnp.allclose(x, y) for x, y in zip(a, b))


def test_checkpointing_imports_when_orbax_import_fails(monkeypatch, tmp_path):
    """Checkpointing falls back to pickle when Orbax cannot import."""
    import made.utils.checkpointing as checkpointing

    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "orbax.checkpoint" or name.startswith("orbax.checkpoint."):
            raise AttributeError("simulated JAX/Orbax mismatch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    reloaded = importlib.reload(checkpointing)
    try:
        assert reloaded.ocp is None
        ckpt = reloaded.CheckpointManager(str(tmp_path / "ckpt"))
        assert ckpt.restore() is None
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)
        importlib.reload(checkpointing)


def test_made_cell_roundtrip(tmp_path):
    """MaDECell leaves survive save → restore unchanged."""
    cell = _make_small_cell(jax.random.key(0))
    state = TrainState(model=cell, opt_state_I=None, opt_state_T=None,
                       key=jax.random.key(1), step=0)

    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    ckpt.save(state, 0)

    restored = ckpt.restore(0)
    assert restored is not None
    assert isinstance(restored.model, MaDECell)
    assert _all_close(_leaves(cell), _leaves(restored.model))


def test_made_cell_step_metadata(tmp_path):
    """Step and phase survive round-trip."""
    cell = _make_small_cell(jax.random.key(2))
    state = TrainState(model=cell, opt_state_I=None, opt_state_T=None,
                       key=jax.random.key(3), step=42, phase=2)

    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    ckpt.save(state, 42)

    restored = ckpt.restore(42)
    assert restored.step == 42
    assert restored.phase == 2


def test_made_model_without_encoder_roundtrip(tmp_path):
    """MaDEModel (encoder=None) round-trips leaf-by-leaf."""
    cell = _make_small_cell(jax.random.key(4))
    model = MaDEModel(cell=cell, encoder=None)
    state = TrainState(model=model, opt_state_I=None, opt_state_T=None,
                       key=jax.random.key(5), step=0)

    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    ckpt.save(state, 0)

    restored = ckpt.restore(0)
    assert restored is not None
    assert isinstance(restored.model, MaDEModel)
    assert _all_close(_leaves(model), _leaves(restored.model))


def test_latest_step_after_multiple_saves(tmp_path):
    """latest_step returns the highest step saved."""
    cell = _make_small_cell(jax.random.key(6))
    ckpt = CheckpointManager(str(tmp_path / "ckpt"), save_interval=1)
    for step in [0, 5, 10]:
        state = TrainState(model=cell, opt_state_I=None, opt_state_T=None,
                           key=jax.random.key(step), step=step)
        ckpt.save(state, step)
    assert ckpt.latest_step() == 10


def test_made_model_with_encoder_roundtrip(tmp_path):
    """MaDEModel with MetadataEncoder round-trips leaf-by-leaf."""
    from made.models import MaDEModel
    from made.utils.config import ModelConfig, CorrectorConfig

    physics, constraints = build_system("kinematic_bicycle")
    model_cfg = ModelConfig(
        inverse_hidden=(16, 16),
        residual_hidden=(16, 16),
        use_metadata_encoder=True,
        metadata_dim=4,
    )
    corrector_cfg = CorrectorConfig(mode="enabled")
    model = MaDEModel.from_config(
        physics, constraints, model_cfg, corrector_cfg, key=jax.random.key(7)
    )
    assert model.encoder is not None, "Expected encoder to be created"

    state = TrainState(model=model, opt_state_I=None, opt_state_T=None,
                       key=jax.random.key(8), step=0)
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    ckpt.save(state, 0)

    restored = ckpt.restore(0)
    assert restored is not None
    assert isinstance(restored.model, MaDEModel)
    assert restored.model.encoder is not None
    assert _all_close(_leaves(model), _leaves(restored.model))


def test_restore_none_on_empty(tmp_path):
    """restore returns None when no checkpoint exists."""
    ckpt = CheckpointManager(str(tmp_path / "empty"))
    assert ckpt.restore() is None


# ---------------------------------------------------------------------------
# comparability_signature resume checks (Phase 2 — cadence controls)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from made.training.trainer import _check_train_meta, _early_stopping_config_meta
from made.utils.config import DataConfig, TrainingConfig

_DEFAULT_TRAINING_CFG = TrainingConfig()


def _write_sig(ckpt_dir: Path, step: int, sig: dict, cfg: TrainingConfig | None = None) -> None:
    """Write a minimal train_meta.json with a matching early_stopping_config and given sig."""
    _cfg = cfg or _DEFAULT_TRAINING_CFG
    step_dir = ckpt_dir / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": _cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": _cfg.i_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(_cfg),
        "comparability_signature": sig,
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))


def test_comparability_signature_mismatch_raises(tmp_path):
    """_check_train_meta raises when cadence fields differ from stored signature."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={"validation_interval_epochs": 2, "checkpoint_save_interval": None,
             "best_checkpoint_interval_epochs": 1},
    )
    cfg = TrainingConfig(validation_interval_epochs=1)  # differs from stored 2
    with pytest.raises(ValueError, match="comparability_signature mismatch"):
        _check_train_meta(ckpt, 0, cfg)


def test_comparability_signature_match_passes(tmp_path):
    """_check_train_meta does not raise when signature matches stored values."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={"validation_interval_epochs": 1, "checkpoint_save_interval": None,
             "best_checkpoint_interval_epochs": 1},
    )
    cfg = TrainingConfig(validation_interval_epochs=1)
    _check_train_meta(ckpt, 0, cfg)  # should not raise


def test_comparability_signature_absent_skips_check(tmp_path):
    """_check_train_meta does not raise when stored meta has no signature (old checkpoints)."""
    cfg = TrainingConfig(validation_interval_epochs=5)
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": cfg.i_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        # no comparability_signature key — simulates old checkpoint
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    _check_train_meta(ckpt, 0, cfg)  # old checkpoint without sig — should not raise


def test_resume_old_meta_without_t_side_grad_clip_loads_cleanly(tmp_path):
    """Old train_meta without t_side_grad_clip_norm must not fail under new defaults."""
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    _check_train_meta(ckpt, 0, cfg)


def test_resume_old_meta_without_i_side_grad_clip_loads_cleanly(tmp_path):
    """Old train_meta without i_side_grad_clip_norm must not fail under new defaults."""
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    _check_train_meta(ckpt, 0, cfg)


def test_resume_mismatch_i_side_grad_clip_raises(tmp_path):
    cfg = TrainingConfig(i_side_grad_clip_norm=1.0)
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": 0.5,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="i_side_grad_clip_norm"):
        _check_train_meta(ckpt, 0, cfg)


def test_resume_mismatch_t_side_grad_clip_raises(tmp_path):
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": 0.5,
        "i_side_grad_clip_norm": cfg.i_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="t_side_grad_clip_norm"):
        _check_train_meta(ckpt, 0, cfg)


def test_resume_old_meta_without_warmup_steps_loads_cleanly(tmp_path):
    """Old train_meta without warmup_steps must not fail under new defaults (sentinel guard)."""
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": cfg.i_side_grad_clip_norm,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    _check_train_meta(ckpt, 0, cfg)


def test_resume_mismatch_warmup_steps_raises(tmp_path):
    """Resuming with a different warmup_steps raises ValueError containing 'warmup_steps'."""
    cfg = TrainingConfig(warmup_steps=100)
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps_per_epoch": None,
        "val_steps_per_epoch": None,
        "t_side_grad_clip_norm": cfg.t_side_grad_clip_norm,
        "i_side_grad_clip_norm": cfg.i_side_grad_clip_norm,
        "warmup_steps": 50,
        "early_stopping_config": _early_stopping_config_meta(cfg),
        "comparability_signature": {
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
        },
    }
    (step_dir / "train_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="warmup_steps"):
        _check_train_meta(ckpt, 0, cfg)


def test_resume_mismatch_noise_scale_raises(tmp_path):
    """Resuming with a different noise_scale raises ValueError containing 'noise_scale'."""
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    # Write meta with noise_scale=0.0 in the comparability signature.
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
            "noise_scale": 0.0,
        },
    )
    # Attempt resume with noise_scale=0.1 — should raise on 'noise_scale' key.
    data_cfg_new = DataConfig(noise_scale=0.1)
    with pytest.raises(ValueError, match="noise_scale"):
        _check_train_meta(ckpt, 0, cfg, data_cfg=data_cfg_new)


def test_resume_old_meta_without_noise_scale_loads_cleanly(tmp_path):
    """Old checkpoints without noise_scale in comparability_signature load under any noise_scale.

    The overlap-key comparison only checks keys present in BOTH stored and current signatures,
    so an absent noise_scale key in stored meta never raises.
    """
    cfg = TrainingConfig()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    # Write meta WITHOUT noise_scale (simulates a pre-Anomaly-1-fix checkpoint).
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": cfg.validation_interval_epochs,
            "checkpoint_save_interval": cfg.checkpoint_save_interval,
            "best_checkpoint_interval_epochs": cfg.best_checkpoint_interval_epochs,
            # no noise_scale key
        },
    )
    # Both noise_scale=0.0 and noise_scale=0.1 must load cleanly.
    _check_train_meta(ckpt, 0, cfg, data_cfg=DataConfig(noise_scale=0.0))
    _check_train_meta(ckpt, 0, cfg, data_cfg=DataConfig(noise_scale=0.1))


# ---------------------------------------------------------------------------
# pretrained_phase1_path helper tests (Phase C3)
# ---------------------------------------------------------------------------

from made.training.trainer import (
    _maybe_copy_pretrained_checkpoint,
    _rewrite_phase1_seed_meta,
    _self_heal_curriculum_seed_meta,
)


def _phase1_meta() -> str:
    """Minimal valid train_meta.json for a Phase-1-end checkpoint."""
    return json.dumps({"resume": {"phase": 1, "epoch_in_phase": 0, "batch_offset": 0,
                                  "validation_completed": True}})


def test_pretrained_phase1_path_seeds_local_dir(tmp_path):
    """Helper copies src checkpoint into an empty local dir."""
    src_dir = tmp_path / "src"
    step_dir = src_dir / "100"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"fake-state")
    (step_dir / "train_meta.json").write_text(_phase1_meta())

    local_dir = tmp_path / "local"
    local_dir.mkdir()

    _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))

    assert (local_dir / "100" / "state.pkl").exists()
    assert (local_dir / "100" / "train_meta.json").exists()


def test_pretrained_phase1_path_preserves_local_when_present(tmp_path):
    """Helper short-circuits when local already has a checkpoint subdir."""
    src_dir = tmp_path / "src"
    src_step = src_dir / "100"
    src_step.mkdir(parents=True)
    (src_step / "state.pkl").write_bytes(b"src-state")
    (src_step / "train_meta.json").write_text(_phase1_meta())

    local_dir = tmp_path / "local"
    local_step = local_dir / "200"
    local_step.mkdir(parents=True)
    original_bytes = b"local-state-original"
    (local_step / "state.pkl").write_bytes(original_bytes)

    _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))

    # existing local checkpoint must be untouched
    assert (local_dir / "200" / "state.pkl").read_bytes() == original_bytes
    # src step must NOT have been copied (local already had a checkpoint)
    assert not (local_dir / "100").exists()


def test_pretrained_phase1_path_none_is_noop(tmp_path):
    """Passing None raises no exception and makes no filesystem changes."""
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _maybe_copy_pretrained_checkpoint(None, str(local_dir))
    # directory still exists and is empty
    assert local_dir.exists()
    assert list(local_dir.iterdir()) == []


def test_pretrained_phase1_path_missing_raises(tmp_path):
    """A nonexistent pretrained_path raises FileNotFoundError mentioning the path."""
    missing = str(tmp_path / "does_not_exist")
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        _maybe_copy_pretrained_checkpoint(missing, str(local_dir))


# ---------------------------------------------------------------------------
# M2 — Phase-1 source validation tests
# ---------------------------------------------------------------------------


def test_pretrained_phase1_path_rejects_phase2_only_source(tmp_path):
    """Source whose only step subdirs are Phase 2 must raise.

    The curriculum needs a Phase-1 checkpoint; if the source has no
    Phase-1 step at all, the helper rejects the path.
    """
    src_dir = tmp_path / "src"
    step_dir = src_dir / "200"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"fake-state")
    (step_dir / "train_meta.json").write_text(
        json.dumps({"resume": {"phase": 2, "epoch_in_phase": 5, "batch_offset": 0,
                               "validation_completed": True}})
    )
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(ValueError, match="no Phase-1 checkpoint"):
        _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))


def test_pretrained_phase1_path_picks_phase1_under_phase2_steps(tmp_path):
    """When the source has both Phase-1 and Phase-2 step dirs interleaved,
    the highest Phase-1 step is selected and copied; Phase-2 dirs are not.

    This is the standard ``made`` case: it trains Phase 1 + Phase 2 in the
    same checkpoints/ tree, leaving Phase-1 step dirs intact alongside
    later Phase-2 ones. The curriculum reuses Phase 1.
    """
    src_dir = tmp_path / "src"
    for step in (62, 124, 186):
        d = src_dir / str(step)
        d.mkdir(parents=True)
        (d / "state.pkl").write_bytes(f"phase1-{step}".encode())
        (d / "train_meta.json").write_text(
            json.dumps({"resume": {"phase": 1, "epoch_in_phase": 0,
                                   "batch_offset": 0, "validation_completed": True}})
        )
    for step in (248, 310):
        d = src_dir / str(step)
        d.mkdir(parents=True)
        (d / "state.pkl").write_bytes(f"phase2-{step}".encode())
        (d / "train_meta.json").write_text(
            json.dumps({"resume": {"phase": 2, "epoch_in_phase": 0,
                                   "batch_offset": 0, "validation_completed": True}})
        )

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))

    # Highest Phase-1 step (186) is copied with its train_meta.
    assert (local_dir / "186" / "state.pkl").exists()
    assert (local_dir / "186" / "train_meta.json").exists()
    # Earlier Phase-1 step dirs are NOT copied — only the chosen one.
    assert not (local_dir / "62").exists()
    assert not (local_dir / "124").exists()
    # Phase-2 step dirs are NOT copied so the local tree is Phase-1-only.
    assert not (local_dir / "248").exists()
    assert not (local_dir / "310").exists()


def test_pretrained_phase1_path_accepts_phase1_source(tmp_path):
    """Source checkpoint at Phase 1 must succeed and copy contents."""
    src_dir = tmp_path / "src"
    step_dir = src_dir / "100"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"real-state")
    (step_dir / "train_meta.json").write_text(
        json.dumps({"resume": {"phase": 1, "epoch_in_phase": 9, "batch_offset": 0,
                               "validation_completed": True}})
    )
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))
    assert (local_dir / "100" / "state.pkl").exists()


def test_pretrained_phase1_path_rejects_empty_source(tmp_path):
    """Source directory with no numeric step subdirectories raises ValueError."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "some_other_file.txt").write_text("not a checkpoint")
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(ValueError, match="no numeric step"):
        _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))


def test_pretrained_phase1_path_rejects_missing_meta(tmp_path):
    """Source step dir without train_meta.json raises ValueError mentioning 'train_meta.json'."""
    src_dir = tmp_path / "src"
    step_dir = src_dir / "100"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"fake-state")
    # Deliberately no train_meta.json
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    with pytest.raises(ValueError, match="train_meta.json"):
        _maybe_copy_pretrained_checkpoint(str(src_dir), str(local_dir))


# ---------------------------------------------------------------------------
# H3 — pretrained_phase1_path in comparability_signature
# ---------------------------------------------------------------------------


def test_pretrained_phase1_path_in_comparability_signature(tmp_path):
    """train_meta.json written by _write_train_meta includes pretrained_phase1_path in sig."""
    from made.training.trainer import _write_train_meta

    cfg = TrainingConfig(pretrained_phase1_path="/some/phase1/checkpoints")
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    # _write_train_meta needs a step subdir to exist first (it writes directly to step/train_meta)
    step_dir = tmp_path / "ckpt" / "0"
    step_dir.mkdir(parents=True)
    _write_train_meta(ckpt, 0, cfg)
    meta_path = step_dir / "train_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    sig = meta.get("comparability_signature", {})
    assert "pretrained_phase1_path" in sig
    assert sig["pretrained_phase1_path"] == "/some/phase1/checkpoints"


def test_pretrained_phase1_path_mismatch_rejected(tmp_path):
    """Resuming with a different pretrained_phase1_path raises ValueError."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
            "pretrained_phase1_path": "/original/path/checkpoints",
        },
    )
    cfg = TrainingConfig(
        validation_interval_epochs=1,
        pretrained_phase1_path="/different/path/checkpoints",
    )
    with pytest.raises(ValueError, match="pretrained_phase1_path"):
        _check_train_meta(ckpt, 0, cfg)


def test_pretrained_phase1_path_old_checkpoint_without_field_loads_cleanly(tmp_path):
    """Old checkpoints without pretrained_phase1_path in signature load under any value."""
    ckpt = CheckpointManager(str(tmp_path / "ckpt"))
    _write_sig(
        tmp_path / "ckpt",
        step=0,
        sig={
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
            # no pretrained_phase1_path key — old checkpoint
        },
    )
    cfg = TrainingConfig(
        validation_interval_epochs=1,
        pretrained_phase1_path="/any/path/checkpoints",
    )
    _check_train_meta(ckpt, 0, cfg)  # must not raise


# ---------------------------------------------------------------------------
# Curriculum meta-rewrite regression test (Phase C4)
# ---------------------------------------------------------------------------


def test_maybe_copy_rewrites_train_meta_for_curriculum(tmp_path):
    """_maybe_copy_pretrained_checkpoint rewrites the copied train_meta.json so that
    the curriculum trainer sees the correct comparability_signature, starts at Phase 2,
    and has a clean Phase-2 early-stopping history.
    """
    src = tmp_path / "src"
    step_dir = src / "1000"
    step_dir.mkdir(parents=True)
    # Minimal state file — helper only checks existence.
    (step_dir / "state.pkl").write_bytes(b"")

    # Source meta: Phase-1 source (not itself a curriculum run).
    source_meta = {
        "comparability_signature": {
            "pretrained_phase1_path": None,
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
        },
        "resume": {
            "phase": 1,
            "epoch_in_phase": 5,
            "batch_offset": 0,
            "validation_completed": True,
        },
        "early_stopping": {
            "1": {
                "best_loss": 0.001,
                "best_step": 800,
                "wait": 2,
                "history": [{"loss": 0.001, "step": 800.0}],
                "reasons": [],
            },
            "2": {
                "best_loss": float("inf"),
                "best_step": None,
                "wait": 0,
                "history": [],
                "reasons": [],
            },
        },
        "early_stop_reasons": [],
        "early_stop_best_loss": 0.001,
        "early_stop_best_step": 800,
        "early_stopping_config": _early_stopping_config_meta(TrainingConfig()),
    }
    (step_dir / "train_meta.json").write_text(json.dumps(source_meta), encoding="utf-8")

    dest = tmp_path / "dest"
    dest.mkdir()

    _maybe_copy_pretrained_checkpoint(str(src), str(dest))

    copied_meta_path = dest / "1000" / "train_meta.json"
    assert copied_meta_path.exists(), "train_meta.json must be present in copied step dir"
    copied = json.loads(copied_meta_path.read_text(encoding="utf-8"))

    # (a) comparability_signature.pretrained_phase1_path patched to str(src).
    sig = copied.get("comparability_signature", {})
    assert sig["pretrained_phase1_path"] == str(src), (
        f"Expected pretrained_phase1_path={str(src)!r}, got {sig['pretrained_phase1_path']!r}"
    )

    # (b) Resume cursor advanced to Phase 2, epoch 0.
    resume = copied["resume"]
    assert resume == {
        "phase": 2,
        "epoch_in_phase": 0,
        "batch_offset": 0,
        "validation_completed": False,
    }, f"Unexpected resume: {resume}"

    # (c) Phase-1 early-stopping record preserved; Phase-2 reset to default.
    es = copied["early_stopping"]
    assert es["1"]["best_step"] == 800, "Phase-1 early-stopping record must be preserved"
    es2 = es["2"]
    assert math.isinf(float(es2["best_loss"])), "Phase-2 best_loss must be inf (default)"
    assert es2["best_step"] is None
    assert es2["wait"] == 0
    assert es2["history"] == []
    assert es2["reasons"] == []

    # (d) Active-phase summary fields reset.
    assert copied["early_stop_best_step"] is None
    assert copied["early_stop_reasons"] == []

    # (e) End-to-end: _check_train_meta accepts the rewritten meta when the
    # curriculum cfg's pretrained_phase1_path matches str(src). Locks the
    # rewrite output to the strict-equality contract enforced at resume.
    cm = CheckpointManager(str(dest))
    cfg = TrainingConfig(pretrained_phase1_path=str(src))
    _check_train_meta(cm, 1000, cfg)  # must not raise


def test_rewrite_phase1_seed_meta_is_idempotent(tmp_path):
    """Calling _rewrite_phase1_seed_meta twice must not clobber Phase-2 progress.

    The second call sees a meta already in the curriculum-seed state
    (signature.pretrained_phase1_path matches and resume.phase != 1) and
    returns False without touching the file.
    """
    src_path = "/some/source/checkpoints"
    step_dir = tmp_path / "1000"
    step_dir.mkdir()
    meta_path = step_dir / "train_meta.json"

    # Curriculum-seed state mid-Phase-2: ten epochs in, with realistic ES history.
    advanced_meta = {
        "comparability_signature": {
            "pretrained_phase1_path": src_path,
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
        },
        "resume": {
            "phase": 2,
            "epoch_in_phase": 10,
            "batch_offset": 0,
            "validation_completed": True,
        },
        "early_stopping": {
            "1": {
                "best_loss": 0.001,
                "best_step": 800,
                "wait": 2,
                "history": [{"loss": 0.001, "step": 800.0}],
                "reasons": [],
            },
            "2": {
                "best_loss": 0.002,
                "best_step": 850,
                "wait": 1,
                "history": [
                    {"loss": 0.005, "step": 820.0},
                    {"loss": 0.004, "step": 830.0},
                    {"loss": 0.003, "step": 840.0},
                    {"loss": 0.002, "step": 850.0},
                    {"loss": 0.0025, "step": 860.0},
                ],
                "reasons": [],
            },
        },
        "early_stop_reasons": [],
        "early_stop_best_loss": 0.002,
        "early_stop_best_step": 850,
        "early_stopping_config": _early_stopping_config_meta(TrainingConfig()),
    }
    meta_path.write_text(json.dumps(advanced_meta), encoding="utf-8")

    # Idempotency: returns False, file unchanged.
    result = _rewrite_phase1_seed_meta(meta_path, src_path)
    assert result is False, "Already-rewritten meta must be a no-op (returns False)"

    after = json.loads(meta_path.read_text(encoding="utf-8"))
    # Resume cursor preserved (Phase-2 progress NOT clobbered).
    assert after["resume"] == {
        "phase": 2,
        "epoch_in_phase": 10,
        "batch_offset": 0,
        "validation_completed": True,
    }, "Resume cursor must be preserved on idempotent call"
    # Phase-2 ES history preserved.
    assert after["early_stopping"]["2"]["best_step"] == 850
    assert len(after["early_stopping"]["2"]["history"]) == 5
    assert after["early_stop_best_step"] == 850


def test_rewrite_phase1_seed_meta_self_heals_stale_partial_seed(tmp_path):
    """Stale Phase-1 source seed (left by a pre-fix failed run) is migrated on demand.

    Mirrors the production scenario: a curriculum cell crashed before its first
    Phase-2 checkpoint, leaving a state.pkl + meta with pretrained_phase1_path=None
    and resume.phase=1. The next train() call's self-heal step calls
    _rewrite_phase1_seed_meta on the highest step's meta, which then satisfies
    _check_train_meta on resume.
    """
    src_path = "/some/source/checkpoints"
    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "1000"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"")

    # Stale source-style meta (the exact production crash state).
    stale_meta = {
        "comparability_signature": {
            "pretrained_phase1_path": None,
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
        },
        "resume": {
            "phase": 1,
            "epoch_in_phase": 5,
            "batch_offset": 0,
            "validation_completed": True,
        },
        "early_stopping": {
            "1": {
                "best_loss": 0.001,
                "best_step": 800,
                "wait": 2,
                "history": [{"loss": 0.001, "step": 800.0}],
                "reasons": [],
            },
            "2": {
                "best_loss": float("inf"),
                "best_step": None,
                "wait": 0,
                "history": [],
                "reasons": [],
            },
        },
        "early_stop_reasons": [],
        "early_stop_best_loss": 0.001,
        "early_stop_best_step": 800,
        "early_stopping_config": _early_stopping_config_meta(TrainingConfig()),
    }
    meta_path = step_dir / "train_meta.json"
    meta_path.write_text(json.dumps(stale_meta), encoding="utf-8")

    result = _rewrite_phase1_seed_meta(meta_path, src_path)
    assert result is True, "Stale source-style meta must be rewritten (returns True)"

    healed = json.loads(meta_path.read_text(encoding="utf-8"))
    assert healed["comparability_signature"]["pretrained_phase1_path"] == src_path
    assert healed["resume"]["phase"] == 2
    assert healed["resume"]["epoch_in_phase"] == 0

    # End-to-end: after self-heal, _check_train_meta must accept the resume.
    cm = CheckpointManager(str(ckpt_dir))
    cfg = TrainingConfig(pretrained_phase1_path=src_path)
    _check_train_meta(cm, 1000, cfg)  # must not raise


def test_self_heal_curriculum_seed_meta_picks_highest_step(tmp_path):
    """When multiple step subdirs exist, self-heal rewrites only the highest one.

    Locks the max(...) selection at trainer.py — a regression to lexicographic
    sort or to min(...) would silently rewrite the wrong meta.
    """
    src_path = "/some/source/checkpoints"
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Build the stale source-style meta to use for both step dirs.
    stale_meta = {
        "comparability_signature": {
            "pretrained_phase1_path": None,
            "validation_interval_epochs": 1,
            "checkpoint_save_interval": None,
            "best_checkpoint_interval_epochs": 1,
        },
        "resume": {
            "phase": 1,
            "epoch_in_phase": 5,
            "batch_offset": 0,
            "validation_completed": True,
        },
        "early_stopping": {
            "1": {"best_loss": 0.001, "best_step": 800, "wait": 0,
                  "history": [], "reasons": []},
            "2": {"best_loss": float("inf"), "best_step": None, "wait": 0,
                  "history": [], "reasons": []},
        },
        "early_stop_reasons": [],
        "early_stop_best_loss": 0.001,
        "early_stop_best_step": 800,
        "early_stopping_config": _early_stopping_config_meta(TrainingConfig()),
    }

    for step in (500, 1000):
        step_dir = ckpt_dir / str(step)
        step_dir.mkdir()
        (step_dir / "state.pkl").write_bytes(b"")
        (step_dir / "train_meta.json").write_text(json.dumps(stale_meta), encoding="utf-8")

    cm = CheckpointManager(str(ckpt_dir))
    _self_heal_curriculum_seed_meta(cm, src_path)

    # Highest (1000) was rewritten.
    high = json.loads((ckpt_dir / "1000" / "train_meta.json").read_text(encoding="utf-8"))
    assert high["resume"]["phase"] == 2
    assert high["comparability_signature"]["pretrained_phase1_path"] == src_path

    # Lower (500) was left as-is.
    low = json.loads((ckpt_dir / "500" / "train_meta.json").read_text(encoding="utf-8"))
    assert low["resume"]["phase"] == 1
    assert low["comparability_signature"]["pretrained_phase1_path"] is None


def test_rewrite_phase1_seed_meta_handles_missing_signature(tmp_path):
    """Source meta lacking comparability_signature is migrated cleanly:
    resume/early_stopping fields are rewritten, signature is left absent,
    and _check_train_meta does not raise (legacy old-checkpoint path).
    """
    src_path = "/some/source/checkpoints"
    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "1000"
    step_dir.mkdir(parents=True)
    (step_dir / "state.pkl").write_bytes(b"")

    no_sig_meta = {
        "resume": {
            "phase": 1,
            "epoch_in_phase": 5,
            "batch_offset": 0,
            "validation_completed": True,
        },
        "early_stopping": {
            "1": {"best_loss": 0.001, "best_step": 800, "wait": 0,
                  "history": [], "reasons": []},
            "2": {"best_loss": float("inf"), "best_step": None, "wait": 0,
                  "history": [], "reasons": []},
        },
        "early_stop_reasons": [],
        "early_stop_best_loss": 0.001,
        "early_stop_best_step": 800,
        "early_stopping_config": _early_stopping_config_meta(TrainingConfig()),
    }
    meta_path = step_dir / "train_meta.json"
    meta_path.write_text(json.dumps(no_sig_meta), encoding="utf-8")

    result = _rewrite_phase1_seed_meta(meta_path, src_path)
    assert result is True

    healed = json.loads(meta_path.read_text(encoding="utf-8"))
    # Signature is left absent (helper only mutates if the key exists).
    assert "comparability_signature" not in healed
    # Resume + early-stopping fields rewritten.
    assert healed["resume"] == {
        "phase": 2, "epoch_in_phase": 0, "batch_offset": 0, "validation_completed": False,
    }
    assert healed["early_stop_best_step"] is None

    # End-to-end: _check_train_meta passes (legacy no-signature path).
    cm = CheckpointManager(str(ckpt_dir))
    cfg = TrainingConfig(pretrained_phase1_path=src_path)
    _check_train_meta(cm, 1000, cfg)  # must not raise
