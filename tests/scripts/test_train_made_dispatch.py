"""Default-tier dispatch tests for scripts/ind/train_made._train_variant.

Exercises every legal variant (`made.utils.config._LEGAL_VARIANTS`) plus the unknown-variant
error path. Runs entirely in-process (no subprocess) to stay well under the 60 s SLA.
Budget: ~10 s on CPU + float64 with batch_size=8 and steps_per_epoch=1.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)

import pytest
from dataclasses import replace

from made.utils.config import _LEGAL_VARIANTS, load_config
from scripts.ind.train_made import _build_data_loaders, _train_variant

_CONFIG_PATH = "configs/ind/made.json"

# Sentinels per variant (relative to output_dir)
_SENTINEL_CHECKS: dict[str, list[str]] = {
    "made_phase1": ["checkpoints"],
    "made_phase2": ["checkpoints"],
    "made_no_residual": ["checkpoints"],
    "made_no_corrector": ["checkpoints"],
    "mlp": ["checkpoints/mlp_model.pkl"],
    "fab": ["checkpoints/fab_model.pkl"],
    "clamp": ["checkpoints/.clamp"],
}


def _minimal_config(variant: str):
    """Load the shipped inD config, retag it with `variant`, and shrink to 1-step smoke size.

    made_phase2 overrides pretrained_phase1_path=None so the trainer skips the Phase-1 restore
    and runs from a fresh init. This is intentional: we only need to verify the dispatch fires
    the right code path and writes the sentinel -- full Phase-2 quality is not a default-tier
    obligation.
    """
    config = load_config(_CONFIG_PATH)
    config = replace(
        config,
        variant=variant,
        training=replace(
            config.training,
            batch_size=8,
            steps_per_epoch=1,
            val_steps_per_epoch=1,
            num_epochs_phase1=1,
            num_epochs_phase2=1 if variant == "made_phase2" else 0,
            pretrained_phase1_path=None,  # safe no-op for non-phase2 variants
            # This smoke run is too short for phase 1 to record a best step, so the
            # shipped default (True) would raise; disable the restore for this test.
            restore_phase1_best_before_phase2=False,
        ),
        fab_baseline=replace(
            config.fab_baseline,
            num_epochs=1,
            steps_per_epoch=1,
        ),
        mlp_baseline=replace(
            config.mlp_baseline,
            num_epochs=1,
            steps_per_epoch=1,
        ),
    )
    return config


def _stub_loaders(config, seed: int = 0):
    """Build stub in-memory loaders (no real inD data required)."""
    train_loader, val_loader, _train_states, _train_lengths = _build_data_loaders(
        data_dir="<unused>",
        config=config,
        use_stub=True,
        num_devices=1,
        seed=seed,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Parametrised dispatch test
# ---------------------------------------------------------------------------


_SLOW_VARIANTS = {"made_phase1", "made_phase2", "made_no_residual", "made_no_corrector"}

_VARIANT_PARAMS = [
    pytest.param(v, marks=pytest.mark.slow) if v in _SLOW_VARIANTS else pytest.param(v)
    for v in sorted(_LEGAL_VARIANTS)
]


@pytest.mark.parametrize("variant", _VARIANT_PARAMS)
def test_train_variant_sentinel_and_summary(variant, tmp_path):
    """Each variant (a) writes its checkpoint sentinel and (b) produces a
    training_summary.json with a finite final_loss and the correct variant tag.
    """
    config = _minimal_config(variant)
    config = replace(config, output_dir=str(tmp_path))

    train_loader, val_loader = _stub_loaders(config)

    _train_variant(config, train_loader, val_loader, tmp_path, seed=0)

    # (a) sentinel exists
    for sentinel in _SENTINEL_CHECKS[variant]:
        sentinel_path = tmp_path / sentinel
        if sentinel_path.suffix == "" and not sentinel_path.name.startswith("."):
            # directory sentinel — just needs to exist and be non-empty
            assert sentinel_path.exists(), f"Sentinel dir missing: {sentinel_path}"
            assert any(sentinel_path.iterdir()), f"Sentinel dir empty: {sentinel_path}"
        else:
            assert sentinel_path.exists(), f"Sentinel file missing: {sentinel_path}"

    # (b) training_summary.json
    summary_path = tmp_path / "training_summary.json"
    assert summary_path.exists(), "training_summary.json not written"
    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["variant"] == variant, (
        f"Expected variant={variant!r}, got {summary['variant']!r}"
    )
    assert math.isfinite(summary["final_loss"]), (
        f"final_loss is non-finite for variant={variant!r}: {summary['final_loss']}"
    )


# ---------------------------------------------------------------------------
# Unknown-variant error
# ---------------------------------------------------------------------------


def test_unknown_variant_raises(tmp_path):
    """_train_variant must raise ValueError containing 'Unknown variant' for bogus input."""
    config = load_config(_CONFIG_PATH)
    config = replace(config, variant="bogus", output_dir=str(tmp_path))
    config = replace(
        config,
        training=replace(config.training, batch_size=8, steps_per_epoch=1),
    )
    train_loader, val_loader = _stub_loaders(config)
    with pytest.raises(ValueError, match="Unknown variant"):
        _train_variant(config, train_loader, val_loader, tmp_path, seed=0)


# ---------------------------------------------------------------------------
# Config-payload contract: MaDE-family configs train with state-transition noise
# ---------------------------------------------------------------------------


def test_made_family_configs_have_training_noise() -> None:
    """The shipped inD MaDE config trains with the canonical small Gaussian
    state-transition augmentation (data.noise_scale > 0.0)."""
    cfg = load_config(_CONFIG_PATH)
    assert cfg.data.noise_scale > 0.0, (
        "configs/ind/made.json must have data.noise_scale > 0.0"
    )
