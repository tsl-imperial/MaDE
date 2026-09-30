"""Tests for baseline training loops: logging, early stopping, epoch budgets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from made.baselines.mlp_baseline import (
    MLPBaseline,
    _weighted_mean_loss as _mlp_weighted_mean_loss,
    train_mlp_baseline,
)
from made.baselines.fab_baseline import (
    FABBaseline,
    _weighted_mean_loss as _fab_weighted_mean_loss,
    train_fab_baseline,
)
from made.utils.config import MLPBaselineConfig, FABBaselineConfig


STATE_DIM = 4
BATCH_SIZE = 8


def _pair_batch() -> dict[str, jnp.ndarray]:
    x_prev = jnp.zeros((BATCH_SIZE, STATE_DIM))
    x_curr = jnp.ones((BATCH_SIZE, STATE_DIM)) * 0.1
    return {"x_prev": x_prev, "x_curr": x_curr}


# ------------------------------------------------------------------ MLP tests --


def test_mlp_baseline_logs_train_loss(monkeypatch):
    """Per-step 'loss' key is logged to wandb during MLP training."""
    logged: list[dict] = []
    monkeypatch.setattr("made.baselines.mlp_baseline.log_metrics", lambda m, **kw: logged.append(m))

    model = MLPBaseline(STATE_DIM, hidden=(8,), key=jax.random.key(0))
    config = MLPBaselineConfig(num_epochs=1, es_patience=0)
    batch = _pair_batch()
    train_mlp_baseline(model, [batch], [batch], config, key=jax.random.key(1))

    assert any("loss" in m for m in logged), "expected at least one 'loss' log"
    assert any("val/loss" in m for m in logged), "expected at least one 'val/loss' log"


def test_mlp_baseline_early_stopping(monkeypatch):
    """Early stopping halts MLP training before num_epochs when val/loss stagnates."""
    val_logs: list[float] = []

    def fake_log(metrics: dict, **kw: object) -> None:
        if "val/loss" in metrics:
            val_logs.append(metrics["val/loss"])

    monkeypatch.setattr("made.baselines.mlp_baseline.log_metrics", fake_log)

    model = MLPBaseline(STATE_DIM, hidden=(8,), key=jax.random.key(0))
    # es_min_delta=1e8 makes "improvement" impossible after the first epoch sets best_val_loss.
    config = MLPBaselineConfig(num_epochs=10, es_patience=2, es_min_delta=1e8)
    batch = _pair_batch()
    train_mlp_baseline(model, [batch], [batch], config, key=jax.random.key(1))

    assert len(val_logs) < 10, "early stopping should have fired before 10 epochs"


def test_mlp_baseline_full_budget(monkeypatch):
    """With es_patience=0 (disabled) MLP trains for the full epoch budget."""
    val_logs: list[float] = []

    def fake_log(metrics: dict, **kw: object) -> None:
        if "val/loss" in metrics:
            val_logs.append(metrics["val/loss"])

    monkeypatch.setattr("made.baselines.mlp_baseline.log_metrics", fake_log)

    num_epochs = 3
    model = MLPBaseline(STATE_DIM, hidden=(8,), key=jax.random.key(0))
    config = MLPBaselineConfig(num_epochs=num_epochs, es_patience=0)
    batch = _pair_batch()
    train_mlp_baseline(model, [batch], [batch], config, key=jax.random.key(1))

    assert len(val_logs) == num_epochs


def test_baseline_validation_losses_are_sample_weighted():
    losses = [(512, 1.0), (384, 10.0)]
    expected = (512 * 1.0 + 384 * 10.0) / 896

    assert _mlp_weighted_mean_loss(losses) == expected
    assert _fab_weighted_mean_loss(losses) == expected


# ------------------------------------------------------------------ FAB tests --


def test_fab_baseline_logs_loss(monkeypatch):
    """Per-step 'loss' and per-epoch 'val/loss' keys are logged during FAB training."""
    logged: list[dict] = []
    monkeypatch.setattr("made.baselines.fab_baseline.log_metrics", lambda m, **kw: logged.append(m))

    model = FABBaseline(STATE_DIM, latent_dim=4, num_experts=2, hidden=(8,), key=jax.random.key(0))
    config = FABBaselineConfig(num_epochs=1, es_patience=0)
    batch = _pair_batch()
    train_fab_baseline(model, [batch], [batch], config, key=jax.random.key(1))

    assert any("loss" in m for m in logged)
    assert any("val/loss" in m for m in logged)


def test_fab_baseline_early_stopping(monkeypatch):
    """Early stopping halts FAB training before num_epochs."""
    val_logs: list[float] = []

    def fake_log(metrics: dict, **kw: object) -> None:
        if "val/loss" in metrics:
            val_logs.append(metrics["val/loss"])

    monkeypatch.setattr("made.baselines.fab_baseline.log_metrics", fake_log)

    model = FABBaseline(STATE_DIM, latent_dim=4, num_experts=2, hidden=(8,), key=jax.random.key(0))
    config = FABBaselineConfig(num_epochs=10, es_patience=2, es_min_delta=1e8)
    batch = _pair_batch()
    train_fab_baseline(model, [batch], [batch], config, key=jax.random.key(1))

    assert len(val_logs) < 10
