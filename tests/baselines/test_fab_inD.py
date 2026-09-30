"""FAB baseline smoke tests for the inD pipeline.

Tests:
- FAB forward pass on a tiny batch.
- save_fab_checkpoint / FABBaseline.from_checkpoint round-trip.
- train_fab_baseline smoke run (2 steps, no NaN).
"""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

jax.config.update("jax_enable_x64", True)

from made.baselines.fab_baseline import FABBaseline, save_fab_checkpoint, train_fab_baseline
from made.utils.config import FABBaselineConfig


def _make_fab(key: jax.Array, state_dim: int = 4) -> FABBaseline:
    return FABBaseline(
        state_dim=state_dim,
        latent_dim=8,
        num_experts=2,
        hidden=(16, 16),
        latent_radius=1.0,
        key=key,
    )


def test_fab_forward_no_nan():
    """FAB forward pass produces finite output."""
    model = _make_fab(jax.random.key(0))
    x_prev = jnp.zeros(4, dtype=jnp.float64)
    x_curr = jnp.ones(4, dtype=jnp.float64) * 0.1
    pred_prev, pred_curr = model(x_prev, x_curr)
    assert pred_prev.shape == (4,)
    assert pred_curr.shape == (4,)
    assert jnp.all(jnp.isfinite(pred_prev))
    assert jnp.all(jnp.isfinite(pred_curr))


def test_fab_checkpoint_roundtrip(tmp_path):
    """save_fab_checkpoint / FABBaseline.from_checkpoint round-trips all leaves."""
    model = _make_fab(jax.random.key(1))
    save_fab_checkpoint(model, str(tmp_path / "fab_ckpt"))
    restored = FABBaseline.from_checkpoint(str(tmp_path / "fab_ckpt"))
    assert isinstance(restored, FABBaseline)
    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    rest_leaves = jax.tree_util.tree_leaves(eqx.filter(restored, eqx.is_array))
    assert len(orig_leaves) == len(rest_leaves)
    for a, b in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(a, b), "FAB checkpoint leaves differ after round-trip"


def test_fab_from_checkpoint_raises_on_missing(tmp_path):
    """from_checkpoint raises FileNotFoundError when checkpoint is absent."""
    with pytest.raises(FileNotFoundError):
        FABBaseline.from_checkpoint(str(tmp_path / "no_such_dir"))


def test_fab_train_smoke():
    """train_fab_baseline runs 2 steps without NaN."""
    model = _make_fab(jax.random.key(2))
    batch_size = 8
    state_dim = 4
    batch = {
        "x_prev": jax.random.normal(jax.random.key(3), (batch_size, state_dim), dtype=jnp.float64),
        "x_curr": jax.random.normal(jax.random.key(4), (batch_size, state_dim), dtype=jnp.float64),
    }
    config = FABBaselineConfig(lr=1e-3, num_epochs=1, steps_per_epoch=2)
    trained = train_fab_baseline(
        model,
        [batch],
        [],
        config,
        key=jax.random.key(5),
    )
    assert isinstance(trained, FABBaseline)
    x_prev = jnp.zeros(state_dim, dtype=jnp.float64)
    x_curr = jnp.ones(state_dim, dtype=jnp.float64) * 0.1
    pred_prev, pred_curr = trained(x_prev, x_curr)
    assert jnp.all(jnp.isfinite(pred_prev))
    assert jnp.all(jnp.isfinite(pred_curr))
