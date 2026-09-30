"""MLP baseline smoke tests for the inD pipeline (Plan 1 / inD baseline coverage).

Mirrors ``tests/baselines/test_fab_inD.py`` but exercises the location-aware
metadata path (Plan 1 / inD): the integer ``location_id`` column at index 4
must flow through an embedding lookup before reaching the MLP, the same way
:class:`MetadataEncoder` handles it in MaDE.
"""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.baselines.mlp_baseline import MLPBaseline, train_mlp_baseline
from made.data.ind_data import (
    IND_LOCATION_ID_INDEX,
    IND_METADATA_DIM,
    IND_NUM_LOCATIONS,
    IND_STATE_DIM,
)
from made.utils.config import MLPBaselineConfig


def _make_mlp_inD(key: jax.Array) -> MLPBaseline:
    return MLPBaseline(
        state_dim=IND_STATE_DIM,
        hidden=(16, 16),
        metadata_dim=IND_METADATA_DIM,
        num_locations=IND_NUM_LOCATIONS,
        embedding_dim=8,
        location_id_index=IND_LOCATION_ID_INDEX,
        key=key,
    )


def _ind_metadata(loc_id: int = 1) -> jax.Array:
    """Return a single inD-shaped metadata vector ``[length, width, car=1, truck=0, loc_id]``."""
    return jnp.array([4.5, 1.8, 1.0, 0.0, float(loc_id)], dtype=jnp.float64)


def test_mlp_forward_no_nan_with_inD_metadata():
    """MLP forward through the location-embedded metadata path produces finite output."""
    model = _make_mlp_inD(jax.random.key(0))
    x_prev = jnp.zeros(IND_STATE_DIM, dtype=jnp.float64)
    x_curr = jnp.ones(IND_STATE_DIM, dtype=jnp.float64) * 0.1
    metadata = _ind_metadata(loc_id=2)
    pred_prev, pred_curr = model(x_prev, x_curr, metadata)
    assert pred_prev.shape == (IND_STATE_DIM,)
    assert pred_curr.shape == (IND_STATE_DIM,)
    assert jnp.all(jnp.isfinite(pred_prev))
    assert jnp.all(jnp.isfinite(pred_curr))


def test_mlp_location_embedding_field_exists():
    """When num_locations > 0 the baseline must expose a location_embedding."""
    model = _make_mlp_inD(jax.random.key(1))
    assert model.location_embedding is not None
    assert model.location_embedding.weight.shape == (IND_NUM_LOCATIONS, 8)


def test_mlp_location_embedding_disabled_by_default():
    """Without num_locations the legacy scalar-only path is preserved."""
    model = MLPBaseline(state_dim=IND_STATE_DIM, hidden=(16, 16), key=jax.random.key(2))
    assert model.location_embedding is None


def test_mlp_different_locations_yield_different_outputs():
    """Two metadata vectors that differ only in location_id should produce
    different predictions once the embedding has been initialised."""
    model = _make_mlp_inD(jax.random.key(3))
    x_prev = jnp.zeros(IND_STATE_DIM, dtype=jnp.float64)
    x_curr = jnp.ones(IND_STATE_DIM, dtype=jnp.float64) * 0.1
    out_loc1 = model(x_prev, x_curr, _ind_metadata(loc_id=1))
    out_loc4 = model(x_prev, x_curr, _ind_metadata(loc_id=4))
    # The embedding init has small but non-zero variance — outputs must differ
    # (sanity check that location_id is actually flowing through the network).
    assert not jnp.allclose(out_loc1[0], out_loc4[0])
    assert not jnp.allclose(out_loc1[1], out_loc4[1])


def test_mlp_train_smoke_inD_metadata():
    """train_mlp_baseline runs 2 steps with inD-shaped metadata without NaN."""
    model = _make_mlp_inD(jax.random.key(4))
    batch_size = 4
    metadata = jnp.stack([_ind_metadata(loc_id=(i % IND_NUM_LOCATIONS) + 1)
                          for i in range(batch_size)])
    batch = {
        "x_prev": jax.random.normal(jax.random.key(5),
                                    (batch_size, IND_STATE_DIM), dtype=jnp.float64),
        "x_curr": jax.random.normal(jax.random.key(6),
                                    (batch_size, IND_STATE_DIM), dtype=jnp.float64),
        "metadata": metadata,
    }
    config = MLPBaselineConfig(lr=1e-3, num_epochs=1, steps_per_epoch=2)
    trained = train_mlp_baseline(model, [batch], [], config, key=jax.random.key(7))
    assert isinstance(trained, MLPBaseline)
    pred_prev, pred_curr = trained(
        jnp.zeros(IND_STATE_DIM, dtype=jnp.float64),
        jnp.ones(IND_STATE_DIM, dtype=jnp.float64) * 0.1,
        _ind_metadata(loc_id=1),
    )
    assert jnp.all(jnp.isfinite(pred_prev))
    assert jnp.all(jnp.isfinite(pred_curr))
