"""Tests for location-aware MetadataEncoder.

Verifies:
- Location embedding gradient flows end-to-end.
- Location-agnostic mode (num_locations=0) still callable.
- Backward compatibility: old-style construction without location args.
"""

# ruff: noqa: E402
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

jax.config.update("jax_enable_x64", True)

from made.models.encoder import MetadataEncoder


def _make_location_encoder(key: jax.Array) -> MetadataEncoder:
    """Small location-aware encoder matching the inD metadata schema."""
    param_scales = jnp.ones(1, dtype=jnp.float64)  # 1 physics param for test
    return MetadataEncoder(
        metadata_dim=5,       # [length, width, car_oh, truck_bus_oh, location_id]
        param_dim=1,
        hidden=(16,),
        param_scales=param_scales,
        num_locations=4,
        embedding_dim=8,
        location_id_index=4,
        key=key,
    )


def _make_scalar_encoder(key: jax.Array) -> MetadataEncoder:
    """Scalar-only encoder (backward-compat mode, num_locations=0)."""
    param_scales = jnp.ones(1, dtype=jnp.float64)
    return MetadataEncoder(
        metadata_dim=4,  # no location column
        param_dim=1,
        hidden=(16,),
        param_scales=param_scales,
        key=key,
    )


def test_location_encoder_forward_no_nan():
    """Location-aware encoder forward pass produces finite output."""
    enc = _make_location_encoder(jax.random.key(0))
    # metadata: [length, width, car=1, truck=0, location_id=2]
    meta = jnp.array([4.5, 1.8, 1.0, 0.0, 2.0], dtype=jnp.float64)
    out = enc(meta)
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


def test_location_encoder_output_in_range():
    """Output is bounded by param_scales (sigmoid * scales ∈ [0, scales])."""
    enc = _make_location_encoder(jax.random.key(1))
    meta = jnp.array([4.5, 1.8, 1.0, 0.0, 3.0], dtype=jnp.float64)
    out = enc(meta)
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 1.0)  # param_scales = 1.0


def test_location_encoder_gradient_flows():
    """Gradient flows through the location embedding and MLP."""
    enc = _make_location_encoder(jax.random.key(2))
    meta = jnp.array([4.5, 1.8, 1.0, 0.0, 1.0], dtype=jnp.float64)

    def _loss(encoder: MetadataEncoder) -> jax.Array:
        return jnp.sum(encoder(meta) ** 2)

    grads = eqx.filter_grad(_loss)(enc)
    # Embedding gradients should be non-None arrays
    assert grads.embedding is not None
    emb_grads = grads.embedding.weight
    assert emb_grads is not None
    # The embedding row for location_id=1 (→ row 0) should receive gradient.
    assert jnp.any(jnp.abs(emb_grads[0]) > 0.0), (
        "Row 0 of embedding should receive gradient for location_id=1"
    )
    # Other rows (1–3) should be zero (not used in this forward pass).
    assert jnp.all(emb_grads[1:] == 0.0), (
        "Unused embedding rows should have zero gradient"
    )


def test_location_encoder_different_locations_different_output():
    """Different location IDs produce different encoder outputs."""
    enc = _make_location_encoder(jax.random.key(3))
    base = jnp.array([4.5, 1.8, 1.0, 0.0], dtype=jnp.float64)

    outputs = []
    for loc_id in [1.0, 2.0, 3.0, 4.0]:
        meta = jnp.concatenate([base, jnp.array([loc_id])])
        outputs.append(enc(meta))

    # At least some pairs should differ (embeddings initialised differently)
    all_same = all(jnp.allclose(outputs[0], out) for out in outputs[1:])
    # With random init, all four are extremely unlikely to be identical
    assert not all_same, "Different location IDs should produce different outputs"


def test_scalar_encoder_still_callable():
    """Backward-compat scalar encoder (num_locations=0) still works."""
    enc = _make_scalar_encoder(jax.random.key(4))
    meta = jnp.array([4.5, 1.8, 1.0, 0.0], dtype=jnp.float64)
    out = enc(meta)
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


def test_scalar_encoder_no_embedding():
    """Scalar encoder has no embedding (embedding is None)."""
    enc = _make_scalar_encoder(jax.random.key(5))
    assert enc.embedding is None


def test_location_encoder_vmap():
    """Location-aware encoder is vmap-compatible."""
    enc = _make_location_encoder(jax.random.key(6))
    batch_meta = jnp.array(
        [
            [4.5, 1.8, 1.0, 0.0, 1.0],
            [5.0, 2.0, 0.0, 1.0, 2.0],
            [4.0, 1.7, 1.0, 0.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    out = jax.vmap(enc)(batch_meta)
    assert out.shape == (3, 1)
    assert jnp.all(jnp.isfinite(out))


def test_location_encoder_all_four_locations_finite():
    """All four vendor location IDs (1–4) produce finite outputs."""
    enc = _make_location_encoder(jax.random.key(7))
    for loc_id in [1.0, 2.0, 3.0, 4.0]:
        meta = jnp.array([4.5, 1.8, 1.0, 0.0, loc_id], dtype=jnp.float64)
        out = enc(meta)
        assert jnp.all(jnp.isfinite(out)), f"NaN for location_id={loc_id}"
