"""Contract tests for the E05 Transformer upstream predictor."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import assemble_context
from made.upstream.transformer_predictor import TransformerPredictor

H, F, D, M = 6, 5, 4, 5
STATE_MEAN = jnp.asarray([50.0, 30.0, 0.0, 5.0])
STATE_STD = jnp.asarray([20.0, 15.0, 1.5, 3.0])


def _make_predictor(key: jax.Array) -> TransformerPredictor:
    return TransformerPredictor(
        horizon=F,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_width=32,
        decoder_width=32,
        decoder_depth=2,
        key=key,
    )


def _sample_context(seed: int = 0, location_id: float = 2.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.standard_normal((H, D)), axis=0) + np.array([50.0, 30.0, 0.0, 5.0])
    metadata = np.array([4.5, 2.0, 1.0, 0.0, location_id])
    return assemble_context(jnp.asarray(states), jnp.asarray(metadata))


def test_output_shape_and_finiteness() -> None:
    model = _make_predictor(jax.random.key(0))
    prediction = model(_sample_context())
    assert prediction.shape == (F, D)
    assert bool(jnp.all(jnp.isfinite(prediction)))
    assert model.state_dim == D


def test_vmap_batching_matches_per_sample_loop() -> None:
    model = _make_predictor(jax.random.key(0))
    batch = jnp.stack([_sample_context(seed) for seed in range(3)])
    predictions = jax.vmap(model)(batch)
    assert predictions.shape == (3, F, D)
    for i in range(3):
        single = model(batch[i])
        np.testing.assert_allclose(np.asarray(predictions[i]), np.asarray(single))


def test_construction_is_deterministic() -> None:
    a = _make_predictor(jax.random.key(7))
    b = _make_predictor(jax.random.key(7))
    context = _sample_context()
    np.testing.assert_array_equal(np.asarray(a(context)), np.asarray(b(context)))


def test_two_calls_are_identical() -> None:
    """Deterministic inference: no dropout / sampling at call time."""
    model = _make_predictor(jax.random.key(0))
    context = _sample_context()
    first = model(context)
    second = model(context)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_translation_invariance() -> None:
    """Shifting absolute x,y must shift the prediction by exactly the same offset."""
    model = _make_predictor(jax.random.key(0))
    context = _sample_context()
    shift = jnp.asarray([100.0, -40.0, 0.0, 0.0] + [0.0] * M)
    shifted = model(context + shift[None, :])
    baseline = model(context)
    np.testing.assert_allclose(
        np.asarray(shifted), np.asarray(baseline + shift[None, :D]), rtol=0, atol=1e-9
    )


def test_metadata_agnostic_at_init() -> None:
    """Zero-initialised query conditioning: untrained outputs ignore metadata."""
    model = _make_predictor(jax.random.key(0))
    context_a = _sample_context(seed=0, location_id=1.0)
    context_b = _sample_context(seed=0, location_id=4.0)
    context_b = context_b.at[:, D].set(10.2).at[:, D + 1].set(2.5)
    np.testing.assert_array_equal(np.asarray(model(context_a)), np.asarray(model(context_b)))


def test_gradients_flow_but_norm_stats_frozen() -> None:
    model = _make_predictor(jax.random.key(0))
    context = _sample_context()

    def loss(m):
        return jnp.mean(m(context) ** 2)

    grads = eqx.filter_grad(loss)(model)
    trainable_norm = jnp.linalg.norm(grads.head.layers[0].weight) + jnp.linalg.norm(
        grads.encoder_layers[0].ffn.layers[0].weight
    )
    assert float(trainable_norm) > 0.0
    np.testing.assert_array_equal(np.asarray(grads.state_mean), np.zeros(D))
    np.testing.assert_array_equal(np.asarray(grads.state_std), np.zeros(D))


@pytest.mark.parametrize("location_id", [1.0, 4.0])
def test_boundary_location_ids(location_id: float) -> None:
    model = _make_predictor(jax.random.key(0))
    prediction = model(_sample_context(location_id=location_id))
    assert bool(jnp.all(jnp.isfinite(prediction)))


def test_single_encoder_layer_constructs_and_runs() -> None:
    """Regression test: a key-split off-by-one previously left `layer_keys` one
    key short, which raised IndexError at num_layers=1 (and silently duplicated
    the last two layers' init keys at the default num_layers=2)."""
    model = TransformerPredictor(
        horizon=F,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        d_model=16,
        num_layers=1,
        num_heads=4,
        ff_width=32,
        decoder_width=32,
        decoder_depth=2,
        key=jax.random.key(0),
    )
    prediction = model(_sample_context())
    assert prediction.shape == (F, D)
    assert bool(jnp.all(jnp.isfinite(prediction)))


def test_encoder_layers_are_independently_initialised() -> None:
    """With num_layers >= 2, distinct layers must not share an init key."""
    model = _make_predictor(jax.random.key(0))
    assert len(model.encoder_layers) >= 2
    weight_0 = np.asarray(model.encoder_layers[0].self_attn.query_proj.weight)
    weight_1 = np.asarray(model.encoder_layers[1].self_attn.query_proj.weight)
    assert not np.array_equal(weight_0, weight_1)
