"""Round-trip tests for the E05 predictor factory (make -> save -> load)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import assemble_context
from made.upstream.factory import load_predictor, make_predictor, save_predictor

H, F, D, M = 5, 4, 4, 5
STATE_MEAN = jnp.asarray([50.0, 30.0, 0.0, 5.0])
STATE_STD = jnp.asarray([20.0, 15.0, 1.5, 3.0])


def _sample_context(seed: int = 0, location_id: float = 2.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.standard_normal((H, D)), axis=0) + np.array([50.0, 30.0, 0.0, 5.0])
    metadata = np.array([4.5, 2.0, 1.0, 0.0, location_id])
    return assemble_context(jnp.asarray(states), jnp.asarray(metadata))


@pytest.mark.parametrize("kind", ["lstm", "ssm", "transformer"])
def test_round_trip_identical_outputs(kind: str, tmp_path) -> None:
    model = make_predictor(
        kind, horizon=F, state_mean=STATE_MEAN, state_std=STATE_STD, key=jax.random.key(0)
    )
    context = _sample_context()
    before = model(context)

    config_dict = {
        "kind": kind,
        "horizon": F,
        "history": H,
        "stride": 1,
        "dt": 0.2,
        "state_mean": STATE_MEAN,
        "state_std": STATE_STD,
    }
    save_predictor(tmp_path, model, config_dict)

    assert (tmp_path / "predictor.eqx").exists()
    assert (tmp_path / "predictor_config.json").exists()

    loaded, loaded_config = load_predictor(tmp_path)
    after = loaded(context)

    np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=0, atol=1e-12)
    assert loaded_config["kind"] == kind
    assert loaded_config["horizon"] == F
    assert loaded_config["history"] == H
    assert loaded_config["stride"] == 1
    assert loaded_config["dt"] == 0.2
    # State stats round-trip as plain JSON-serialisable lists.
    assert isinstance(loaded_config["state_mean"], list)
    np.testing.assert_allclose(loaded_config["state_mean"], np.asarray(STATE_MEAN))


@pytest.mark.parametrize("kind", ["lstm", "ssm", "transformer"])
def test_round_trip_preserves_vmap_batching(kind: str, tmp_path) -> None:
    model = make_predictor(
        kind, horizon=F, state_mean=STATE_MEAN, state_std=STATE_STD, key=jax.random.key(3)
    )
    save_predictor(
        tmp_path,
        model,
        {
            "kind": kind,
            "horizon": F,
            "history": H,
            "stride": 1,
            "dt": 0.2,
            "state_mean": STATE_MEAN,
            "state_std": STATE_STD,
        },
    )
    loaded, _ = load_predictor(tmp_path)
    batch = jnp.stack([_sample_context(seed) for seed in range(3)])
    predictions = jax.vmap(loaded)(batch)
    assert predictions.shape == (3, F, D)


@pytest.mark.parametrize("kind", ["lstm", "ssm", "transformer"])
def test_make_predictor_overrides_reach_constructor(kind: str) -> None:
    overrides = {"decoder_width": 8, "decoder_depth": 1}
    model = make_predictor(
        kind,
        horizon=F,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        key=jax.random.key(1),
        **overrides,
    )
    out = model(_sample_context())
    assert out.shape == (F, D)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        make_predictor(
            "gru", horizon=F, state_mean=STATE_MEAN, state_std=STATE_STD, key=jax.random.key(0)
        )


def test_save_predictor_absolutises_directory(tmp_path) -> None:
    model = make_predictor(
        "lstm", horizon=F, state_mean=STATE_MEAN, state_std=STATE_STD, key=jax.random.key(0)
    )
    nested = tmp_path / "nested" / "dir"
    save_predictor(
        nested,
        model,
        {
            "kind": "lstm",
            "horizon": F,
            "history": H,
            "stride": 1,
            "dt": 0.2,
            "state_mean": STATE_MEAN,
            "state_std": STATE_STD,
        },
    )
    assert nested.exists()
    assert (nested / "predictor.eqx").exists()
