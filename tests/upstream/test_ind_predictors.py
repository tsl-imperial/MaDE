"""Contract tests for the E05 upstream predictors (LSTM + compact selective-SSM)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from made.data.window_dataset import assemble_context
from made.upstream import LSTMPredictor, SSMPredictor
from made.upstream.ssm_predictor import _SSMBlock

H, F, D, M = 6, 5, 4, 5
STATE_MEAN = jnp.asarray([50.0, 30.0, 0.0, 5.0])
STATE_STD = jnp.asarray([20.0, 15.0, 1.5, 3.0])


def _make_predictor(kind: str, key: jax.Array):
    cls = {"lstm": LSTMPredictor, "ssm": SSMPredictor}[kind]
    return cls(horizon=F, state_mean=STATE_MEAN, state_std=STATE_STD, key=key)


def _sample_context(seed: int = 0, location_id: float = 2.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.standard_normal((H, D)), axis=0) + np.array([50.0, 30.0, 0.0, 5.0])
    metadata = np.array([4.5, 2.0, 1.0, 0.0, location_id])
    return assemble_context(jnp.asarray(states), jnp.asarray(metadata))


@pytest.fixture(params=["lstm", "ssm"])
def predictor_kind(request) -> str:
    return request.param


def test_output_shape_and_finiteness(predictor_kind: str) -> None:
    model = _make_predictor(predictor_kind, jax.random.key(0))
    prediction = model(_sample_context())
    assert prediction.shape == (F, D)
    assert bool(jnp.all(jnp.isfinite(prediction)))
    assert model.state_dim == D


def test_vmap_batching(predictor_kind: str) -> None:
    model = _make_predictor(predictor_kind, jax.random.key(0))
    batch = jnp.stack([_sample_context(seed) for seed in range(3)])
    predictions = jax.vmap(model)(batch)
    assert predictions.shape == (3, F, D)
    single = model(batch[1])
    np.testing.assert_allclose(np.asarray(predictions[1]), np.asarray(single))


def test_construction_is_deterministic(predictor_kind: str) -> None:
    a = _make_predictor(predictor_kind, jax.random.key(7))
    b = _make_predictor(predictor_kind, jax.random.key(7))
    context = _sample_context()
    np.testing.assert_array_equal(np.asarray(a(context)), np.asarray(b(context)))


def test_translation_invariance(predictor_kind: str) -> None:
    """Shifting absolute x,y must shift the prediction by exactly the same offset."""
    model = _make_predictor(predictor_kind, jax.random.key(0))
    context = _sample_context()
    shift = jnp.asarray([100.0, -40.0, 0.0, 0.0] + [0.0] * M)
    shifted = model(context + shift[None, :])
    baseline = model(context)
    np.testing.assert_allclose(
        np.asarray(shifted), np.asarray(baseline + shift[None, :D]), rtol=0, atol=1e-9
    )


def test_metadata_agnostic_at_init(predictor_kind: str) -> None:
    """Zero-initialised state conditioning: untrained outputs ignore metadata."""
    model = _make_predictor(predictor_kind, jax.random.key(0))
    context_a = _sample_context(seed=0, location_id=1.0)
    context_b = _sample_context(seed=0, location_id=4.0)
    context_b = context_b.at[:, D].set(10.2).at[:, D + 1].set(2.5)
    np.testing.assert_array_equal(np.asarray(model(context_a)), np.asarray(model(context_b)))


def test_gradients_flow_but_norm_stats_frozen(predictor_kind: str) -> None:
    model = _make_predictor(predictor_kind, jax.random.key(0))
    context = _sample_context()

    def loss(m):
        return jnp.mean(m(context) ** 2)

    grads = eqx.filter_grad(loss)(model)
    trainable_norm = (
        jnp.linalg.norm(grads.decoder.layers[0].weight)
        + jnp.linalg.norm(
            grads.cell.weight_ih if predictor_kind == "lstm" else grads.blocks[0].A_log
        )
    )
    assert float(trainable_norm) > 0.0
    np.testing.assert_array_equal(np.asarray(grads.state_mean), np.zeros(D))
    np.testing.assert_array_equal(np.asarray(grads.state_std), np.zeros(D))


@pytest.mark.parametrize("location_id", [1.0, 4.0])
def test_boundary_location_ids(predictor_kind: str, location_id: float) -> None:
    model = _make_predictor(predictor_kind, jax.random.key(0))
    prediction = model(_sample_context(location_id=location_id))
    assert bool(jnp.all(jnp.isfinite(prediction)))


def _np_sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _np_silu(v: np.ndarray) -> np.ndarray:
    return v * _np_sigmoid(v)


def _np_softplus(v: np.ndarray) -> np.ndarray:
    return np.logaddexp(v, 0.0)


def _np_layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x)
    # eqx.nn.LayerNorm: population variance (ddof=0), clamped to be non-negative.
    variance = max(np.var(x), 0.0)
    inv = 1.0 / np.sqrt(variance + eps)
    return weight * ((x - mean) * inv) + bias


def _np_causal_depthwise_conv(
    x_in: np.ndarray, conv_weight: np.ndarray, conv_bias: np.ndarray, kernel_size: int
) -> np.ndarray:
    """Mirror ``eqx.nn.Conv1d(groups=d_inner, padding=((K-1, 0),))`` on ``x_in.T``.

    ``x_in`` is ``[T, d_inner]`` (channels-last); ``conv_weight`` is
    ``[d_inner, 1, K]`` and ``conv_bias`` is ``[d_inner, 1]``, matching
    eqx.nn.Conv1d's depthwise (feature_group_count=d_inner) weight/bias layout.
    Left-only padding of ``K - 1`` zeros makes this causal: output row ``t``
    depends only on input rows ``t - (K - 1) .. t``.
    """
    T, d_inner = x_in.shape
    pad = kernel_size - 1
    x_padded = np.concatenate([np.zeros((pad, d_inner)), x_in], axis=0)  # [T + pad, d_inner]
    out = np.zeros((T, d_inner))
    for t in range(T):
        window = x_padded[t : t + kernel_size]  # [K, d_inner]
        for c in range(d_inner):
            out[t, c] = np.sum(conv_weight[c, 0, :] * window[:, c]) + conv_bias[c, 0]
    return out


def _reference_ssm_block(block: _SSMBlock, x_seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Pure Python/numpy reimplementation of ``_SSMBlock.__call__``.

    Mirrors made/upstream/ssm_predictor.py operation-for-operation: pre-norm ->
    in_proj -> split(x_in, z) -> causal depthwise conv -> silu -> x_proj ->
    split(delta_raw, b, c) -> softplus(delta) -> A = -exp(A_log) -> per-step
    selective-scan recurrence h = a_bar * h + bx (h0 from init_state_proj) ->
    y = einsum(hs, c) + D skip -> gate by silu(z) -> out_proj -> residual add.
    """
    T, _d_model = x_seq.shape
    d_inner, d_state, dt_rank = block.d_inner, block.d_state, block.dt_rank

    norm_weight = np.asarray(block.norm.weight)
    norm_bias = np.asarray(block.norm.bias)
    normed = np.stack([_np_layer_norm(x_seq[t], norm_weight, norm_bias) for t in range(T)])

    in_proj_weight = np.asarray(block.in_proj.weight)
    in_proj_bias = np.asarray(block.in_proj.bias)
    xz = normed @ in_proj_weight.T + in_proj_bias  # [T, 2 * d_inner]
    x_in = xz[:, :d_inner]
    z = xz[:, d_inner:]

    conv_weight = np.asarray(block.conv.weight)  # [d_inner, 1, K]
    conv_bias = np.asarray(block.conv.bias)  # [d_inner, 1]
    x_conv_pre = _np_causal_depthwise_conv(x_in, conv_weight, conv_bias, conv_weight.shape[-1])
    x_conv = _np_silu(x_conv_pre)

    x_proj_weight = np.asarray(block.x_proj.weight)  # use_bias=False
    dbc = x_conv @ x_proj_weight.T  # [T, dt_rank + 2 * d_state]

    dt_proj_weight = np.asarray(block.dt_proj.weight)
    dt_proj_bias = np.asarray(block.dt_proj.bias)
    delta = _np_softplus(dbc[:, :dt_rank] @ dt_proj_weight.T + dt_proj_bias)  # [T, d_inner]
    b = dbc[:, dt_rank : dt_rank + d_state]  # [T, d_state]
    c = dbc[:, dt_rank + d_state :]  # [T, d_state]

    a = -np.exp(np.asarray(block.A_log))  # [d_inner, d_state]
    D_skip = np.asarray(block.D)  # [d_inner]

    init_state_proj_weight = np.asarray(block.init_state_proj.weight)
    init_state_proj_bias = np.asarray(block.init_state_proj.bias)
    h0 = (init_state_proj_weight @ static + init_state_proj_bias).reshape(d_inner, d_state)

    hs = np.zeros((T, d_inner, d_state))
    h = h0
    for t in range(T):
        a_bar_t = np.exp(delta[t][:, None] * a)  # [d_inner, d_state]
        bx_t = delta[t][:, None] * b[t][None, :] * x_conv[t][:, None]  # [d_inner, d_state]
        h = a_bar_t * h + bx_t
        hs[t] = h

    y = np.einsum("tds,ts->td", hs, c) + D_skip[None, :] * x_conv
    y = y * _np_silu(z)

    out_proj_weight = np.asarray(block.out_proj.weight)
    out_proj_bias = np.asarray(block.out_proj.bias)
    out = y @ out_proj_weight.T + out_proj_bias

    return x_seq + out


def test_ssm_block_matches_pure_python_reference() -> None:
    """Selective-scan numerics reference: guards the S6 recurrence, the causal
    depthwise conv's left-only padding, and eqx.nn.Conv1d's (channels, length)
    layout against silent algebraic drift that a shape/finiteness test would
    never catch (e.g. a flipped kernel, a right-padded conv, or an off-by-one
    in the scan carry)."""
    d_model, static_dim, d_state, expand, conv_kernel = 4, 8, 2, 2, 2
    block = _SSMBlock(
        d_model=d_model,
        static_dim=static_dim,
        d_state=d_state,
        expand=expand,
        conv_kernel=conv_kernel,
        key=jax.random.key(0),
    )

    rng = np.random.default_rng(0)
    time_steps = 5
    x_seq_np = rng.standard_normal((time_steps, d_model))
    static_np = rng.standard_normal((static_dim,))
    x_seq = jnp.asarray(x_seq_np)
    static = jnp.asarray(static_np)

    expected = np.asarray(block(x_seq, static))
    reference = _reference_ssm_block(block, x_seq_np, static_np)

    np.testing.assert_allclose(reference, expected, atol=1e-10)
