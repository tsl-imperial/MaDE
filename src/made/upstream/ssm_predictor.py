"""Compact selective state-space (S6) sequence model upstream predictor.

A faithful but small S6 (selective state-space) implementation in Equinox: no
custom CUDA kernel — the selective scan runs as a ``jax.lax.scan`` over the
context window, adequate at this scale (H ≈ 10 steps).

Metadata conditioning enters ONLY through the initial scan state of each block:
``h0 [d_inner, d_state] = init_state_proj(static)``, zero-initialised so an
untrained predictor is metadata-agnostic (the standard S6 ``h0 = 0`` at
init), mirroring the LSTM predictor's zero-initialised ``(h0, c0)`` projection.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from made.upstream.base import UpstreamPredictor
from made.upstream.features import (
    WINDOW_FEATURE_DIM,
    canonicalise_window,
    static_metadata_features,
)


class _SSMBlock(eqx.Module):
    """Pre-norm residual block: LayerNorm → selective SSM mixer → residual add."""

    norm: eqx.nn.LayerNorm
    in_proj: eqx.nn.Linear
    conv: eqx.nn.Conv1d
    x_proj: eqx.nn.Linear
    dt_proj: eqx.nn.Linear
    init_state_proj: eqx.nn.Linear
    A_log: jax.Array
    D: jax.Array
    out_proj: eqx.nn.Linear
    d_inner: int
    d_state: int
    dt_rank: int

    def __init__(
        self,
        *,
        d_model: int,
        static_dim: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_init: float = 0.05,
        key: jax.Array,
    ):
        in_key, conv_key, x_key, dt_key, init_key, out_key = jax.random.split(key, 6)
        d_inner = expand * d_model
        dt_rank = max(1, d_model // 16)

        self.norm = eqx.nn.LayerNorm(d_model)
        self.in_proj = eqx.nn.Linear(d_model, 2 * d_inner, key=in_key)
        # Depthwise causal conv over time: pad (K-1) on the left only.
        self.conv = eqx.nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=conv_kernel,
            groups=d_inner,
            padding=((conv_kernel - 1, 0),),
            key=conv_key,
        )
        self.x_proj = eqx.nn.Linear(d_inner, dt_rank + 2 * d_state, use_bias=False, key=x_key)
        dt_proj = eqx.nn.Linear(dt_rank, d_inner, key=dt_key)
        # Bias so softplus(bias) == dt_init: discretisation steps start at a
        # sane magnitude instead of softplus(0) ≈ 0.69.
        self.dt_proj = eqx.tree_at(
            lambda m: m.bias,
            dt_proj,
            jnp.full_like(dt_proj.bias, math.log(math.expm1(dt_init))),
        )
        init_state_proj = eqx.nn.Linear(static_dim, d_inner * d_state, key=init_key)
        # Zero-init: untrained model starts metadata-agnostic (standard h0 = 0).
        self.init_state_proj = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            init_state_proj,
            (
                jnp.zeros_like(init_state_proj.weight),
                jnp.zeros_like(init_state_proj.bias),
            ),
        )
        # S4D-real initialisation: A_n = -(n + 1) per state channel.
        self.A_log = jnp.log(
            jnp.broadcast_to(jnp.arange(1.0, d_state + 1.0), (d_inner, d_state))
        )
        self.D = jnp.ones(d_inner)
        self.out_proj = eqx.nn.Linear(d_inner, d_model, key=out_key)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

    def __call__(self, x_seq: jax.Array, static: jax.Array) -> jax.Array:
        normed = jax.vmap(self.norm)(x_seq)
        xz = jax.vmap(self.in_proj)(normed)
        x_in, z = xz[:, : self.d_inner], xz[:, self.d_inner :]

        x_conv = self.conv(x_in.T).T
        x_conv = jax.nn.silu(x_conv)

        dbc = jax.vmap(self.x_proj)(x_conv)
        delta = jax.nn.softplus(jax.vmap(self.dt_proj)(dbc[:, : self.dt_rank]))
        b = dbc[:, self.dt_rank : self.dt_rank + self.d_state]
        c = dbc[:, self.dt_rank + self.d_state :]

        a = -jnp.exp(self.A_log)
        a_bar = jnp.exp(delta[:, :, None] * a[None])
        bx = delta[:, :, None] * b[:, None, :] * x_conv[:, :, None]

        h0 = self.init_state_proj(static).reshape(self.d_inner, self.d_state)

        def _step(h, inputs):
            a_t, bx_t = inputs
            h_next = a_t * h + bx_t
            return h_next, h_next

        _, hs = jax.lax.scan(_step, h0, (a_bar, bx))

        y = jnp.einsum("tds,ts->td", hs, c) + self.D[None, :] * x_conv
        y = y * jax.nn.silu(z)
        return x_seq + jax.vmap(self.out_proj)(y)


class SSMPredictor(UpstreamPredictor):
    """Selective-SSM encoder over the context window + MLP delta decoder.

    Input/output contract is identical to ``LSTMPredictor``: raw context
    ``[H, state_dim + metadata_dim]`` in, absolute states ``[horizon,
    state_dim]`` out (cumulative deltas from the last context state).
    """

    input_proj: eqx.nn.Linear
    blocks: tuple[_SSMBlock, ...]
    final_norm: eqx.nn.LayerNorm
    decoder: eqx.nn.MLP
    location_embedding: eqx.nn.Embedding
    state_mean: jax.Array
    state_std: jax.Array
    horizon: int
    metadata_dim: int
    location_id_index: int
    _state_dim: int

    def __init__(
        self,
        *,
        horizon: int,
        state_mean: jax.Array,
        state_std: jax.Array,
        d_model: int = 64,
        num_blocks: int = 2,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        decoder_width: int = 128,
        decoder_depth: int = 2,
        state_dim: int = 4,
        metadata_dim: int = 5,
        num_locations: int = 4,
        embedding_dim: int = 4,
        location_id_index: int = 4,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_blocks + 3)
        input_key, decoder_key, embed_key = keys[0], keys[1], keys[2]
        block_keys = keys[3:]

        self.input_proj = eqx.nn.Linear(WINDOW_FEATURE_DIM, d_model, key=input_key)
        embedding = eqx.nn.Embedding(num_locations, embedding_dim, key=embed_key)
        # Mirror MetadataEncoder's small-init convention for location embeddings.
        self.location_embedding = eqx.tree_at(
            lambda e: e.weight, embedding, embedding.weight * 0.01
        )
        static_dim = (metadata_dim - 1) + embedding_dim
        self.blocks = tuple(
            _SSMBlock(
                d_model=d_model,
                static_dim=static_dim,
                d_state=d_state,
                expand=expand,
                conv_kernel=conv_kernel,
                key=block_keys[i],
            )
            for i in range(num_blocks)
        )
        self.final_norm = eqx.nn.LayerNorm(d_model)
        self.decoder = eqx.nn.MLP(
            in_size=d_model,
            out_size=horizon * state_dim,
            width_size=decoder_width,
            depth=decoder_depth,
            activation=jax.nn.relu,
            key=decoder_key,
        )
        self.state_mean = jnp.asarray(state_mean)
        self.state_std = jnp.asarray(state_std)
        self.horizon = horizon
        self.metadata_dim = metadata_dim
        self.location_id_index = location_id_index
        self._state_dim = state_dim

    def __call__(self, context: jax.Array) -> jax.Array:
        states = context[:, : self._state_dim]
        meta = context[0, self._state_dim :]
        features = canonicalise_window(states, self.state_mean, self.state_std)
        static = static_metadata_features(meta, self.location_embedding, self.location_id_index)

        x = jax.vmap(self.input_proj)(features)
        for block in self.blocks:
            x = block(x, static)
        latent = self.final_norm(x[-1])

        deltas = self.decoder(latent).reshape(self.horizon, self._state_dim)
        return states[-1] + jnp.cumsum(deltas, axis=0)

    @property
    def state_dim(self) -> int:
        return self._state_dim
