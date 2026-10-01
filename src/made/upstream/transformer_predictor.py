# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Transformer upstream trajectory predictor for the inD real-data experiment.

Design reference: Giuliari et al. 2020, "Transformer Networks for Trajectory
Forecasting" (arXiv:2003.08111) — encode the observed history with a
Transformer encoder, decode the future with cross-attention over the encoded
history. Deliberate deviation from the paper: Giuliari et al. decode
autoregressively (one future step at a time, feeding each prediction back as
the next decoder input); this predictor instead decodes all ``F`` future
steps in a single shot via ``F`` fixed learned query vectors that cross-attend
to the encoded history once. One-shot decoding keeps the predictor
deterministic (no sampling / no exposure bias from feeding back its own
predictions) and avoids ``F`` sequential decoder calls, matching the
deterministic, single-forward-pass contract of ``LSTMPredictor`` and
``SSMPredictor``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from made.upstream.base import UpstreamPredictor
from made.upstream.features import (
    WINDOW_FEATURE_DIM,
    canonicalise_window,
    static_metadata_features,
)


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jax.Array:
    """Standard (non-trainable) sinusoidal positional encoding, ``[seq_len, d_model]``.

    Even columns get ``sin``, odd columns get ``cos``, at geometrically
    increasing wavelengths — the original Transformer (Vaswani et al. 2017)
    scheme also used by Giuliari et al. 2020's encoder. Handles odd
    ``d_model`` by truncating the trailing ``cos`` column.

    Args:
        seq_len: Sequence length.
        d_model: Model width.
    Returns:
        Encoding, shape (seq_len, d_model).
    """
    positions = jnp.arange(seq_len, dtype=jnp.float64)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2, dtype=jnp.float64) * (-jnp.log(10000.0) / d_model)
    )
    angles = positions * div_term[None, :]
    n_cos = d_model // 2
    pe = jnp.zeros((seq_len, d_model), dtype=jnp.float64)
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles[:, :n_cos]))
    return pe


class _EncoderLayer(eqx.Module):
    """Pre-LN Transformer encoder layer: LN -> self-attention -> residual,
    LN -> MLP -> residual (Xiong et al. 2020 pre-norm placement)."""

    norm1: eqx.nn.LayerNorm
    self_attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ffn: eqx.nn.MLP

    def __init__(
        self, *, d_model: int, num_heads: int, ff_width: int, key: jax.Array
    ) -> None:
        """Build one encoder layer.

        Args:
            d_model: Model width.
            num_heads: Attention heads per layer.
            ff_width: Feed-forward width.
            key: PRNG key.
        """
        attn_key, ffn_key = jax.random.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.self_attn = eqx.nn.MultiheadAttention(num_heads, d_model, key=attn_key)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ffn = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=ff_width,
            depth=1,
            activation=jax.nn.gelu,
            key=ffn_key,
        )

    def __call__(self, x_seq: jax.Array) -> jax.Array:
        """Apply self-attention and the feed-forward block.

        Args:
            x_seq: Sequence, shape (T, d_model).
        Returns:
            Sequence of the same shape.
        """
        normed = jax.vmap(self.norm1)(x_seq)
        attn_out = self.self_attn(normed, normed, normed)
        x = x_seq + attn_out
        normed2 = jax.vmap(self.norm2)(x)
        ffn_out = jax.vmap(self.ffn)(normed2)
        return x + ffn_out


class TransformerPredictor(UpstreamPredictor):
    """Encoder-decoder Transformer over the context window (Giuliari et al. 2020 style).

    Input/output contract is identical to ``LSTMPredictor``/``SSMPredictor``: raw
    context ``[H, state_dim + metadata_dim]`` in, absolute states ``[horizon,
    state_dim]`` out (cumulative deltas from the last context state).

    Pipeline: per-step features (``canonicalise_window``) are linearly embedded
    to ``d_model`` and summed with a sinusoidal positional encoding, then passed
    through ``num_layers`` pre-LN Transformer encoder layers
    (``eqx.nn.MultiheadAttention`` self-attention + MLP). Decoding is one-shot:
    ``horizon`` learned query vectors cross-attend once to the encoded history,
    and a per-query MLP head emits per-step state deltas that are cumulatively
    summed from the last context state (see module docstring for why this
    deviates from Giuliari et al.'s autoregressive decoder).

    Metadata conditioning enters ONLY through an additive, zero-initialised
    projection onto the decoder queries: ``query_cond = query_cond_proj(static)``
    reshaped to ``[horizon, d_model]`` and added to the learned base queries.
    The projection's output layer is zero-initialised, so an untrained predictor
    is metadata-agnostic — mirroring ``LSTMPredictor``'s zero-init ``init_proj``
    and ``SSMPredictor``'s zero-init ``init_state_proj`` convention.
    """

    input_proj: eqx.nn.Linear
    encoder_layers: tuple[_EncoderLayer, ...]
    encoder_norm: eqx.nn.LayerNorm
    decoder_queries: jax.Array
    query_cond_proj: eqx.nn.Linear
    cross_attn: eqx.nn.MultiheadAttention
    head: eqx.nn.MLP
    location_embedding: eqx.nn.Embedding
    state_mean: jax.Array
    state_std: jax.Array
    horizon: int
    d_model: int
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
        num_layers: int = 2,
        num_heads: int = 4,
        ff_width: int = 128,
        decoder_width: int = 128,
        decoder_depth: int = 2,
        state_dim: int = 4,
        metadata_dim: int = 5,
        num_locations: int = 4,
        embedding_dim: int = 4,
        location_id_index: int = 4,
        key: jax.Array,
    ) -> None:
        # Two independent splits (rather than one `num_layers + N` split) so the
        # named-key count and the layer-key count can never drift out of sync.
        """Build the Transformer encoder and delta decoder.

        Args:
            horizon: Number of future steps to predict.
            state_mean: Train-split state mean, shape (4,).
            state_std: Train-split state std, shape (4,).
            d_model: Model width.
            num_layers: Number of encoder layers.
            num_heads: Attention heads per layer.
            ff_width: Feed-forward width.
            decoder_width: Hidden width of the delta decoder.
            decoder_depth: Hidden depth of the delta decoder.
            state_dim: State dimension.
            metadata_dim: Metadata width.
            num_locations: Number of location ids.
            embedding_dim: Location embedding size.
            location_id_index: Metadata column holding the location id.
            key: PRNG key.
        """
        named_key, layers_key = jax.random.split(key, 2)
        input_key, embed_key, query_key, query_cond_key, cross_key, head_key = jax.random.split(
            named_key, 6
        )
        layer_keys = jax.random.split(layers_key, num_layers)

        self.input_proj = eqx.nn.Linear(WINDOW_FEATURE_DIM, d_model, key=input_key)
        embedding = eqx.nn.Embedding(num_locations, embedding_dim, key=embed_key)
        # Mirror MetadataEncoder's small-init convention for location embeddings.
        self.location_embedding = eqx.tree_at(
            lambda e: e.weight, embedding, embedding.weight * 0.01
        )
        self.encoder_layers = tuple(
            _EncoderLayer(d_model=d_model, num_heads=num_heads, ff_width=ff_width, key=layer_keys[i])
            for i in range(num_layers)
        )
        self.encoder_norm = eqx.nn.LayerNorm(d_model)

        self.decoder_queries = jax.random.normal(query_key, (horizon, d_model)) * 0.02

        static_dim = (metadata_dim - 1) + embedding_dim
        query_cond_proj = eqx.nn.Linear(static_dim, horizon * d_model, key=query_cond_key)
        # Zero-init the output layer: untrained model starts metadata-agnostic.
        self.query_cond_proj = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            query_cond_proj,
            (
                jnp.zeros_like(query_cond_proj.weight),
                jnp.zeros_like(query_cond_proj.bias),
            ),
        )

        self.cross_attn = eqx.nn.MultiheadAttention(num_heads, d_model, key=cross_key)
        self.head = eqx.nn.MLP(
            in_size=d_model,
            out_size=state_dim,
            width_size=decoder_width,
            depth=decoder_depth,
            activation=jax.nn.relu,
            key=head_key,
        )

        self.state_mean = jnp.asarray(state_mean)
        self.state_std = jnp.asarray(state_std)
        self.horizon = horizon
        self.d_model = d_model
        self.metadata_dim = metadata_dim
        self.location_id_index = location_id_index
        self._state_dim = state_dim

    def __call__(self, context: jax.Array) -> jax.Array:
        """Predict future states from a context window.

        Args:
            context: Context window, shape (H, 4).
        Returns:
            Predicted states, shape (horizon, 4).
        """
        states = context[:, : self._state_dim]
        meta = context[0, self._state_dim :]
        features = canonicalise_window(states, self.state_mean, self.state_std)
        static = static_metadata_features(meta, self.location_embedding, self.location_id_index)

        tokens = jax.vmap(self.input_proj)(features)
        tokens = tokens + _sinusoidal_positional_encoding(tokens.shape[0], self.d_model)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        encoded = jax.vmap(self.encoder_norm)(tokens)

        query_cond = self.query_cond_proj(static).reshape(self.horizon, self.d_model)
        queries = self.decoder_queries + query_cond

        decoded = self.cross_attn(queries, encoded, encoded)
        deltas = jax.vmap(self.head)(decoded)
        return states[-1] + jnp.cumsum(deltas, axis=0)

    @property
    def state_dim(self) -> int:
        """State dimension of the predictions.

        Returns:
            State dimension.
        """
        return self._state_dim
