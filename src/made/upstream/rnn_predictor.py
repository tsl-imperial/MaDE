"""LSTM upstream trajectory predictor for the inD real-data experiment."""

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


class LSTMPredictor(UpstreamPredictor):
    """LSTM encoder over the context window + MLP delta decoder.

    Input contract (single sample; batch via ``jax.vmap``):
      ``context``: ``[H, state_dim + metadata_dim]`` — raw (unnormalised) states
      with the static 5-vec inD metadata tiled onto every row
      (see ``made.data.window_dataset.assemble_context``). The trailing metadata
      column carries the vendor location ID in ``{1..num_locations}``.

    Metadata conditioning enters ONLY through the initial recurrent state:
    ``(h0, c0) = init_proj(static)`` with ``tanh`` on the ``h0`` half (an LSTM
    hidden state is ``o ⊙ tanh(c)`` and therefore lives in ``(-1, 1)``). The
    final layer of ``init_proj`` is zero-initialised, so an untrained predictor
    is metadata-agnostic — mirroring the repo's ``init_scale=0.0`` residual
    convention.

    Output: ``[horizon, state_dim]`` absolute states, decoded as cumulative
    deltas from the last context state.
    """

    cell: eqx.nn.LSTMCell
    decoder: eqx.nn.MLP
    init_proj: eqx.nn.MLP
    location_embedding: eqx.nn.Embedding
    state_mean: jax.Array
    state_std: jax.Array
    horizon: int
    hidden_size: int
    metadata_dim: int
    location_id_index: int
    _state_dim: int

    def __init__(
        self,
        *,
        horizon: int,
        state_mean: jax.Array,
        state_std: jax.Array,
        hidden_size: int = 64,
        decoder_width: int = 128,
        decoder_depth: int = 2,
        state_dim: int = 4,
        metadata_dim: int = 5,
        num_locations: int = 4,
        embedding_dim: int = 4,
        location_id_index: int = 4,
        key: jax.Array,
    ):
        cell_key, decoder_key, embed_key, init_key = jax.random.split(key, 4)
        self.cell = eqx.nn.LSTMCell(WINDOW_FEATURE_DIM, hidden_size, key=cell_key)
        embedding = eqx.nn.Embedding(num_locations, embedding_dim, key=embed_key)
        # Mirror MetadataEncoder's small-init convention for location embeddings.
        self.location_embedding = eqx.tree_at(
            lambda e: e.weight, embedding, embedding.weight * 0.01
        )
        static_dim = (metadata_dim - 1) + embedding_dim
        init_proj = eqx.nn.MLP(
            in_size=static_dim,
            out_size=2 * hidden_size,
            width_size=hidden_size,
            depth=1,
            activation=jax.nn.relu,
            key=init_key,
        )
        # Zero-init the output layer: untrained model starts metadata-agnostic.
        self.init_proj = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            init_proj,
            (
                jnp.zeros_like(init_proj.layers[-1].weight),
                jnp.zeros_like(init_proj.layers[-1].bias),
            ),
        )
        self.decoder = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=horizon * state_dim,
            width_size=decoder_width,
            depth=decoder_depth,
            activation=jax.nn.relu,
            key=decoder_key,
        )
        self.state_mean = jnp.asarray(state_mean)
        self.state_std = jnp.asarray(state_std)
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.metadata_dim = metadata_dim
        self.location_id_index = location_id_index
        self._state_dim = state_dim

    def __call__(self, context: jax.Array) -> jax.Array:
        states = context[:, : self._state_dim]
        meta = context[0, self._state_dim :]
        features = canonicalise_window(states, self.state_mean, self.state_std)

        static = static_metadata_features(meta, self.location_embedding, self.location_id_index)
        init_flat = self.init_proj(static)
        init = (
            jnp.tanh(init_flat[: self.hidden_size]),
            init_flat[self.hidden_size :],
        )

        def _step(carry, x):
            return self.cell(x, carry), None

        (h_final, _), _ = jax.lax.scan(_step, init, features)

        deltas = self.decoder(h_final).reshape(self.horizon, self._state_dim)
        return states[-1] + jnp.cumsum(deltas, axis=0)

    @property
    def state_dim(self) -> int:
        return self._state_dim
