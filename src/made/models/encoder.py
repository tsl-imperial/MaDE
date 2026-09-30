"""Metadata encoder for system parameters.

Location-aware mode (Plan 1 / inD):
  When ``num_locations > 0``, the encoder expects a metadata vector of shape
  ``[metadata_dim]`` whose column at ``location_id_index`` carries an integer
  location ID in the vendor range ``{1..num_locations}`` (NOT zero-indexed).
  Internally the encoder:
    1. Extracts the integer column and applies ``id - 1`` to convert to a
       zero-indexed row into an ``eqx.nn.Embedding`` table.
    2. Concatenates the embedding with the remaining float columns.
    3. Passes the concatenated vector through the existing MLP.

  Backward-compat: when ``num_locations <= 0`` the ``Embedding`` field is
  ``None`` and the call path is identical to the original scalar-only encoder.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class MetadataEncoder(eqx.Module):
    """Encode agent metadata into bounded physics parameters."""

    mlp: eqx.nn.MLP
    param_scales: jax.Array
    embedding: eqx.nn.Embedding | None
    # Stored as static Python ints so they are never traced
    location_id_index: int
    num_locations: int
    metadata_dim: int
    unbounded_scale: bool
    lref_residual: bool

    def __init__(
        self,
        metadata_dim: int,
        param_dim: int,
        hidden: tuple[int, ...],
        param_scales: jax.Array,
        *,
        num_locations: int = 0,
        embedding_dim: int = 8,
        location_id_index: int = 4,
        unbounded_scale: bool = False,
        lref_residual: bool = False,
        key: jax.Array,
    ):
        self.unbounded_scale = unbounded_scale
        self.lref_residual = lref_residual
        self.metadata_dim = metadata_dim
        self.location_id_index = location_id_index
        self.num_locations = num_locations

        if num_locations > 0:
            emb_key, mlp_key = jax.random.split(key)
            # Small-variance init so day-1 behaviour is near location-agnostic.
            embedding_weights = jax.random.normal(emb_key, (num_locations, embedding_dim)) * 0.01
            self.embedding = eqx.nn.Embedding(
                num_embeddings=num_locations,
                embedding_size=embedding_dim,
                weight=embedding_weights,
            )
            # MLP input: (metadata_dim - 1) float cols + embedding_dim
            mlp_in = (metadata_dim - 1) + embedding_dim
        else:
            self.embedding = None
            mlp_key = key
            mlp_in = metadata_dim

        width = hidden[0] if hidden else max(param_dim, 1)
        depth = len(hidden)
        self.mlp = eqx.nn.MLP(
            in_size=mlp_in,
            out_size=param_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=mlp_key,
        )
        self.param_scales = param_scales

    def __call__(self, metadata: jax.Array) -> jax.Array:
        if self.embedding is not None:
            # Slice off the location-id integer column at location_id_index.
            # Use jnp.concatenate on float slices to keep the operation traceable.
            idx = self.location_id_index
            float_cols = jnp.concatenate(
                [metadata[:idx], metadata[idx + 1 :]],
                axis=0,
            )
            # Convert vendor ID {1..num_locations} to zero-indexed {0..num_locations-1}.
            loc_id = jnp.asarray(metadata[idx], dtype=jnp.int32) - 1
            emb = self.embedding(loc_id)  # shape [embedding_dim]
            inputs = jnp.concatenate([float_cols, emb], axis=0)
        else:
            inputs = metadata
        raw = self.mlp(inputs)
        # L = L_REF + raw: signed, unbounded, no clamp. L_REF is imported from the
        # single constant the scorers use rather than written again here, so the known model and
        # the scored model cannot drift apart.
        if getattr(self, "lref_residual", False):
            from made.evaluation.real_data_eval import L_REF

            return L_REF + raw
        # `getattr` with a default, NOT `self.unbounded_scale`. Checkpoints saved before this
        # field existed deserialise into a MetadataEncoder that has no such field at all, and a
        # direct attribute read raises `AttributeError` on every one of them -- which would break
        # every existing inD checkpoint rather than leaving it bit-identical.
        if getattr(self, "unbounded_scale", False):
            # Positive and UNBOUNDED ABOVE. `param_scales` is deliberately not
            # applied here: multiplying softplus by it would reintroduce a scale this
            # deliberately removes, and on the inD path that scale is 1.0 anyway.
            return jax.nn.softplus(raw)
        return jax.nn.sigmoid(raw) * self.param_scales
