# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Per-step MLP baseline.

Optional location-aware mode: when ``num_locations > 0`` and ``metadata_dim > 0``, the integer
``location_id`` column at ``location_id_index`` is replaced by an ``eqx.nn.Embedding`` lookup
before the metadata is concatenated onto the state. Same convention as
:class:`made.models.encoder.MetadataEncoder` so location embedding behaves the same way across
MaDE and its baselines.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from made.utils.config import MLPBaselineConfig
from made.utils.logging import log_metrics


class MLPBaseline(eqx.Module):
    """Baseline that directly predicts corrected consecutive states."""

    mlp: eqx.nn.MLP
    state_dim: int
    metadata_dim: int
    location_embedding: eqx.nn.Embedding | None
    num_locations: int
    location_id_index: int

    def __init__(
        self,
        state_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        metadata_dim: int = 0,
        *,
        num_locations: int = 0,
        embedding_dim: int = 8,
        location_id_index: int = 4,
        key: jax.Array,
    ) -> None:
        """Build the MLP and, when requested, the location embedding.

        Args:
            state_dim: State dimension.
            hidden: Hidden layer widths.
            metadata_dim: Metadata width; 0 disables metadata.
            num_locations: Number of location ids; 0 disables the embedding.
            embedding_dim: Location embedding size.
            location_id_index: Metadata column holding the location id.
            key: PRNG key.
        """
        self.state_dim = state_dim
        self.metadata_dim = metadata_dim
        self.num_locations = num_locations
        self.location_id_index = location_id_index

        if num_locations > 0 and metadata_dim > 0:
            emb_key, mlp_key = jax.random.split(key)
            emb_weights = jax.random.normal(emb_key, (num_locations, embedding_dim)) * 0.01
            self.location_embedding = eqx.nn.Embedding(
                num_embeddings=num_locations,
                embedding_size=embedding_dim,
                weight=emb_weights,
            )
            effective_metadata_dim = (metadata_dim - 1) + embedding_dim
        else:
            self.location_embedding = None
            mlp_key = key
            effective_metadata_dim = metadata_dim

        width = hidden[0] if hidden else max(state_dim, 1)
        depth = len(hidden)
        self.mlp = eqx.nn.MLP(
            in_size=2 * state_dim + effective_metadata_dim,
            out_size=2 * state_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=mlp_key,
        )

    def _embed_metadata(self, metadata: jax.Array) -> jax.Array:
        """Replace the integer ``location_id`` column with its embedding lookup.

        Args:
            metadata: Metadata vector, shape (metadata_dim,).
        Returns:
            Metadata with the location id column replaced by its embedding.
        """
        if self.location_embedding is None:
            return metadata
        idx = self.location_id_index
        float_cols = jnp.concatenate([metadata[:idx], metadata[idx + 1 :]], axis=0)
        loc_id = jnp.asarray(metadata[idx], dtype=jnp.int32) - 1
        emb = self.location_embedding(loc_id)
        return jnp.concatenate([float_cols, emb], axis=0)

    def __call__(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Predict a corrected state pair.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            metadata: Optional metadata vector.
        Returns:
            Tuple of corrected (x_prev, x_curr).
        """
        inputs = [x_prev, x_curr]
        if metadata is not None:
            inputs.append(self._embed_metadata(metadata))
        output = self.mlp(jnp.concatenate(inputs))
        return output[: self.state_dim], output[self.state_dim :]

    def correct_pair(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply the shared baseline correction protocol.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            metadata: Optional metadata vector.
        Returns:
            Tuple of corrected (x_prev, x_curr).
        """
        return self(x_prev, x_curr, metadata)


def _mlp_batch_loss(model: MLPBaseline, batch: Any) -> jax.Array:
    """Mean squared error to the target pair, over a batch.

    Args:
        model: Model to evaluate.
        batch: Batch dict; perturbed and target keys are optional.
    Returns:
        Scalar loss.
    """
    x_prev = batch.get("x_prev_perturbed", batch["x_prev"])
    x_curr = batch.get("x_curr_perturbed", batch["x_curr"])
    metadata = batch.get("metadata")
    target_prev = batch.get("x_prev_target", batch["x_prev"])
    target_curr = batch.get("x_curr_target", batch["x_curr"])

    if metadata is None:
        pred_prev, pred_curr = jax.vmap(lambda xp, xc: model(xp, xc))(x_prev, x_curr)
    else:
        pred_prev, pred_curr = jax.vmap(model)(x_prev, x_curr, metadata)
    return jnp.mean((pred_prev - target_prev) ** 2) + jnp.mean((pred_curr - target_curr) ** 2)


def _batch_size(batch: dict[str, jax.Array]) -> int:
    """Number of samples in a batch.

    Args:
        batch: Batch with key x_prev.
    Returns:
        Batch size.
    """
    return int(batch["x_prev"].shape[0])


def _weighted_mean_loss(losses: list[tuple[int, float]]) -> float:
    """Sample-weighted mean of per-batch losses.

    Args:
        losses: List of (batch size, loss) pairs.
    Returns:
        Weighted mean, or infinity when there are no samples.
    """
    total_weight = sum(weight for weight, _ in losses)
    if total_weight <= 0:
        return float("inf")
    return sum(weight * loss for weight, loss in losses) / total_weight


def train_mlp_baseline(
    model: MLPBaseline,
    train_loader: Iterable[dict[str, jax.Array]],
    val_loader: Iterable[dict[str, jax.Array]],
    config: MLPBaselineConfig,
    *,
    key: jax.Array,
) -> MLPBaseline:
    """Train the MLP baseline with a simple MSE objective, logging per-step loss to wandb.

    Args:
        model: Model to train.
        train_loader: Iterable of training batches.
        val_loader: Iterable of validation batches; may be empty.
        config: Training hyperparameters.
        key: PRNG key for batch subsampling.
    Returns:
        The trained model.
    """
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31)))
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _step(
        current_model: MLPBaseline,
        current_opt_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[MLPBaseline, optax.OptState, jax.Array]:
        """One optimiser step on a batch.

        Args:
            current_model: Current model.
            current_opt_state: Current optimiser state.
            batch: Training batch.
        Returns:
            Tuple (updated model, updated optimiser state, loss).
        """
        loss, grads = eqx.filter_value_and_grad(_mlp_batch_loss)(current_model, batch)
        updates, next_opt_state = optimizer.update(grads, current_opt_state)
        next_model = eqx.apply_updates(current_model, updates)
        return next_model, next_opt_state, loss

    @eqx.filter_jit
    def _val_step(current_model: MLPBaseline, batch: dict[str, jax.Array]) -> jax.Array:
        """Validation loss on a batch.

        Args:
            current_model: Current model.
            batch: Validation batch.
        Returns:
            Scalar loss.
        """
        return _mlp_batch_loss(current_model, batch)

    train_batches = list(train_loader)
    val_batches = list(val_loader)

    trained = model
    global_step = 0
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in tqdm(range(config.num_epochs), desc="train MLP", unit="epoch"):
        if config.steps_per_epoch is not None and config.steps_per_epoch < len(train_batches):
            chosen = rng.choice(len(train_batches), size=config.steps_per_epoch, replace=False)
            epoch_batches = [train_batches[i] for i in chosen]
        else:
            epoch_batches = train_batches
        for batch in tqdm(epoch_batches, desc=f"epoch {epoch}", leave=False, unit="batch"):
            trained, opt_state, loss = _step(trained, opt_state, batch)
            global_step += 1
            log_metrics({"loss": float(loss)}, step=global_step)

        if val_batches:
            val_losses = [(_batch_size(b), float(_val_step(trained, b))) for b in val_batches]
            val_mean = _weighted_mean_loss(val_losses)
            log_metrics({"val/loss": val_mean}, step=global_step)
            if config.es_patience > 0:
                if val_mean < best_val_loss - config.es_min_delta:
                    best_val_loss = val_mean
                    patience_count = 0
                else:
                    patience_count += 1
                    if (
                        patience_count >= config.es_patience
                        and epoch + 1 >= config.es_min_epochs
                    ):
                        break

    return trained
