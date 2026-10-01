# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Autoencoder-style FAB baseline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from made.utils.config import FABBaselineConfig
from made.utils.logging import log_metrics


class GatingNetwork(eqx.Module):
    """Softmax gating over decoder experts."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        hidden: tuple[int, ...],
        *,
        key: jax.Array,
    ) -> None:
        """Build the gating MLP.

        Args:
            latent_dim: Latent input size.
            num_experts: Number of decoder experts.
            hidden: Hidden layer widths.
            key: PRNG key.
        """
        width = hidden[0] if hidden else max(num_experts, 1)
        depth = len(hidden)
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=num_experts,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        """Return softmax weights over the experts.

        Args:
            z: Latent vector, shape (latent_dim,).
        Returns:
            Expert weights, shape (num_experts,).
        """
        return jax.nn.softmax(self.mlp(z))


class FABBaseline(eqx.Module):
    """Latent-projection autoencoder baseline.

    Carries optional input-normalisation statistics, identity unless `with_input_statistics`
    sets them, so existing checkpoints stay bit-identical.
    """

    encoder: eqx.nn.MLP
    decoders: tuple[eqx.nn.MLP, ...]
    gating: GatingNetwork
    state_dim: int
    latent_dim: int
    latent_radius: float
    # Input normalisation stats, carried into inference. Identity by default (zero mean, unit
    # scale) so an instance never given statistics behaves as before.
    #
    # Static leaves, not arrays: as `jax.Array` fields they would be caught by
    # `eqx.filter(model, eqx.is_array)`, land in the optimiser state, and train as free
    # parameters -- statistics that drift during training are no longer the training set's
    # statistics. As tuples they are static: `filter_jit` closes over them, `apply_updates`
    # cannot touch them.
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 32,
        num_experts: int = 4,
        hidden: tuple[int, ...] = (256, 256),
        latent_radius: float = 1.0,
        *,
        key: jax.Array,
    ) -> None:
        """Build the encoder, decoder experts and gating network.

        Args:
            state_dim: State dimension.
            latent_dim: Latent dimension.
            num_experts: Number of decoder experts.
            hidden: Hidden layer widths.
            latent_radius: Radius of the latent ball.
            key: PRNG key.
        """
        enc_key, gate_key, *decoder_keys = jax.random.split(key, num_experts + 2)
        width = hidden[0] if hidden else max(latent_dim, 1)
        depth = len(hidden)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.latent_radius = latent_radius
        self.encoder = eqx.nn.MLP(
            in_size=2 * state_dim,
            out_size=latent_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=enc_key,
        )
        self.decoders = tuple(
            eqx.nn.MLP(
                in_size=latent_dim,
                out_size=2 * state_dim,
                width_size=width,
                depth=depth,
                activation=jax.nn.relu,
                key=decoder_key,
            )
            for decoder_key in decoder_keys
        )
        self.gating = GatingNetwork(latent_dim, num_experts, hidden, key=gate_key)
        self.input_mean = (0.0,) * (2 * state_dim)
        self.input_scale = (1.0,) * (2 * state_dim)

    def _stats(self) -> tuple[jax.Array, jax.Array]:
        """Mean and scale, tolerating instances that predate the fields.

        Uses `getattr`, not a direct read: `from_checkpoint` unpickles, so an object written
        before these fields existed comes back without them and a direct read raises
        AttributeError. A pre-field checkpoint reads as identity, the behaviour it trained with.

        Returns:
            Tuple (mean, scale), each of shape (2 * state_dim,).
        """
        mean = getattr(self, "input_mean", None)
        scale = getattr(self, "input_scale", None)
        if mean is None or scale is None:
            return jnp.zeros((2 * self.state_dim,)), jnp.ones((2 * self.state_dim,))
        return jnp.asarray(mean), jnp.asarray(scale)

    def normalise(self, raw: jax.Array) -> jax.Array:
        """Raw `[x_prev, x_curr]` -> normalised. The inference half of input normalisation.

        Args:
            raw: Raw concatenated pair, shape (2 * state_dim,).
        Returns:
            Normalised pair.
        """
        mean, scale = self._stats()
        return (raw - mean) / scale

    def denormalise(self, normalised: jax.Array) -> jax.Array:
        """Normalised -> raw.

        The decoder reconstructs the encoder's input, so once inputs are normalised its output
        lives in normalised space. Returning that directly would hand the caller a corrected
        pair in the wrong units.

        Args:
            normalised: Pair in normalised space.
        Returns:
            Pair in raw units.
        """
        mean, scale = self._stats()
        return normalised * scale + mean

    def encode(self, x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
        """Encode a consecutive state pair to a latent vector.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
        Returns:
            Latent vector, shape (latent_dim,).
        """
        return self.encoder(self.normalise(jnp.concatenate([x_prev, x_curr])))

    def decode(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decode a latent vector to a state pair, mixing the experts by gating weights.

        Args:
            z: Latent vector, shape (latent_dim,).
        Returns:
            Tuple (x_prev, x_curr) in raw units.
        """
        weights = self.gating(z)
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs) if eqx.is_array(xs[0]) else xs[0],
            *self.decoders,
        )
        outputs = eqx.filter_vmap(lambda d: d(z))(stacked)
        merged = self.denormalise(jnp.sum(weights[:, None] * outputs, axis=0))
        return merged[: self.state_dim], merged[self.state_dim :]

    def project(self, z: jax.Array) -> jax.Array:
        """Project a latent vector onto the latent ball.

        Args:
            z: Latent vector.
        Returns:
            Latent vector, rescaled if its norm exceeds the radius.
        """
        z_norm = jnp.linalg.norm(z)
        safe_norm = jnp.maximum(z_norm, 1e-12)
        scale = jnp.where(z_norm > self.latent_radius, self.latent_radius / safe_norm, 1.0)
        return z * scale

    def __call__(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Encode, project and decode a state pair.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
        Returns:
            Tuple of corrected (x_prev, x_curr).
        """
        z = self.encode(x_prev, x_curr)
        return self.decode(self.project(z))

    def correct_pair(
        self,
        x_prev: jax.Array,
        x_curr: jax.Array,
        metadata: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply FAB projection through the shared baseline protocol.

        Args:
            x_prev: Previous state.
            x_curr: Current state.
            metadata: Ignored; present for protocol compatibility.
        Returns:
            Tuple of corrected (x_prev, x_curr).
        """
        del metadata
        return self(x_prev, x_curr)

    @classmethod
    def from_checkpoint(cls, path: str) -> "FABBaseline":
        """Load a FABBaseline from a checkpoint directory.

        Expects the checkpoint layout written by ``save_fab_checkpoint``:
        a ``fab_model.pkl`` file inside ``path``.

        Args:
            path: Checkpoint directory.
        Returns:
            The loaded model.
        Raises:
            FileNotFoundError: If no checkpoint file exists under `path`.
            TypeError: If the file does not contain a FABBaseline.
        """
        import pickle
        from pathlib import Path

        pkl_path = Path(path) / "fab_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"FAB checkpoint not found at {pkl_path}. "
                "Run FAB training first."
            )
        try:
            import cloudpickle as _pickle
        except ImportError:
            _pickle = pickle
        with pkl_path.open("rb") as handle:
            model = _pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(
                f"Checkpoint at {path!r} contains a {type(model).__name__}, "
                f"expected {cls.__name__}."
            )
        return model


def save_fab_checkpoint(model: "FABBaseline", path: str) -> None:
    """Persist a trained FABBaseline to ``path/fab_model.pkl``.

    The checkpoint is loadable by ``FABBaseline.from_checkpoint(path)``.

    Args:
        model: Trained model to save.
        path: Output checkpoint directory.
    """
    import pickle
    from pathlib import Path

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import cloudpickle as _pickle
    except ImportError:
        _pickle = pickle
    with (out / "fab_model.pkl").open("wb") as handle:
        _pickle.dump(model, handle)


def _fab_reconstruction_loss(
    model: FABBaseline,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Mean squared reconstruction error over both states of each pair.

    Args:
        model: Model to evaluate.
        batch: Batch with keys x_prev and x_curr.
    Returns:
        Scalar loss.
    """
    x_prev = batch["x_prev"]
    x_curr = batch["x_curr"]
    pred_prev, pred_curr = jax.vmap(model)(x_prev, x_curr)
    return jnp.mean((pred_prev - x_prev) ** 2) + jnp.mean((pred_curr - x_curr) ** 2)


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



# A channel with no variation in the training data would divide by zero. Positive-scale guard
# on the normaliser, not a tolerance or convergence test; fires only where the training set is
# constant along a dimension, where the normalised value is zero either way.
_SCALE_FLOOR = 1e-8


def fab_input_statistics(
    batches: Iterable[dict[str, Any]], state_dim: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-dimension mean and standard deviation of `[x_prev, x_curr]` over TRAINING batches.

    Returned as plain tuples so they can be stored as static module leaves and
    therefore cannot be trained -- see the note on `FABBaseline.input_mean`.

    Args:
        batches: Training batches with keys x_prev and x_curr.
        state_dim: State dimension.
    Returns:
        Tuple (mean, scale) of per-dimension tuples.
    Raises:
        ValueError: If `batches` is empty.
    """
    stacked = []
    for batch in batches:
        x_prev, x_curr = batch["x_prev"], batch["x_curr"]
        stacked.append(np.concatenate(
            [np.asarray(x_prev, dtype=np.float64), np.asarray(x_curr, dtype=np.float64)],
            axis=-1))
    if not stacked:
        raise ValueError("no training batches; cannot compute input statistics")
    flat = np.concatenate(stacked, axis=0).reshape(-1, 2 * state_dim)
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), _SCALE_FLOOR)
    return tuple(float(v) for v in mean), tuple(float(v) for v in scale)


def with_input_statistics(
    model: "FABBaseline", mean: Iterable[float], scale: Iterable[float]
) -> "FABBaseline":
    """Return a copy of `model` carrying `mean`/`scale`.

    Shallow copy with the two fields overwritten:

    - `eqx.tree_at` addresses pytree leaves, and the statistics are static tuples precisely so
      the optimiser cannot reach them; it raises "`where` must use just the PyTree structure".
    - `dataclasses.replace` routes through `__init__`, which takes `state_dim`/`latent_dim`/
      `key` rather than these fields, and would re-initialise the networks from a key.

    `object.__setattr__` is required because the dataclass is frozen.

    Args:
        model: Model to copy.
        mean: Per-dimension mean, length 2 * state_dim.
        scale: Per-dimension scale, length 2 * state_dim.
    Returns:
        Copy of the model carrying the statistics.
    """
    import copy

    updated = copy.copy(model)
    object.__setattr__(updated, "input_mean", tuple(float(v) for v in mean))
    object.__setattr__(updated, "input_scale", tuple(float(v) for v in scale))
    return updated


def train_fab_baseline(
    model: FABBaseline,
    train_loader: Iterable[dict[str, jax.Array]],
    val_loader: Iterable[dict[str, jax.Array]],
    config: FABBaselineConfig,
    *,
    key: jax.Array,
) -> FABBaseline:
    """Train the FAB baseline with reconstruction-first objectives, logging per-step loss to wandb.

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
        current_model: FABBaseline,
        current_opt_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[FABBaseline, optax.OptState, jax.Array]:
        """One optimiser step on a batch.

        Args:
            current_model: Current model.
            current_opt_state: Current optimiser state.
            batch: Training batch.
        Returns:
            Tuple (updated model, updated optimiser state, loss).
        """
        loss, grads = eqx.filter_value_and_grad(_fab_reconstruction_loss)(current_model, batch)
        updates, next_opt_state = optimizer.update(grads, current_opt_state)
        next_model = eqx.apply_updates(current_model, updates)
        return next_model, next_opt_state, loss

    @eqx.filter_jit
    def _val_step(current_model: FABBaseline, batch: dict[str, jax.Array]) -> jax.Array:
        """Validation loss on a batch.

        Args:
            current_model: Current model.
            batch: Validation batch.
        Returns:
            Scalar loss.
        """
        return _fab_reconstruction_loss(current_model, batch)

    train_batches = list(train_loader)
    val_batches = list(val_loader)

    # Statistics from the training split only, computed once, baked into the model before the
    # optimiser is re-initialised over it. Validation batches are not included.
    if config.normalise_inputs:
        mean, scale = fab_input_statistics(train_batches, model.state_dim)
        model = with_input_statistics(model, mean, scale)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    trained = model
    global_step = 0
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in tqdm(range(config.num_epochs), desc="train FAB", unit="epoch"):
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
