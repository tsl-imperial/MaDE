"""Deterministic data-source and loader helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from made.data.simulation_data import load_split
from made.physics import resolve_params
from made.utils.config import DataConfig

__all__ = [
    "InMemoryDataLoader",
    "InMemoryDataSource",
    "MetadataAttachTransform",
    "TrajectoryWindowTransform",
    "create_data_loader",
    "create_data_source",
]

# Salt to derive the per-epoch noise RNG from the loader seed without colliding with the
# shuffle RNG stream. Arbitrary fixed integer; changing it breaks reproducibility of past
# noisy runs.
_NOISE_SALT: int = 0x9E3779B97F4A7C15  # 64-bit fractional golden ratio


class _MapTransformBase:
    """Fallback base class when Grain is unavailable."""

    def map(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError


class TrajectoryWindowTransform(_MapTransformBase):
    """Extract overlapping `(x_prev, x_curr)` windows from one trajectory."""

    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def map(self, trajectory_data: dict[str, Any]) -> list[dict[str, Any]]:
        states = trajectory_data["states"]
        controls = trajectory_data.get("controls")
        length = int(trajectory_data.get("length", states.shape[0]))
        params = trajectory_data.get("params", jnp.array([], dtype=jnp.float64))
        windows: list[dict[str, Any]] = []
        for idx in range(1, length):
            sample = {
                "x_prev": states[idx - 1],
                "x_curr": states[idx],
                "params": params,
            }
            if controls is not None:
                sample["u_gt"] = controls[idx - 1]
            windows.append(sample)
        return windows


class MetadataAttachTransform(_MapTransformBase):
    """Attach per-trajectory metadata to each emitted window."""

    def map(self, trajectory_data: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = trajectory_data.get("metadata")
        windows = trajectory_data["windows"]
        if metadata is None:
            return windows
        return [dict(window, metadata=metadata) for window in windows]


@dataclass
class InMemoryDataSource:
    """Simple source over an in-memory sample list."""

    samples: list[dict[str, Any]]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


class InMemoryDataLoader:
    """Deterministic batch loader with optional shuffling and per-epoch state noise."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = True,
        noise_scale: float = 0.0,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_remainder = drop_remainder
        self.noise_scale = float(noise_scale)
        self._epoch_counter = 0
        # Pre-stack once into contiguous numpy arrays. Turns each batch into one fancy-index
        # plus one host->device transfer per key instead of one transfer per sample per key
        # per batch: measured 56.9 ms/batch -> 0.267 ms/batch, 213x, values bit-identical.
        self._columns: dict[str, np.ndarray] | None = None
        if samples:
            self._columns = {
                key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0)
                for key in samples[0]
            }

    def __iter__(self) -> Iterable[dict[str, jax.Array]]:
        indices = np.arange(len(self.samples))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        # Per-epoch noise RNG, distinct seed-stream from shuffle RNG so a given
        # (seed, noise_scale) is reproducible even if shuffle is toggled off. Uses numpy
        # (matching shuffle RNG style) while eval-time noise uses jax.random.normal; the two
        # streams are independent and won't match byte-for-byte, but both are N(0,
        # noise_scale^2), which is the relevant guarantee.
        if self.noise_scale > 0.0:
            epoch_idx = self._epoch_counter
            noise_rng = np.random.default_rng(
                np.array([self.seed, epoch_idx, _NOISE_SALT], dtype=np.uint64)
            )
        else:
            noise_rng = None
        self._epoch_counter += 1

        stop = len(indices) - self.batch_size + 1 if self.drop_remainder else len(indices)
        for start in range(0, stop, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if len(batch_indices) == 0:
                continue
            if self._columns is not None:
                batch = {
                    key: jnp.asarray(column[batch_indices])
                    for key, column in self._columns.items()
                }
            else:  # pragma: no cover - empty sample list
                batch_samples = [self.samples[int(index)] for index in batch_indices]
                keys = batch_samples[0].keys()
                batch = {
                    key: jnp.stack(
                        [jnp.asarray(sample[key]) for sample in batch_samples], axis=0
                    )
                    for key in keys
                }
            if noise_rng is not None:
                for k in ("x_prev", "x_curr"):
                    if k in batch:
                        noise = noise_rng.standard_normal(batch[k].shape).astype(np.float64)
                        batch[k] = batch[k] + jnp.asarray(self.noise_scale * noise)
            yield batch


def _simulated_params(data_dir: str) -> jax.Array:
    metadata_path = Path(data_dir) / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    physics_config = payload["physics_config"]
    system = physics_config.get("true_system", physics_config.get("system"))
    return resolve_params(system, physics_config.get("true_params", {}))


def create_data_source(
    data_dir: str,
    split: str,
    config: DataConfig,
) -> InMemoryDataSource:
    """Create a deterministic in-memory source for one dataset split."""
    if config.noise_scale < 0:
        raise ValueError(f"DataConfig.noise_scale must be >= 0, got {config.noise_scale}")
    window_transform = TrajectoryWindowTransform(window_size=2)
    samples: list[dict[str, Any]] = []

    states, controls = load_split(data_dir, split)
    params = _simulated_params(data_dir)
    for idx in range(states.shape[0]):
        trajectory = {
            "states": states[idx],
            "controls": controls[idx],
            "length": int(states.shape[1]),
            "params": params,
        }
        samples.extend(window_transform.map(trajectory))
    return InMemoryDataSource(samples)


def create_data_loader(
    source: InMemoryDataSource,
    batch_size: int,
    num_devices: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
    noise_scale: float = 0.0,
) -> InMemoryDataLoader:
    """Create a deterministic data loader over a source."""
    if batch_size % num_devices != 0:
        raise AssertionError(
            f"Batch size {batch_size} must be divisible by number of devices {num_devices}."
        )
    return InMemoryDataLoader(
        source.samples,
        batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_remainder=drop_remainder,
        noise_scale=noise_scale,
    )
