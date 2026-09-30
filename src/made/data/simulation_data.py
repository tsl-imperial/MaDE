"""Synthetic trajectory generation and storage."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from made.evaluation.perturbation import add_observation_noise
from made.physics import (
    PARAMETER_ORDER,
    build_system,
    generate_trajectories,
    resolve_params,
)
from made.utils.config import DataConfig, PhysicsConfig

__all__ = [
    "generate_and_save",
    "load_split",
]


def _instantiate_system(system: str):
    if system not in PARAMETER_ORDER:
        raise ValueError(f"Unsupported physics system '{system}'.")
    physics, constraints = build_system(system)
    return physics, constraints, PARAMETER_ORDER[system]


def _true_params_array(physics_config: PhysicsConfig) -> tuple[jax.Array, tuple[str, ...]]:
    _, _, param_order = _instantiate_system(physics_config.true_system)
    return resolve_params(physics_config.true_system, physics_config.true_params), tuple(param_order)


def _save_split(output_dir: Path, split: str, states: jax.Array, controls: jax.Array) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "states.npy", np.asarray(states))
    np.save(split_dir / "controls.npy", np.asarray(controls))


# A fixed, distinct stream id for generation-time observation noise, so its keys cannot collide
# with the split keys drawn from the same root key.
_GENERATION_NOISE_STREAM = 0x4E4F4953  # "NOIS"


def generate_and_save(
    physics_config: PhysicsConfig,
    data_config: DataConfig,
    output_dir: str,
    key: jax.Array,
) -> None:
    """Generate train/val/test splits and persist them to disk."""
    physics, constraints, param_order = _instantiate_system(physics_config.true_system)
    true_params, param_order = _true_params_array(physics_config)
    key_train, key_val, key_test = jax.random.split(key, 3)

    train_states, train_controls = generate_trajectories(
        physics,
        constraints,
        data_config.num_trajectories_train,
        data_config.trajectory_length,
        physics_config.dt,
        key_train,
        true_params,
        data_config.control_profile,
        data_config.control_tau,
        data_config.min_speed,
        control_sample_min=data_config.control_sample_min,
        control_sample_max=data_config.control_sample_max,
    )
    val_states, val_controls = generate_trajectories(
        physics,
        constraints,
        data_config.num_trajectories_val,
        data_config.trajectory_length,
        physics_config.dt,
        key_val,
        true_params,
        data_config.control_profile,
        data_config.control_tau,
        data_config.min_speed,
        control_sample_min=data_config.control_sample_min,
        control_sample_max=data_config.control_sample_max,
    )
    test_states, test_controls = generate_trajectories(
        physics,
        constraints,
        data_config.num_trajectories_test,
        data_config.trajectory_length,
        physics_config.dt,
        key_test,
        true_params,
        data_config.control_profile,
        data_config.control_tau,
        data_config.min_speed,
        control_sample_min=data_config.control_sample_min,
        control_sample_max=data_config.control_sample_max,
    )

    # Observation noise is added to the
    # generated data ON EVERY SPLIT -- train, validation and test alike -- using the same
    # mechanism as elsewhere, `add_observation_noise` with the config's `noise_scale`.
    # **Perturbation stays out**: that job belongs to the upstream predictors.
    #
    # Noise is applied to STATES only, which is what "observation noise" means here, and before
    # the normalisation statistics are computed, so the recorded mean and std describe the data
    # a model actually sees.
    #
    # The three noise keys are derived with `fold_in` rather than by splitting `key` into six.
    # Splitting differently would change `key_train`/`key_val`/`key_test` and therefore the
    # TRAJECTORIES, so every dataset not being regenerated would stop reproducing. With
    # `fold_in`, a run with the flag off is byte-identical to before this change.
    if data_config.add_generation_noise and data_config.noise_scale > 0.0:
        noise_train, noise_val, noise_test = jax.random.split(
            jax.random.fold_in(key, _GENERATION_NOISE_STREAM), 3)
        train_states = add_observation_noise(train_states, data_config.noise_scale, noise_train)
        val_states = add_observation_noise(val_states, data_config.noise_scale, noise_val)
        test_states = add_observation_noise(test_states, data_config.noise_scale, noise_test)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _save_split(root, "train", train_states, train_controls)
    _save_split(root, "val", val_states, val_controls)
    _save_split(root, "test", test_states, test_controls)

    metadata = {
        "physics_config": asdict(physics_config),
        "data_config": asdict(data_config),
        "param_order": list(param_order),
        "state_mean": np.asarray(jnp.mean(train_states, axis=(0, 1))).tolist(),
        "state_std": np.asarray(jnp.std(train_states, axis=(0, 1))).tolist(),
        "control_mean": np.asarray(jnp.mean(train_controls, axis=(0, 1))).tolist(),
        "control_std": np.asarray(jnp.std(train_controls, axis=(0, 1))).tolist(),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_split(data_dir: str, split: str) -> tuple[jax.Array, jax.Array]:
    """Load one simulated split from disk."""
    split_dir = Path(data_dir) / split
    states = np.load(split_dir / "states.npy")
    controls = np.load(split_dir / "controls.npy")
    return (
        jnp.asarray(states, dtype=jnp.float64),
        jnp.asarray(controls, dtype=jnp.float64),
    )
