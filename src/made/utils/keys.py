"""PRNG key helpers."""

import jax


def init_keys(seed: int) -> dict[str, jax.Array]:
    """Initialise the named PRNG streams used across the project."""
    root_key = jax.random.key(seed)
    init_key, data_key, training_key = jax.random.split(root_key, 3)
    return {
        "init": init_key,
        "data": data_key,
        "training": training_key,
    }


def split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a key into the next persistent key and a one-shot subkey."""
    new_key, subkey = jax.random.split(key)
    return new_key, subkey
