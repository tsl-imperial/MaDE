# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""PRNG key helpers."""

import jax


def init_keys(seed: int) -> dict[str, jax.Array]:
    """Initialise the named PRNG streams used across the project.

    Args:
        seed: Root seed.

    Returns:
        Dict with ``init``, ``data`` and ``training`` keys.
    """
    root_key = jax.random.key(seed)
    init_key, data_key, training_key = jax.random.split(root_key, 3)
    return {
        "init": init_key,
        "data": data_key,
        "training": training_key,
    }


def split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a key into the next persistent key and a one-shot subkey.

    Args:
        key: Key to split.

    Returns:
        Tuple ``(new_key, subkey)``.
    """
    new_key, subkey = jax.random.split(key)
    return new_key, subkey
