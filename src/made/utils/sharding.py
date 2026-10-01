# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Helpers for single-device and data-parallel sharding."""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def create_mesh() -> Mesh:
    """Create the project-standard one-dimensional batch mesh.

    Returns:
        Mesh over all devices with axis ``"batch"``.
    """
    return jax.make_mesh((jax.device_count(),), ("batch",))


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Return the sharding used for batched data.

    Args:
        mesh: Device mesh.

    Returns:
        Sharding that splits the leading axis over ``"batch"``.
    """
    return NamedSharding(mesh, PartitionSpec("batch"))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return the sharding used for replicated model parameters.

    Args:
        mesh: Device mesh.

    Returns:
        Fully replicated sharding.
    """
    return NamedSharding(mesh, PartitionSpec())


def shard_batch(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Shard a batch across devices after validating divisibility.

    Args:
        x: Batch array.
        mesh: Device mesh.

    Returns:
        ``x`` placed with the data sharding.

    Raises:
        ValueError: If the batch size is not divisible by the device count.
    """
    batch_axis = mesh.shape["batch"]
    if x.shape[0] % batch_axis != 0:
        raise ValueError(
            f"Batch size {x.shape[0]} must be divisible by device count {batch_axis}."
        )
    return jax.device_put(x, data_sharding(mesh))
