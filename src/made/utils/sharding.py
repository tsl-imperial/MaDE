"""Helpers for single-device and data-parallel sharding."""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def create_mesh() -> Mesh:
    """Create the project-standard one-dimensional batch mesh."""
    return jax.make_mesh((jax.device_count(),), ("batch",))


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Return the sharding used for batched data."""
    return NamedSharding(mesh, PartitionSpec("batch"))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return the sharding used for replicated model parameters."""
    return NamedSharding(mesh, PartitionSpec())


def shard_batch(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Shard a batch across devices after validating divisibility."""
    batch_axis = mesh.shape["batch"]
    if x.shape[0] % batch_axis != 0:
        raise ValueError(
            f"Batch size {x.shape[0]} must be divisible by device count {batch_axis}."
        )
    return jax.device_put(x, data_sharding(mesh))
