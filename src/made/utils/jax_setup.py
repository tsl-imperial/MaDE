"""Process-wide JAX configuration for MaDE.

Call ``configure()`` before creating any JAX array or importing anything that
does. It enables float64 (ODE solvers require it) and pins
``jax_threefry_partitionable`` to ``True`` explicitly, so that PRNG streams do
not silently depend on the installed JAX version's default (this flag
defaulted to ``False`` before JAX 0.5 and ``True`` from JAX 0.5 onward); the
paper's results were produced with it set to ``True``.
"""

import jax


def configure() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_threefry_partitionable", True)
