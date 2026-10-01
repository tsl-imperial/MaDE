# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

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
    """Enable float64 and pin ``jax_threefry_partitionable`` to ``True``."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_threefry_partitionable", True)
