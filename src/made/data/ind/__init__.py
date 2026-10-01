# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""inD dataset preprocessing and loading pipeline.

Contract: every public symbol on this package must be importable in an env
without JAX. JAX is allowed only inside function bodies. Enforced by
``tests/data/test_ind_jax_free.py``.
"""

from made.data.ind.loader import (
    InDRecordingBundle,
    InDSplitArrays,
    iter_recordings,
    load_ind_split,
    load_manifest,
)
from made.data.ind.preprocess import preprocess_ind
from made.data.ind.validate import ValidationReport, validate_preprocessed

__all__ = [
    "InDRecordingBundle",
    "InDSplitArrays",
    "ValidationReport",
    "iter_recordings",
    "load_ind_split",
    "load_manifest",
    "preprocess_ind",
    "validate_preprocessed",
]
