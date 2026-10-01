# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Utility layer."""

from typing import Any

from made.utils.config import (
    CorrectorConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PhysicsConfig,
    TrainingConfig,
    UpstreamConfig,
    from_json,
    load_config,
    override_config,
    save_config,
    to_json,
)
from made.utils.jax_setup import configure
from made.utils.keys import init_keys, split_key
from made.utils.logging import (
    e01_condition,
    finish_logging,
    init_logging,
    log_artifact,
    log_metrics,
    set_local_metrics_path,
)
from made.utils.sharding import create_mesh, data_sharding, replicated_sharding, shard_batch

_CHECKPOINT_EXPORTS = {"CheckpointManager", "TrainState"}


def __getattr__(name: str) -> Any:
    """Lazily import the checkpoint exports so importing ``made.utils`` stays light.

    Args:
        name: Attribute being looked up.

    Returns:
        ``CheckpointManager`` or ``TrainState``.

    Raises:
        AttributeError: If ``name`` is not a lazy export.
    """
    if name in _CHECKPOINT_EXPORTS:
        from made.utils.checkpointing import CheckpointManager, TrainState

        exports = {
            "CheckpointManager": CheckpointManager,
            "TrainState": TrainState,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckpointManager",
    "CorrectorConfig",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PhysicsConfig",
    "TrainState",
    "TrainingConfig",
    "UpstreamConfig",
    "configure",
    "create_mesh",
    "data_sharding",
    "e01_condition",
    "finish_logging",
    "from_json",
    "init_keys",
    "init_logging",
    "load_config",
    "log_artifact",
    "log_metrics",
    "override_config",
    "set_local_metrics_path",
    "replicated_sharding",
    "save_config",
    "shard_batch",
    "split_key",
    "to_json",
]
