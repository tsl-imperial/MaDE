"""Utility layer."""

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


def __getattr__(name: str):
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
