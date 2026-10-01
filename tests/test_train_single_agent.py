# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp

from made.data.grain_pipeline import InMemoryDataSource
from made.utils.config import ExperimentConfig
from scripts.sim import train

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import pytest


def _source(split: str) -> InMemoryDataSource:
    """Build a one-sample in-memory data source tagged with its split name.

    Args:
        split: Split name stored on the source.

    Returns:
        The data source.
    """
    source = InMemoryDataSource(
        [
            {
                "x_prev": jnp.zeros(4, dtype=jnp.float64),
                "x_curr": jnp.zeros(4, dtype=jnp.float64),
                "params": jnp.zeros(0, dtype=jnp.float64),
            }
        ]
    )
    source.split = split
    return source


def test_main_programmatic_uses_val_split_without_dropping_remainder(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: "Path",
) -> None:
    """Check that main programmatic uses val split without dropping remainder."""
    created_sources: list[str] = []
    created_loaders: list[dict] = []
    train_call: dict = {}

    def fake_create_data_source(data_dir: Any, split: str, config: Any) -> InMemoryDataSource:
        """Record the requested split and return a stub source.

        Args:
            data_dir: Ignored data directory.
            split: Requested split name.
            config: Ignored config.

        Returns:
            A stub source tagged with `split`.
        """
        del data_dir, config
        created_sources.append(split)
        return _source(split)

    def fake_create_data_loader(
        source: InMemoryDataSource,
        batch_size: int,
        num_devices: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_remainder: bool = True,
        noise_scale: float = 0.0,
    ) -> SimpleNamespace:
        """Record the loader arguments and return a stub loader.

        Args:
            source: Source the loader reads from.
            batch_size: Requested batch size.
            num_devices: Requested device count.
            shuffle: Requested shuffle flag.
            seed: Requested seed.
            drop_remainder: Requested drop-remainder flag.
            noise_scale: Requested noise scale.

        Returns:
            A stub loader carrying the source split.
        """
        loader = SimpleNamespace(split=source.split)
        created_loaders.append(
            {
                "split": source.split,
                "batch_size": batch_size,
                "num_devices": num_devices,
                "shuffle": shuffle,
                "seed": seed,
                "drop_remainder": drop_remainder,
                "noise_scale": noise_scale,
                "loader": loader,
            }
        )
        return loader

    def fake_train(
        cell: Any,
        train_loader: Any,
        val_loader: Any,
        config: Any,
        checkpoint_manager: Any,
        *,
        mesh: Any = None,
    ) -> object:
        """Record the loader splits and return a stub result.

        Args:
            cell: Ignored cell.
            train_loader: Training loader whose split is recorded.
            val_loader: Validation loader whose split is recorded.
            config: Ignored config.
            checkpoint_manager: Ignored checkpoint manager.
            mesh: Ignored device mesh.

        Returns:
            A placeholder object.
        """
        del cell, config, checkpoint_manager, mesh
        train_call["train_split"] = train_loader.split
        train_call["val_split"] = val_loader.split
        return object()

    monkeypatch.setattr(train, "create_data_source", fake_create_data_source)
    monkeypatch.setattr(train, "create_data_loader", fake_create_data_loader)
    monkeypatch.setattr(train, "init_logging", lambda cfg: None)
    monkeypatch.setattr(train.MaDECell, "from_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "train", fake_train)

    cfg = ExperimentConfig(experiment_name="made", output_dir=str(tmp_path))
    train.main_programmatic(cfg, str(tmp_path), data_dir=str(tmp_path / "data"))

    assert created_sources == ["train", "val"]
    assert train_call == {"train_split": "train", "val_split": "val"}
    assert created_loaders[0]["split"] == "train"
    assert created_loaders[0]["drop_remainder"] is True
    assert created_loaders[0]["shuffle"] is True
    assert created_loaders[1]["split"] == "val"
    assert created_loaders[1]["drop_remainder"] is False
    assert created_loaders[1]["shuffle"] is False
