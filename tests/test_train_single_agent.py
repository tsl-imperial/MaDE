from types import SimpleNamespace

import jax.numpy as jnp

from made.data.grain_pipeline import InMemoryDataSource
from made.utils.config import ExperimentConfig
from scripts.sim import train


def _source(split: str) -> InMemoryDataSource:
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


def test_main_programmatic_uses_val_split_without_dropping_remainder(monkeypatch, tmp_path):
    created_sources: list[str] = []
    created_loaders: list[dict] = []
    train_call: dict = {}

    def fake_create_data_source(data_dir, split, config):
        del data_dir, config
        created_sources.append(split)
        return _source(split)

    def fake_create_data_loader(
        source,
        batch_size,
        num_devices,
        *,
        shuffle=True,
        seed=0,
        drop_remainder=True,
        noise_scale=0.0,
    ):
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

    def fake_train(cell, train_loader, val_loader, config, checkpoint_manager, *, mesh=None):
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
