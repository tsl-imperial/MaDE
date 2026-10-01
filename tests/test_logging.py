# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for experiment logging metadata."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from made.utils.config import load_config
from made.utils.logging import e01_condition


class _WandbStub:
    def __init__(self) -> None:
        """Initialise an empty stub."""
        self.run = None
        self.init_kwargs: dict | None = None
        self.defined_metrics: list[tuple[tuple, dict]] = []
        self.finish_count = 0

    def init(self, **kwargs: Any) -> None:
        """Record init kwargs and start a fake run.

        Args:
            **kwargs: Init keyword arguments.
        """
        self.init_kwargs = kwargs
        self.run = object()

    def finish(self) -> None:
        """Count the finish call and clear the run."""
        self.finish_count += 1
        self.run = None

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """Record a define_metric call.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.defined_metrics.append((args, kwargs))


def test_e01_condition_labels_dynamic_bicycle_regimes() -> None:
    """Checks e01 condition labels dynamic bicycle regimes."""
    base = load_config("configs/sim/dynbicycle_underspecified_made.json")
    no_residual = load_config(
        "configs/sim/dynbicycle_underspecified_made-no-residual.json"
    )
    fully_specified = load_config("configs/sim/di_made.json")

    # The condition is derived from model.known_system only. Canonical
    # underspecified DB applies training-time noise (data.noise_scale=0.02),
    # but the label collapses to "underspecified" — there is no longer a
    # separate "-noise" suffix.
    assert e01_condition(base) == "underspecified"
    assert e01_condition(no_residual) == "underspecified"
    assert e01_condition(fully_specified) == "fully-specified"


def test_init_logging_uses_condition_in_wandb_name_group_config_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks init logging uses condition in wandb name group config and tags."""
    logging_utils = importlib.import_module("made.utils.logging")
    wandb_stub = _WandbStub()
    monkeypatch.setattr(logging_utils, "wandb", wandb_stub)
    monkeypatch.delenv("MADE_WANDB_RUN_NAME", raising=False)

    cfg = load_config("configs/sim/dynbicycle_underspecified_made-no-residual.json")
    logging_utils.init_logging(cfg, project="made-test")

    assert wandb_stub.init_kwargs is not None
    assert (
        wandb_stub.init_kwargs["name"]
        == "e01-dynamic_bicycle-underspecified-made-no-residual-0"
    )
    assert (
        wandb_stub.init_kwargs["group"]
        == "dynamic_bicycle-underspecified-made-no-residual"
    )
    assert wandb_stub.init_kwargs["config"]["e01_condition"] == "underspecified"
    assert "condition:underspecified" in wandb_stub.init_kwargs["tags"]

