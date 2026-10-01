# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tests for resolve_params canonical ordering."""

# ruff: noqa: E402

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from made.physics import PARAMETER_ORDER, resolve_params
from made.data.simulation_data import _true_params_array
from made.utils.config import PhysicsConfig
from scripts.sim.train import _apply_known_params_to_source, _known_param_overrides
from made.utils.config import ExperimentConfig, ModelConfig


def test_kinematic_bicycle_canonical() -> None:
    """Check that kinematic bicycle canonical."""
    result = resolve_params("kinematic_bicycle", {"L": 2.7})
    assert result.shape == (1,)
    assert result.dtype == jnp.float64
    assert float(result[0]) == pytest.approx(2.7)


def test_kinematic_bicycle_default() -> None:
    """Check that kinematic bicycle default."""
    result = resolve_params("kinematic_bicycle", {})
    assert result.shape == (1,)
    assert float(result[0]) == pytest.approx(2.7)


def test_dynamic_bicycle_canonical() -> None:
    """C_f must be first in the dynamic bicycle parameter array."""
    overrides = {"C_f": 1.0, "C_r": 2.0, "m": 3.0, "I_z": 4.0, "l_f": 5.0, "l_r": 6.0}
    result = resolve_params("dynamic_bicycle", overrides)
    assert result.shape == (6,)
    assert result.dtype == jnp.float64
    expected_order = PARAMETER_ORDER["dynamic_bicycle"]
    for i, name in enumerate(expected_order):
        assert float(result[i]) == pytest.approx(overrides[name]), (
            f"Parameter '{name}' at index {i} has wrong value"
        )


def test_dynamic_bicycle_partial_overrides() -> None:
    """Unspecified parameters use defaults, not arbitrary values."""
    result_partial = resolve_params("dynamic_bicycle", {"C_f": 99999.0})
    result_default = resolve_params("dynamic_bicycle", {})
    assert float(result_partial[0]) == pytest.approx(99999.0)
    assert float(result_partial[1]) == pytest.approx(float(result_default[1]))


def test_unknown_key_raises() -> None:
    """Check that unknown key raises."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        resolve_params("kinematic_bicycle", {"mass": 1500.0})


def test_unknown_system_raises() -> None:
    """Check that unknown system raises."""
    with pytest.raises(ValueError, match="Unknown physics system"):
        resolve_params("flying_car", {})


def test_no_params_returns_empty_double_integrator() -> None:
    """Check that no params returns empty double integrator."""
    result = resolve_params("double_integrator", {})
    assert result.shape == (0,)
    assert result.dtype == jnp.float64


def test_no_params_returns_empty_unicycle() -> None:
    """Check that no params returns empty unicycle."""
    result = resolve_params("unicycle", {})
    assert result.shape == (0,)


def test_parameter_order_matches_registry_keys() -> None:
    """PARAMETER_ORDER must cover all 4 systems."""
    for name in ("double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle"):
        assert name in PARAMETER_ORDER


def test_simulation_generation_uses_resolve_params_defaults() -> None:
    """Check that simulation generation uses resolve params defaults."""
    for name in ("kinematic_bicycle", "dynamic_bicycle"):
        generated_params, _ = _true_params_array(PhysicsConfig(true_system=name))
        assert jnp.array_equal(generated_params, resolve_params(name, {}))


def test_known_param_overrides_follow_known_true_semantics() -> None:
    """Check that known param overrides follow known true semantics."""
    fully_specified = ExperimentConfig(
        physics=PhysicsConfig(true_system="kinematic_bicycle", true_params={"L": 3.1})
    )
    assert _known_param_overrides(fully_specified) == {"L": 3.1}

    underspecified = ExperimentConfig(
        physics=PhysicsConfig(true_system="dynamic_bicycle", true_params={"m": 1200.0}),
        model=ModelConfig(known_system="kinematic_bicycle", known_params={"L": 2.8}),
    )
    assert _known_param_overrides(underspecified) == {"L": 2.8}


def test_training_source_params_are_replaced_with_known_params() -> None:
    """Check that training source params are replaced with known params."""
    class Source:
        samples = [{"params": resolve_params("dynamic_bicycle", {})}]

    known_params = resolve_params("kinematic_bicycle", {"L": 2.8})
    source = Source()
    _apply_known_params_to_source(source, known_params)
    assert source.samples[0]["params"].shape == (1,)
    assert jnp.array_equal(source.samples[0]["params"], known_params)
