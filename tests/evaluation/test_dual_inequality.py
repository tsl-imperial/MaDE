# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Unit tests for compute_inequality_dual.

Covers:
- Degenerate case: when the EmpiricalEnvelope matches inD_physical_constraints()
  bounds exactly (non-inf dims), the physical and envelope metrics agree to 1e-12.
- Sanity: compute_inequality_dual returns the expected 4 keys.
- Non-degenerate: physical and envelope metrics can diverge when bounds differ.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from made.evaluation.metrics import (
    EmpiricalEnvelope,
    compute_inequality_dual,
)
from made.physics import inD_physical_constraints


# Helpers

def _make_feasible_xu(M: int = 10) -> tuple[jax.Array, jax.Array]:
    """Return (x, u) that satisfy inD_physical_constraints with zero violations.

    Args:
        M: Number of samples.

    Returns:
        State and control arrays.
    """
    # x: [x, y, theta, v] — all within physical limits; x,y are unbounded so any value ok
    x = jnp.column_stack([
        jnp.zeros(M),          # x position (unbounded)
        jnp.zeros(M),          # y position (unbounded)
        jnp.linspace(-1.0, 1.0, M),   # heading ∈ (-pi, pi)
        jnp.linspace(1.0, 10.0, M),   # speed ∈ (0, 22)
    ])
    # u: [delta, a] — within control limits
    u = jnp.column_stack([
        jnp.linspace(-0.3, 0.3, M),   # steer ∈ (-0.5, 0.5)
        jnp.linspace(-4.0, 3.0, M),   # accel ∈ (-8, 4)
    ])
    return x, u


def _make_violating_xu(M: int = 10) -> tuple[jax.Array, jax.Array]:
    """Return (x, u) that violate some physical bounds.

    Args:
        M: Number of samples.

    Returns:
        State and control arrays.
    """
    x, u = _make_feasible_xu(M)
    # Push first row's speed above v_max=22
    x = x.at[0, 3].set(25.0)
    return x, u


def _make_envelope_matching_physical() -> EmpiricalEnvelope:
    """Build EmpiricalEnvelope that matches inD_physical_constraints finite bounds.

    Physical non-inf bounds:
      state: theta ∈ [-pi, pi], v ∈ [0, 22]
      control: delta ∈ [-0.5, 0.5], a ∈ [-8, 4]

    We set x,y envelope to ±inf so envelope_constraint strips them identically.

    Returns:
        Envelope matching the physical bounds.
    """
    return EmpiricalEnvelope(
        state_min=jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, 0.0]),
        state_max=jnp.array([jnp.inf, jnp.inf, jnp.pi, 22.0]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )


# Tests

def test_compute_inequality_dual_returns_expected_keys() -> None:
    """compute_inequality_dual must return all 4 expected keys."""
    x, u = _make_feasible_xu(10)
    env = _make_envelope_matching_physical()
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, env, physical)
    expected_keys = {
        "inequality_violation_rate_physical",
        "inequality_violation_magnitude_physical",
        "inequality_violation_rate_envelope",
        "inequality_violation_magnitude_envelope",
        # Physical-set state/control breakdown, emitted only when u is not None.
        "inequality_violation_rate_physical_state",
        "inequality_violation_magnitude_physical_state",
        "inequality_violation_rate_physical_control",
        "inequality_violation_magnitude_physical_control",
    }
    assert set(result.keys()) == expected_keys, (
        f"unexpected keys: {set(result.keys()) ^ expected_keys}"
    )


def test_compute_inequality_dual_values_are_python_floats() -> None:
    """Values must be plain Python floats (not jax arrays)."""
    x, u = _make_feasible_xu(10)
    env = _make_envelope_matching_physical()
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, env, physical)
    for k, v in result.items():
        assert isinstance(v, float), f"key {k!r}: expected float, got {type(v)}"


def test_degenerate_envelope_equals_physical() -> None:
    """When envelope bounds match physical bounds, physical == envelope metrics to 1e-12."""
    x, u = _make_violating_xu(10)
    env = _make_envelope_matching_physical()
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, env, physical)
    assert result["inequality_violation_rate_physical"] == pytest.approx(
        result["inequality_violation_rate_envelope"], abs=1e-12
    ), (
        f"rate mismatch: physical={result['inequality_violation_rate_physical']}, "
        f"envelope={result['inequality_violation_rate_envelope']}"
    )
    assert result["inequality_violation_magnitude_physical"] == pytest.approx(
        result["inequality_violation_magnitude_envelope"], abs=1e-12
    ), (
        f"magnitude mismatch: physical={result['inequality_violation_magnitude_physical']}, "
        f"envelope={result['inequality_violation_magnitude_envelope']}"
    )


def test_feasible_xu_gives_zero_violations() -> None:
    """Feasible (x, u) should give zero violation rate and magnitude."""
    x, u = _make_feasible_xu(10)
    env = _make_envelope_matching_physical()
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, env, physical)
    assert result["inequality_violation_rate_physical"] == pytest.approx(0.0, abs=1e-12)
    assert result["inequality_violation_magnitude_physical"] == pytest.approx(0.0, abs=1e-12)
    assert result["inequality_violation_rate_envelope"] == pytest.approx(0.0, abs=1e-12)
    assert result["inequality_violation_magnitude_envelope"] == pytest.approx(0.0, abs=1e-12)


def test_violation_detected_on_violating_xu() -> None:
    """Violating (x, u) should give nonzero rate for both physical and envelope."""
    x, u = _make_violating_xu(10)
    env = _make_envelope_matching_physical()
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, env, physical)
    assert result["inequality_violation_rate_physical"] > 0.0, (
        "physical rate should be positive for violating trajectory"
    )
    assert result["inequality_violation_rate_envelope"] > 0.0, (
        "envelope rate should be positive for violating trajectory"
    )


def test_tighter_envelope_can_give_higher_rate() -> None:
    """A tighter envelope detects more violations than physical bounds."""
    M = 10
    x, u = _make_feasible_xu(M)
    # Tight envelope: v_max = 5.0 (much tighter than physical 22.0)
    # Our feasible x has speeds in [1.0, 10.0], so speeds > 5.0 will violate envelope
    tight_env = EmpiricalEnvelope(
        state_min=jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, 0.0]),
        state_max=jnp.array([jnp.inf, jnp.inf, jnp.pi, 5.0]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )
    physical = inD_physical_constraints()
    result = compute_inequality_dual(x, u, tight_env, physical)
    # Physical: no violations (x within [0,22])
    assert result["inequality_violation_rate_physical"] == pytest.approx(0.0, abs=1e-12)
    # Envelope: some violations (speeds > 5.0)
    assert result["inequality_violation_rate_envelope"] > 0.0, (
        "tight envelope should detect speed violations"
    )
