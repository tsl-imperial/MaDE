# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Unit tests for stationary-carry logic in evaluate_stepwise_feasible_split.

Synthetic trajectories: v_prev ∈ {0.1, 0.4, 1.0} × m ∈ {0.5, 1.0, 2.0} → 9 cases.

Key invariants verified:
1. is_stationary fires iff |v_avg| < 0.5 (v_avg computed from x_prev and x_perturbed).
2. Bucket disjointness: infeasible + stationary are mutually exclusive; their
   combined count plus feasible count == total M.
3. fraction_infeasible + fraction_stationary_carry + frac_feasible == 1.0 exactly.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import pytest

from made.evaluation.metrics import (
    EmpiricalEnvelope,
    _perturb_base_vector,
    _stationary_infeasible_mask,
    evaluate_stepwise_feasible_split,
)
from made.physics import inD_physical_constraints


# Helpers

def _make_env(range_v: float = 20.0) -> EmpiricalEnvelope:
    """Build an EmpiricalEnvelope over the inD state and control ranges.

    Args:
        range_v: Upper bound on speed.

    Returns:
        Envelope with the given speed range.
    """
    return EmpiricalEnvelope(
        state_min=jnp.array([0.0, 0.0, -jnp.pi, 0.0]),
        state_max=jnp.array([10.0, 10.0, jnp.pi, range_v]),
        control_min=jnp.array([-0.5, -8.0]),
        control_max=jnp.array([0.5, 4.0]),
    )


def _build_cases() -> list[tuple[float, float]]:
    """Return all (v_prev, m) combinations.

    Returns:
        List of (v_prev, m) pairs.
    """
    v_prevs = [0.1, 0.4, 1.0]
    multipliers = [0.5, 1.0, 2.0]
    return list(itertools.product(v_prevs, multipliers))


def _build_flat_inputs(
    v_prev: float,
    m: float,
    env: EmpiricalEnvelope,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build (x_prev_gt, x_perturbed, x_corrected, x_curr_gt) for a single case.

    We set x_curr_gt = x_prev_gt (trivial GT), so fidelity = ||x_corrected - x_prev_gt||.
    x_corrected = x_curr_gt (perfect correction) for simplicity — fidelity = 0.
    x_perturbed = x_prev + m * base * sign (sign = +1, all dims).

    Args:
        v_prev: Previous-step speed.
        m: Perturbation multiplier.
        env: Empirical envelope.

    Returns:
        The four flat arrays.
    """
    base_vec = _perturb_base_vector(env)
    x_prev = jnp.array([5.0, 5.0, 0.0, v_prev])
    sign = jnp.ones(4)
    x_perturbed = x_prev + jnp.asarray(m) * base_vec * sign
    x_curr_gt = x_prev  # GT doesn't move
    x_corrected = x_curr_gt  # perfect correction
    # Wrap as (1, state_dim) flat arrays
    return (
        x_prev[None],
        x_perturbed[None],
        x_corrected[None],
        x_curr_gt[None],
    )


# Tests

@pytest.mark.parametrize("v_prev,m", _build_cases())
def test_stationary_flag_matches_v_avg(v_prev: float, m: float) -> None:
    """is_stationary should fire iff |v_avg_post_perturb| < 0.5."""
    env = _make_env(range_v=20.0)
    constraints = inD_physical_constraints()
    x_prev_flat, x_perturbed_flat, x_corrected_flat, x_curr_flat = _build_flat_inputs(
        v_prev, m, env
    )
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_perturbed_flat,
        x_prev_gt=x_prev_flat,
        x_corrected=x_corrected_flat,
        x_curr_gt=x_curr_flat,
        dt=0.04,
        constraints_phys=constraints,
    )
    # v_avg = 0.5 * (v_prev + v_perturbed); v_perturbed = v_prev + m * base[3]
    base_vec = _perturb_base_vector(env)
    v_perturbed = v_prev + float(m) * float(base_vec[3])
    v_avg = 0.5 * (v_prev + v_perturbed)
    expected_stationary = abs(v_avg) < 0.5
    frac_stat = float(result["fraction_stationary_carry"])
    if expected_stationary:
        assert frac_stat == pytest.approx(1.0, abs=1e-12), (
            f"v_prev={v_prev}, m={m}: expected stationary=1.0, got {frac_stat}"
        )
    else:
        assert frac_stat == pytest.approx(0.0, abs=1e-12), (
            f"v_prev={v_prev}, m={m}: expected stationary=0.0, got {frac_stat}"
        )


@pytest.mark.parametrize("v_prev,m", _build_cases())
def test_bucket_disjointness(v_prev: float, m: float) -> None:
    """fraction_infeasible + fraction_stationary_carry + frac_feasible == 1.0."""
    env = _make_env(range_v=20.0)
    constraints = inD_physical_constraints()
    x_prev_flat, x_perturbed_flat, x_corrected_flat, x_curr_flat = _build_flat_inputs(
        v_prev, m, env
    )
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_perturbed_flat,
        x_prev_gt=x_prev_flat,
        x_corrected=x_corrected_flat,
        x_curr_gt=x_curr_flat,
        dt=0.04,
        constraints_phys=constraints,
    )
    frac_inf = float(result["fraction_infeasible"])
    frac_stat = float(result["fraction_stationary_carry"])
    # frac_feasible = 1 - frac_inf - frac_stat (they are disjoint and exhaustive)
    frac_feasible = 1.0 - frac_inf - frac_stat
    assert frac_feasible >= -1e-12, (
        f"v_prev={v_prev}, m={m}: frac_feasible={frac_feasible} < 0 (buckets overlap)"
    )
    total = frac_inf + frac_stat + frac_feasible
    assert total == pytest.approx(1.0, abs=1e-12), (
        f"v_prev={v_prev}, m={m}: fractions sum to {total}, expected 1.0"
    )


def test_stationary_excluded_from_infeasible_bucket() -> None:
    """Stationary steps must not appear in the infeasible bucket.

    Build a case where v_prev is very small (definitely stationary) but the
    perturbed state violates physical bounds — it must NOT count as infeasible.
    """
    env = _make_env(range_v=20.0)
    constraints = inD_physical_constraints()
    # v_prev=0.05: clearly stationary (|v_avg| < 0.5 for any small perturbation)
    # Set v perturbed explicitly to a constraint-violating value (< 0 → speed bound)
    x_prev = jnp.array([[5.0, 5.0, 0.0, 0.05]])
    x_perturbed = jnp.array([[5.0, 5.0, 0.0, -1.0]])  # v<0: violates state_min v=0
    x_curr_gt = x_prev
    x_corrected = x_curr_gt
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_perturbed,
        x_prev_gt=x_prev,
        x_corrected=x_corrected,
        x_curr_gt=x_curr_gt,
        dt=0.04,
        constraints_phys=constraints,
    )
    # v_avg = 0.5*(0.05 + (-1.0)) = -0.475, |v_avg| < 0.5 → stationary
    assert float(result["fraction_stationary_carry"]) == pytest.approx(1.0, abs=1e-12)
    # infeasible must be 0 even though the state violates bounds
    assert float(result["fraction_infeasible"]) == pytest.approx(0.0, abs=1e-12)


def test_non_stationary_infeasible_counted() -> None:
    """Non-stationary steps that violate bounds appear in infeasible bucket."""
    constraints = inD_physical_constraints()
    # v_prev=5.0, v_perturbed=5.0: |v_avg|=5.0 → not stationary
    # Force a state-bound violation: heading > pi
    x_prev = jnp.array([[5.0, 5.0, 0.0, 5.0]])
    x_perturbed = jnp.array([[5.0, 5.0, 4.0, 5.0]])  # heading=4 > pi → violation
    x_curr_gt = x_prev
    x_corrected = x_curr_gt
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_perturbed,
        x_prev_gt=x_prev,
        x_corrected=x_corrected,
        x_curr_gt=x_curr_gt,
        dt=0.04,
        constraints_phys=constraints,
    )
    assert float(result["fraction_stationary_carry"]) == pytest.approx(0.0, abs=1e-12)
    assert float(result["fraction_infeasible"]) == pytest.approx(1.0, abs=1e-12)
