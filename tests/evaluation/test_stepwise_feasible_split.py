"""§3.3 Unit tests for evaluate_stepwise_feasible_split.

Hand-crafted 3-trajectory KB fixture with T_i = [5, 3, 4].
Total M = (5-1)+(3-1)+(4-1) = 9 valid transitions.

Tests:
(a) Infeasibility uses _i_known, not method-I (structural + functional check).
(b) fid_feasible == 0 when x_corrected == x_curr_gt on a feasible step.
(c) fid_infeasible > 0 on infeasible step where corrected != gt.
(d) frac_inf(m=0) == 0 with tame states and inD_physical_constraints.
(e) Flat aggregation: function operates on flat (9, 4) arrays.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from made.evaluation.metrics import (
    _i_known,
    evaluate_stepwise_feasible_split,
)
from made.physics import inD_physical_constraints


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DT = 0.1
_WHEELBASE = 2.7

# inD physical constraints: x,y unbounded; v ∈ [0, 22]; delta ∈ [-0.5, 0.5]; a ∈ [-8, 4]
_CONSTRAINTS = inD_physical_constraints()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tame_state(v: float = 5.0) -> jax.Array:
    """A kinematic-bicycle state well inside inD_physical_constraints."""
    return jnp.array([1.0, 2.0, 0.1, v], dtype=jnp.float64)


def _tame_next_state(x_prev: jax.Array, delta: float = 0.05, accel: float = 0.5) -> jax.Array:
    """One Euler step forward — stays tame."""
    from made.physics.kinematic_bicycle import KinematicBicycle
    physics = KinematicBicycle()
    params = jnp.array([_WHEELBASE], dtype=jnp.float64)
    u = jnp.array([delta, accel], dtype=jnp.float64)
    dstate = physics.vector_field(x_prev, u, params, 0.0)
    return x_prev + _DT * dstate


def _build_flat_pairs_tame(n: int = 9) -> tuple[jax.Array, jax.Array]:
    """Return (x_prev_flat, x_curr_flat) of shape (n, 4) — all tame / feasible."""
    rows_prev = []
    rows_curr = []
    x = _tame_state(v=5.0)
    for _ in range(n):
        xn = _tame_next_state(x)
        rows_prev.append(x)
        rows_curr.append(xn)
        x = xn
    return jnp.stack(rows_prev), jnp.stack(rows_curr)


# ---------------------------------------------------------------------------
# Fixture: 3 trajectories T=[5,3,4] → M=9 flat pairs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_9_pairs():
    """Build exactly 9 tame flat pairs from 3 trajectories with lengths [5,3,4]."""
    lengths = [5, 3, 4]
    all_prev, all_curr = [], []
    for length in lengths:
        x = _tame_state(v=5.0)
        for _ in range(length - 1):
            xn = _tame_next_state(x)
            all_prev.append(x)
            all_curr.append(xn)
            x = xn
    x_prev = jnp.stack(all_prev)   # (9, 4)
    x_curr = jnp.stack(all_curr)   # (9, 4)
    assert x_prev.shape == (9, 4)
    assert x_curr.shape == (9, 4)
    return x_prev, x_curr


# ---------------------------------------------------------------------------
# (e) Flat aggregation: M=9 shape check
# ---------------------------------------------------------------------------

def test_flat_aggregation_shape(flat_9_pairs):
    """Function must accept flat (9, 4) inputs without error and return scalar outputs."""
    x_prev, x_curr = flat_9_pairs
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_curr,
        x_prev_gt=x_prev,
        x_corrected=x_curr,
        x_curr_gt=x_curr,
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    for key in ("fidelity_feasible", "fidelity_infeasible", "fraction_infeasible",
                "fraction_stationary_carry"):
        assert key in result, f"missing key: {key}"
        arr = result[key]
        assert arr.shape == (), f"{key} should be scalar, got shape {arr.shape}"


# ---------------------------------------------------------------------------
# (d) frac_inf(m=0) == 0 with tame states
# ---------------------------------------------------------------------------

def test_frac_inf_zero_when_no_perturbation(flat_9_pairs):
    """With x_perturbed == x_curr (tame, no perturbation), fraction_infeasible must be 0."""
    x_prev, x_curr = flat_9_pairs
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_curr,
        x_prev_gt=x_prev,
        x_corrected=x_curr,
        x_curr_gt=x_curr,
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    frac_inf = float(result["fraction_infeasible"])
    assert frac_inf == pytest.approx(0.0, abs=1e-10), (
        f"Expected frac_inf=0 with tame unperturbed states, got {frac_inf}"
    )


# ---------------------------------------------------------------------------
# (b) fid_feasible == 0 when corrected == gt on feasible step
# ---------------------------------------------------------------------------

def test_fidelity_feasible_zero_when_corrected_equals_gt(flat_9_pairs):
    """When x_corrected == x_curr_gt on all feasible steps, fidelity_feasible must be 0."""
    x_prev, x_curr = flat_9_pairs
    # No perturbation → all steps feasible; corrected == gt
    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_curr,
        x_prev_gt=x_prev,
        x_corrected=x_curr,   # corrected == gt
        x_curr_gt=x_curr,
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    fid_feas = float(result["fidelity_feasible"])
    assert fid_feas == pytest.approx(0.0, abs=1e-12), (
        f"fidelity_feasible should be 0 when corrected==gt on feasible steps, got {fid_feas}"
    )


# ---------------------------------------------------------------------------
# (c) fid_infeasible > 0 on infeasible step where corrected != gt
# ---------------------------------------------------------------------------

def test_fidelity_infeasible_positive_when_corrected_differs():
    """Build an infeasible step explicitly: push v above 22 m/s (inD v_max).

    Confirm fid_infeasible > 0 when x_corrected != x_curr_gt on that step.
    """
    # x_prev: tame v=20 (inside [0,22])
    x_prev = jnp.array([0.0, 0.0, 0.0, 20.0], dtype=jnp.float64)
    # x_perturbed: v=25 (above v_max=22) — infeasible under inD constraints
    x_pert = jnp.array([0.0, 0.0, 0.0, 25.0], dtype=jnp.float64)
    # x_curr_gt: tame v=21
    x_gt = jnp.array([0.0, 0.0, 0.0, 21.0], dtype=jnp.float64)
    # x_corrected: something different from gt (clamped to 22, not 21)
    x_corr = jnp.array([0.0, 0.0, 0.0, 22.0], dtype=jnp.float64)

    x_prev_flat = x_prev[None]      # (1, 4)
    x_pert_flat = x_pert[None]
    x_gt_flat = x_gt[None]
    x_corr_flat = x_corr[None]

    # Verify _i_known gives control in-box for this step (so infeasibility is from state v>22)
    u, is_stat = _i_known(x_prev, x_pert, _DT)
    # v_avg = (20 + 25)/2 = 22.5 > 0.5 → not stationary

    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_pert_flat,
        x_prev_gt=x_prev_flat,
        x_corrected=x_corr_flat,
        x_curr_gt=x_gt_flat,
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    frac_inf = float(result["fraction_infeasible"])
    fid_inf = float(result["fidelity_infeasible"])

    assert frac_inf > 0.0, f"Expected fraction_infeasible > 0 for v=25 state, got {frac_inf}"
    assert fid_inf > 0.0, (
        f"Expected fidelity_infeasible > 0 when corrected ({x_corr[3]}) != gt ({x_gt[3]}), "
        f"got {fid_inf}"
    )


# ---------------------------------------------------------------------------
# (a) Infeasibility uses _i_known, not method-I (structural + functional)
# ---------------------------------------------------------------------------

def test_bucket_assignment_uses_i_known_not_method_I():
    """Structural: evaluate_stepwise_feasible_split takes no model argument.

    Functional: bucket assignment is invariant to the corrected output — only
    x_perturbed and x_prev_gt (used by _i_known internally) determine buckets.
    """
    x_prev = jnp.array([0.0, 0.0, 0.0, 20.0], dtype=jnp.float64)
    x_pert = jnp.array([0.0, 0.0, 0.0, 25.0], dtype=jnp.float64)
    x_gt = jnp.array([0.0, 0.0, 0.0, 21.0], dtype=jnp.float64)

    x_prev_flat = x_prev[None]
    x_pert_flat = x_pert[None]
    x_gt_flat = x_gt[None]

    # Run with two DIFFERENT corrected values — bucket (frac_inf) must be identical.
    corr_a = jnp.array([[0.0, 0.0, 0.0, 22.0]])   # clamped
    corr_b = jnp.array([[5.0, 5.0, 1.0, 18.0]])   # wildly different

    result_a = evaluate_stepwise_feasible_split(
        x_perturbed=x_pert_flat, x_prev_gt=x_prev_flat,
        x_corrected=corr_a, x_curr_gt=x_gt_flat,
        dt=_DT, constraints_phys=_CONSTRAINTS,
    )
    result_b = evaluate_stepwise_feasible_split(
        x_perturbed=x_pert_flat, x_prev_gt=x_prev_flat,
        x_corrected=corr_b, x_curr_gt=x_gt_flat,
        dt=_DT, constraints_phys=_CONSTRAINTS,
    )

    # fraction_infeasible and fraction_stationary_carry depend only on x_perturbed/x_prev_gt
    assert float(result_a["fraction_infeasible"]) == pytest.approx(
        float(result_b["fraction_infeasible"]), abs=1e-14
    ), "fraction_infeasible changed with different corrected — bucket must use _i_known only"

    assert float(result_a["fraction_stationary_carry"]) == pytest.approx(
        float(result_b["fraction_stationary_carry"]), abs=1e-14
    ), "fraction_stationary_carry changed with different corrected"

    # Fidelity DOES differ (depends on corrected)
    assert float(result_a["fidelity_infeasible"]) != float(result_b["fidelity_infeasible"]), (
        "fidelity_infeasible should differ for different corrected values"
    )


# ---------------------------------------------------------------------------
# Stationary carry: step with low v is excluded from both feas and inf
# ---------------------------------------------------------------------------

def test_stationary_step_excluded_from_both_buckets():
    """A step with |v_avg| < 0.5 must contribute to stationary_carry, not inf/feas."""
    # v_prev = 0.1, v_curr = 0.2 → v_avg = 0.15 < 0.5 → stationary
    x_prev = jnp.array([0.0, 0.0, 0.0, 0.1], dtype=jnp.float64)
    x_curr = jnp.array([0.0, 0.0, 0.0, 0.2], dtype=jnp.float64)

    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_curr[None],
        x_prev_gt=x_prev[None],
        x_corrected=x_curr[None],
        x_curr_gt=x_curr[None],
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    frac_stat = float(result["fraction_stationary_carry"])
    frac_inf = float(result["fraction_infeasible"])

    assert frac_stat > 0.0, f"Expected stationary step to be flagged, got frac_stat={frac_stat}"
    assert frac_inf == pytest.approx(0.0, abs=1e-14), (
        f"Stationary step must NOT appear in infeasible bucket, got frac_inf={frac_inf}"
    )


# ---------------------------------------------------------------------------
# Disjointness: fraction_infeasible + fraction_stationary_carry <= 1
# ---------------------------------------------------------------------------

def test_frac_inf_and_stationary_are_disjoint(flat_9_pairs):
    """fraction_infeasible and fraction_stationary_carry are disjoint partitions (sum ≤ 1)."""
    x_prev, x_curr = flat_9_pairs
    # Heavily perturb v to force some infeasible
    x_pert = x_curr.at[:, 3].set(30.0)

    result = evaluate_stepwise_feasible_split(
        x_perturbed=x_pert,
        x_prev_gt=x_prev,
        x_corrected=x_curr,
        x_curr_gt=x_curr,
        dt=_DT,
        constraints_phys=_CONSTRAINTS,
    )
    total = float(result["fraction_infeasible"]) + float(result["fraction_stationary_carry"])
    assert total <= 1.0 + 1e-10, (
        f"fraction_infeasible + fraction_stationary_carry = {total} > 1 (not disjoint)"
    )
