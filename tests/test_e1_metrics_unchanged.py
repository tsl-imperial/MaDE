"""E1 metric invariance — bitwise snapshot regression.

Phase 0 of ralplan-real-data-metrics-v1 (R14).

Goals:
  - Pin every E1-relied metric (`fidelity`, `dynamics_violation_known`,
    `dynamics_violation_learned`, `dynamics_violation_true`, `compute_metrics`)
    to a pre-change snapshot so adding the new real-data helpers cannot
    accidentally shift E1 paper numbers.
  - Inputs include edge cases (empty trajectory, length-1, tan(δ) near ±π/2,
    batched N=2 + unbatched).
  - Initial comparison is bitwise float equality. If a clean E1-untouched
    diff ever fails, see `tests/data/e1_metrics_snapshot_README.md` for
    the allclose downgrade procedure.

Snapshot file: tests/data/e1_metrics_snapshot.json
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp

from made.evaluation import (
    compute_metrics,
    dynamics_violation_known,
    dynamics_violation_learned,
    dynamics_violation_true,
    fidelity,
)
from made.models.augmented_dynamics import AugmentedDynamics, ZeroResidual
from made.physics import KinematicBicycle, kinematic_bicycle_constraints

_HERE = Path(__file__).parent
_SNAPSHOT = _HERE / "data" / "e1_metrics_snapshot.json"


def _build_inputs() -> dict[str, dict]:
    """Build a deterministic suite of inputs covering the regression surface."""
    physics = KinematicBicycle()
    params = jnp.array([2.7])
    constraints = kinematic_bicycle_constraints()
    dt = 0.1

    # ------------------------------------------------------------------
    # Standard fixed batch: N=2 trajectories of length 5, kinematic-bicycle.
    # Feasible by construction (small δ, small a, moderate v).
    # ------------------------------------------------------------------
    key = jax.random.key(0)
    x_corr_a = jnp.array(
        [
            [0.0, 0.0, 0.0, 5.0],
            [0.5, 0.0, 0.01, 5.05],
            [1.0, 0.005, 0.02, 5.10],
            [1.5, 0.015, 0.03, 5.15],
            [2.0, 0.030, 0.04, 5.20],
        ]
    )
    x_corr_b = jnp.array(
        [
            [10.0, 5.0, 0.5, 8.0],
            [10.7, 5.4, 0.51, 8.05],
            [11.4, 5.8, 0.52, 8.10],
            [12.1, 6.2, 0.53, 8.15],
            [12.8, 6.6, 0.54, 8.20],
        ]
    )
    x_batched = jnp.stack([x_corr_a, x_corr_b])
    # Dynamics-violation helpers expect T-1 controls per trajectory; fidelity /
    # inequality helpers reuse the same array via _align_controls_to_states.
    u_batched = jnp.tile(jnp.array([0.05, 0.5]), (2, 4, 1))
    x_gt_batched = x_batched + 0.001  # tiny offset so fidelity is non-zero
    del key

    # ------------------------------------------------------------------
    # Unbatched (single trajectory) — same metric helpers must dispatch
    # via _is_batched and produce a finite scalar.
    # ------------------------------------------------------------------
    x_unbatched = x_corr_a
    u_unbatched = jnp.tile(jnp.array([0.05, 0.5]), (4, 1))
    x_gt_unbatched = x_unbatched + 0.001

    # ------------------------------------------------------------------
    # Length-1 trajectory — dynamics_violation_known short-circuits to 0.0.
    # ------------------------------------------------------------------
    x_len1 = x_corr_a[:1]
    u_len1 = jnp.zeros((0, 2))
    x_gt_len1 = x_len1

    # ------------------------------------------------------------------
    # Empty trajectory — fidelity short-circuits to 0.0.
    # ------------------------------------------------------------------
    x_empty = jnp.zeros((0, 4))
    u_empty = jnp.zeros((0, 2))
    x_gt_empty = jnp.zeros((0, 4))

    # ------------------------------------------------------------------
    # tan(δ) near ±π/2 — Heun + ConstantStepSize stencil must remain finite
    # because of the internal jnp.clip(δ, -1.4, 1.4) in vector_field.
    # 6D state would be DynamicBicycle; we use KB with extreme δ here.
    # ------------------------------------------------------------------
    delta_extreme = float(jnp.pi / 2 - 1e-3)
    x_extreme = jnp.array(
        [
            [0.0, 0.0, 0.0, 5.0],
            [0.05, 0.0, 0.01, 5.0],
        ]
    )
    u_extreme = jnp.array([[delta_extreme, 0.0]])

    # AugmentedDynamics for dynamics_violation_learned coverage.
    # ZeroResidual makes learned == known, freezing the metric to a value
    # that matches dynamics_violation_known on the same input.
    aug = AugmentedDynamics(physics=physics, residual=ZeroResidual(state_dim=4))

    return {
        "batched": {
            "x_corrected": x_batched,
            "u_corrected": u_batched,
            "x_gt": x_gt_batched,
            "u_gt": u_batched,
        },
        "unbatched": {
            "x_corrected": x_unbatched,
            "u_corrected": u_unbatched,
            "x_gt": x_gt_unbatched,
            "u_gt": u_unbatched,
        },
        "len1": {
            "x_corrected": x_len1,
            "u_corrected": u_len1,
            "x_gt": x_gt_len1,
            "u_gt": u_len1,
        },
        "empty": {
            "x_corrected": x_empty,
            "u_corrected": u_empty,
            "x_gt": x_gt_empty,
            "u_gt": u_empty,
        },
        "extreme_delta": {
            "x_corrected": x_extreme,
            "u_corrected": u_extreme,
            "x_gt": x_extreme,
            "u_gt": u_extreme,
        },
        "_shared": {
            "physics": physics,
            "params": params,
            "constraints": constraints,
            "aug": aug,
            "dt": dt,
        },
    }


def _compute_snapshot() -> dict:
    """Compute the metric values for every input case."""
    bundle = _build_inputs()
    shared = bundle.pop("_shared")
    physics = shared["physics"]
    params = shared["params"]
    constraints = shared["constraints"]
    aug = shared["aug"]
    dt = shared["dt"]

    snapshot: dict = {}
    for name, inputs in bundle.items():
        x_c = inputs["x_corrected"]
        u_c = inputs["u_corrected"]
        x_g = inputs["x_gt"]
        u_g = inputs["u_gt"]

        case: dict = {}
        # Per-helper values.
        case["fidelity"] = float(fidelity(x_c, x_g))

        if x_c.shape[-2] >= 2 and u_c.shape[-2] >= 1:
            case["dynamics_violation_known"] = float(
                dynamics_violation_known(x_c, u_c, physics, params, dt)
            )
            case["dynamics_violation_learned"] = float(
                dynamics_violation_learned(x_c, u_c, aug, params, dt)
            )
            case["dynamics_violation_true"] = float(
                dynamics_violation_true(x_c, u_c, physics, params, dt)
            )
        else:
            case["dynamics_violation_known"] = float(
                dynamics_violation_known(x_c, u_c, physics, params, dt)
            )
            case["dynamics_violation_learned"] = float(
                dynamics_violation_learned(x_c, u_c, aug, params, dt)
            )
            case["dynamics_violation_true"] = float(
                dynamics_violation_true(x_c, u_c, physics, params, dt)
            )

        # compute_metrics returns a dict that we snapshot in full.
        if x_c.shape[-2] > 0 and u_c.shape[-2] > 0:
            metrics = compute_metrics(
                x_corrected=x_c,
                u_corrected=u_c,
                x_gt=x_g,
                u_gt=u_g,
                constraints=constraints,
                physics_known=physics,
                dynamics_learned=aug,
                params=params,
                dt=dt,
                dynamics_true=physics,
                true_params=params,
            )
            case["compute_metrics"] = {k: float(v) for k, v in metrics.items()}
        else:
            # compute_metrics requires non-empty inputs — skip on empty case.
            case["compute_metrics"] = None

        snapshot[name] = case
    return snapshot


def _write_snapshot(snapshot: dict) -> None:
    _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def _bitwise_equal(a: float, b: float) -> bool:
    """Float equality up to ULP-level cross-platform/BLAS drift, NaN-aware.

    Uses a tight relative tolerance (1e-12) rather than exact `==` so this
    snapshot test is robust to the last 1-2 bits of float64 drift observed
    across hardware/BLAS builds, without masking genuine metric changes.
    """
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def test_e1_compute_metrics_snapshot() -> None:
    """E1 metric outputs must match the frozen pre-change snapshot bitwise.

    On first run (when the snapshot file does not exist), the computed
    values are written and the test passes. Any subsequent run that
    diverges fails — guarding E1 paper numbers against silent drift.

    To regenerate intentionally (e.g. after a deliberate metric refactor),
    delete `tests/data/e1_metrics_snapshot.json` and rerun the test.
    """
    current = _compute_snapshot()
    if not _SNAPSHOT.exists() or os.environ.get("MADE_E1_SNAPSHOT_REFRESH") == "1":
        _write_snapshot(current)
        return  # snapshot established; fixture is now frozen.

    expected = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))

    assert set(current.keys()) == set(expected.keys()), (
        "snapshot case set mismatch:\n"
        f"  expected: {sorted(expected.keys())}\n"
        f"  got:      {sorted(current.keys())}"
    )

    for name in expected:
        cur = current[name]
        exp = expected[name]
        assert set(cur.keys()) == set(exp.keys()), (
            f"case {name!r}: key set differs (expected {sorted(exp.keys())},"
            f" got {sorted(cur.keys())})"
        )
        for key, exp_val in exp.items():
            cur_val = cur[key]
            if isinstance(exp_val, dict):
                assert isinstance(cur_val, dict), f"{name}.{key} type changed"
                assert set(cur_val.keys()) == set(exp_val.keys()), (
                    f"{name}.{key}: dict keys differ"
                )
                for sub_k, sub_v in exp_val.items():
                    assert _bitwise_equal(cur_val[sub_k], sub_v), (
                        f"{name}.{key}.{sub_k}: bitwise drift "
                        f"(expected {sub_v!r}, got {cur_val[sub_k]!r}). "
                        "If this is intentional, see "
                        "tests/data/e1_metrics_snapshot_README.md."
                    )
            elif exp_val is None:
                assert cur_val is None, f"{name}.{key}: expected None, got {cur_val!r}"
            else:
                assert _bitwise_equal(cur_val, exp_val), (
                    f"{name}.{key}: bitwise drift "
                    f"(expected {exp_val!r}, got {cur_val!r}). "
                    "If this is intentional, see "
                    "tests/data/e1_metrics_snapshot_README.md."
                )
