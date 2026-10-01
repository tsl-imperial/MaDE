# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Tune the EKF/RTS smoother's process and measurement covariances. TRAINING SPLIT ONLY.

Process and measurement noise covariances are tuned on the training split only, never on
test; this script is the only place the smoother sees data before evaluation. It writes the
selected values to a JSON that the evaluation reads; the evaluation tunes nothing.

## Procedure

**Diagonal shapes come from training-split statistics.** Three vectors, all measured on the
training split:

| vector | measured as | meaning |
|---|---|---|
| `r_base` | per-channel MSE of the predictor against ground truth | measurement noise the smoother faces -- the "measurement" is a prediction |
| `q_state_base` | per-channel MSE of the one-step known-model residual on ground-truth trajectories | model mismatch between the known kinematic-bicycle model and recorded inD trajectories |
| `q_ctrl_base` | per-channel MSE of the one-step increment of the known model's recovered controls | random-walk step size, i.e. the control block's process noise |

**Only one scalar is swept -- the whole search space.** Scaling `P0`, `Q` and `R` by a common
factor leaves the Kalman gain exactly unchanged: if `P0 -> aP0`, `Q -> aQ`, `R -> aR` then
`P^- -> aP^-` by induction and `K = aP^-H^T(a(HP^-H^T+R))^{-1}` is invariant, so the state
estimate and its Jacobian are invariant and the induction closes. The estimator depends on
the RATIO alone. Sweeping `q_scale` with `R` pinned at `r_base` and `P0`'s state block pinned
at `r_base` covers that ratio completely.

**Exception:** `P0`'s control block has no `r_base` counterpart -- it is a prior on initial
steering and acceleration, not a measurement covariance -- so it sits outside the invariance
group. Held at the training-split variance of the recovered controls, not tuned. Affects only
the first few steps of each window.

**Selection criterion: training-split ADE.** Reported as a full curve over the grid, not a
single winner, so a reader can see whether the minimum is sharp or the curve is flat.

**The known-model residual is recorded along the same curve and is NOT used to select.** The
smoother is selected on ADE, then reported on Dyn.-K and the inequality metrics.
`CRITERION_SENSITIVITY` records the q_scale the residual would have chosen and how many
decades away it sits.

**Pooled over all five predictor seeds, per (panel, family).** Noise covariance describes a
predictor family's error statistics, which differ by family, so one value per family is the
right granularity. Pooling avoids picking a seed arbitrarily. No per-seed tuning.

**Nothing in this script reads the test split.**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from made.utils.jax_setup import configure  # noqa: E402

configure()

import jax.numpy as jnp  # noqa: E402

from made.baselines.ekf_rts_smoother import KinodynamicSmoother  # noqa: E402
from made.data.ind_data import create_ind_data_source  # noqa: E402
from made.data.window_dataset import assemble_context, make_prediction_windows  # noqa: E402
from made.evaluation.real_data_eval import L_REF  # noqa: E402
from made.physics.kinematic_bicycle import KinematicBicycle  # noqa: E402
from made.upstream.factory import load_predictor  # noqa: E402
from scripts.common.run_guard import OutputBusy, claim_output  # noqa: E402

FAMILIES = ("lstm", "ssm", "transformer")
SEEDS = (0, 1, 2, 3, 4)


def _panels(data_dir: str, predictor_root: str) -> dict[str, dict[str, Any]]:
    """Describe the evaluation panels.

    Args:
        data_dir: Directory of the preprocessed inD data.
        predictor_root: Directory holding the trained predictors.

    Returns:
        Mapping from panel name to its data directory, predictor root and time step.
    """
    return {
        "ind": {
            "data_dir": data_dir,
            "predictor_root": predictor_root,
            "dt": 0.2,
        },
    }


def _physics(panel: str) -> KinematicBicycle:
    """The kinematic bicycle known model.

    Args:
        panel: Panel name (unused; kept for a uniform signature).

    Returns:
        The kinematic bicycle model.
    """
    del panel
    return KinematicBicycle()


def _params() -> jax.Array:
    """`L_REF`, the reference wheelbase the metric stencil scores every row against.

    Returns:
        Parameter array holding ``L_REF``.
    """
    return jnp.asarray([L_REF], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# training-split windows
# ---------------------------------------------------------------------------


def _train_windows(panel: str, cfg: dict, spec: dict, cap: int) -> dict[str, jax.Array]:
    """Build TRAIN-split windows; the split name is hardcoded, no flag to change it.

    Args:
        panel: Panel name.
        cfg: Panel configuration (data directory, ...).
        spec: Window spec with history, horizon and stride.
        cap: Maximum number of windows to keep.

    Returns:
        The window dict.

    Raises:
        ValueError: If the split yields no windows.
    """
    states, metadata, lengths = create_ind_data_source(cfg["data_dir"], "train")
    w = make_prediction_windows(
        states, lengths, metadata,
        history=int(spec["history"]), horizon=int(spec["horizon"]),
        stride=int(spec["stride"]),
    )
    n = int(w["context"].shape[0])
    if n == 0:
        raise ValueError(f"zero train windows for panel {panel} at spec {spec}")
    if n > cap:
        # Fixed stride rather than random draw: reproducible without a key, spans the split.
        idx = jnp.arange(0, n, max(1, n // cap))[:cap]
        w = {k: v[idx] for k, v in w.items() if hasattr(v, "shape") and v.shape[0] == n}
    return w


def _context(panel: str, w: dict[str, jax.Array]) -> jax.Array:
    """Assemble the predictor input for each window.

    Args:
        panel: Panel name (unused).
        w: Window dict with ``context`` and ``metadata``.

    Returns:
        The assembled context batch.
    """
    del panel
    return jax.vmap(assemble_context)(w["context"], w["metadata"])


def _forward(predictor: Any, context: jax.Array, chunk: int) -> jax.Array:
    """Run the predictor over all windows in chunks.

    Args:
        predictor: The trained predictor.
        context: Assembled contexts.
        chunk: Windows per forward call.

    Returns:
        Predicted futures for all windows.
    """
    out = [jax.vmap(predictor)(context[i:i + chunk]) for i in range(0, context.shape[0], chunk)]
    return jnp.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# the three base diagonals
# ---------------------------------------------------------------------------


def _known_step(
    physics: KinematicBicycle, x: jax.Array, u: jax.Array, params: jax.Array, dt: float
) -> jax.Array:
    """One Heun step of the known model, the same step the filter propagates with.

    Args:
        physics: The known model.
        x: State.
        u: Control.
        params: Physics parameters.
        dt: Time step in seconds.

    Returns:
        The next state.
    """
    k1 = physics.vector_field(x, u, params, 0.0)
    k2 = physics.vector_field(x + dt * k1, u, params, 0.0)
    return x + 0.5 * dt * (k1 + k2)


def _model_mismatch(physics: KinematicBicycle, gt: jax.Array, params: jax.Array, dt: float
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-channel one-step known-model residual, control increments, and control variance.

    Measured on ground-truth training trajectories under the known model's own analytic
    control inverse, the same inverse the filter initialises from.

    Args:
        physics: The known model.
        gt: Ground-truth trajectories, shape ``[N, T, D]``.
        params: Physics parameters.
        dt: Time step in seconds.

    Returns:
        Per-channel state residual, control-increment and control-variance estimates.
    """
    def per_window(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Residuals, control increments and controls for one trajectory.

        Args:
            x: Ground-truth trajectory.

        Returns:
            Squared residual, squared control increment and the recovered controls.
        """
        u = jax.vmap(physics.known_control_prior, in_axes=(0, 0, None, None))(
            x[:-1], x[1:], params, dt)
        x_next = jax.vmap(_known_step, in_axes=(None, 0, 0, None, None))(
            physics, x[:-1], u, params, dt)
        return (x_next - x[1:]) ** 2, jnp.diff(u, axis=0) ** 2, u

    resid, du, u = jax.vmap(per_window)(gt)
    return (jnp.mean(resid, axis=(0, 1)),
            jnp.mean(du, axis=(0, 1)),
            jnp.var(u.reshape(-1, u.shape[-1]), axis=0))


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def _ade(x: jax.Array, gt: jax.Array) -> float:
    """Mean position error (ADE) over all windows and steps.

    Args:
        x: Predicted trajectories.
        gt: Ground-truth trajectories.

    Returns:
        The mean position error.
    """
    return float(jnp.mean(jnp.linalg.norm(x[..., :2] - gt[..., :2], axis=-1)))


def _known_residual(physics: KinematicBicycle, x: jax.Array, params: jax.Array, dt: float) -> float:
    """Mean one-step known-model residual of a trajectory batch, under the model's own inverse.

    Dyn.-K in the shape the smoother targets, recorded alongside the ADE curve but not used
    to select. Selection criterion is ADE; the smoother is then reported on Dyn.-K and the
    inequality metrics, so recording both shows whether the ADE-optimal covariance is far
    from the Dyn.-K-optimal one.

    Args:
        physics: The known model.
        x: Trajectory batch.
        params: Physics parameters.
        dt: Time step in seconds.

    Returns:
        The mean one-step residual.
    """
    def per_window(t: jax.Array) -> jax.Array:
        """Per-step residual norms for one trajectory.

        Args:
            t: Trajectory.

        Returns:
            Residual norm at each step.
        """
        u = jax.vmap(physics.known_control_prior, in_axes=(0, 0, None, None))(
            t[:-1], t[1:], params, dt)
        nxt = jax.vmap(_known_step, in_axes=(None, 0, 0, None, None))(
            physics, t[:-1], u, params, dt)
        return jnp.linalg.norm(nxt - t[1:], axis=-1)

    return float(jnp.mean(jax.vmap(per_window)(x)))


def _build(
    physics: KinematicBicycle,
    params: jax.Array,
    dt: float,
    q_state: jax.Array,
    q_ctrl: jax.Array,
    r: jax.Array,
    p0: jax.Array,
) -> KinodynamicSmoother:
    """Build the smoother from the base diagonals.

    Args:
        physics: The known model.
        params: Physics parameters.
        dt: Time step in seconds.
        q_state: Process noise diagonal for the state block.
        q_ctrl: Process noise diagonal for the control block.
        r: Measurement noise diagonal.
        p0: Initial covariance diagonal.

    Returns:
        The configured smoother.
    """
    return KinodynamicSmoother(
        physics=physics, params=params, dt=dt,
        q_diag=jnp.concatenate([q_state, q_ctrl]), r_diag=r, p0_diag=p0)


def _floor(v: jax.Array) -> jax.Array:
    """Keep a channel that moved by exactly zero on the training split from zeroing the matrix.

    Positive-definiteness guard on a covariance diagonal; fires only on a channel with no
    measured variation.

    Args:
        v: Channel variances.

    Returns:
        The variances with a floor of 1e-12.
    """
    return jnp.maximum(v, 1e-12)


def _tune_family(panel: str, cfg: dict, family: str, grid: jax.Array,
                 cap: int, chunk: int, verbose: bool) -> dict:
    """Sweep the noise scale on the training split for one predictor family.

    Args:
        panel: Panel name.
        cfg: Panel configuration.
        family: Predictor family name.
        grid: Candidate noise scales.
        cap: Maximum windows per seed.
        chunk: Windows per predictor forward call.
        verbose: Print progress when True.

    Returns:
        The tuning record: base diagonals, the sweep curve and the selected scale.

    Raises:
        FileNotFoundError: If a predictor checkpoint is missing.
        ValueError: If predictors disagree on the window spec.
        RuntimeError: If every grid point gives a non-finite ADE.
    """
    physics, params, dt = _physics(panel), _params(), float(cfg["dt"])
    root = Path(cfg["predictor_root"])

    spec = None
    preds, gts, x0s = [], [], []
    for seed in SEEDS:
        d = root / f"{family}_stage1_seed{seed}"
        if not (d / "predictor.eqx").exists():
            raise FileNotFoundError(f"missing predictor checkpoint: {d}")
        predictor, pcfg = load_predictor(d)
        s = {k: pcfg[k] for k in ("history", "horizon", "stride")}
        if spec is None:
            spec = s
            w = _train_windows(panel, cfg, spec, cap)
            ctx = _context(panel, w)
            gt, x0 = w["future"], w["context"][:, -1, :]
        elif s != spec:
            raise ValueError(f"{d} window spec {s} disagrees with seed0's {spec}")
        preds.append(_forward(predictor, ctx, chunk))
        gts.append(gt)
        x0s.append(x0)
        if verbose:
            print(f"    seed{seed} forward done", flush=True)

    pred = jnp.concatenate(preds, axis=0)
    gt_all = jnp.concatenate(gts, axis=0)
    x0_all = jnp.concatenate(x0s, axis=0)
    # The smoother sees the anchor state prepended, like every other correction row.
    meas = jnp.concatenate([x0_all[:, None, :], pred], axis=1)
    gt_full = jnp.concatenate([x0_all[:, None, :], gt_all], axis=1)

    r_base = _floor(jnp.mean((pred - gt_all) ** 2, axis=(0, 1)))
    q_state_base, q_ctrl_base, ctrl_var = _model_mismatch(physics, gt_full, params, dt)
    q_state_base, q_ctrl_base, ctrl_var = _floor(q_state_base), _floor(q_ctrl_base), _floor(ctrl_var)
    # P0's state block sits at measurement scale (makes the single-scalar sweep complete);
    # control block is the prior described in the module docstring.
    p0 = jnp.concatenate([r_base, ctrl_var])

    raw_ade = _ade(pred, gt_all)
    raw_dynk = _known_residual(physics, meas, params, dt)
    curve = []
    for q in grid:
        sm = _build(physics, params, dt,
                    float(q) * q_state_base, float(q) * q_ctrl_base, r_base, p0)
        x_s, _ = jax.vmap(sm.smooth_trajectory)(meas)
        curve.append({
            "q_scale": float(q),
            "train_ade": _ade(x_s[:, 1:], gt_all),
            "train_known_model_residual": _known_residual(physics, x_s, params, dt),
        })
        if verbose:
            print(f"    q_scale {float(q):.3e} -> train ADE {curve[-1]['train_ade']:.6f}, "
                  f"Dyn-K {curve[-1]['train_known_model_residual']:.6e}", flush=True)

    finite = [c for c in curve if jnp.isfinite(c["train_ade"])]
    if not finite:
        raise RuntimeError(f"{panel}/{family}: every grid point produced a non-finite ADE")
    best = min(finite, key=lambda c: c["train_ade"])
    at_edge = best["q_scale"] in (float(grid[0]), float(grid[-1]))
    within_1pct = [c["q_scale"] for c in finite
                   if c["train_ade"] <= best["train_ade"] * 1.01]

    # Dyn.-K-optimal setting, recorded but not used to select.
    finite_k = [c for c in curve if jnp.isfinite(c["train_known_model_residual"])]
    best_dynk = min(finite_k, key=lambda c: c["train_known_model_residual"]) if finite_k else None

    return {
        "panel": panel, "family": family, "window_spec": spec,
        "n_windows_pooled": int(meas.shape[0]), "seeds_pooled": list(SEEDS),
        "r_base": [float(v) for v in r_base],
        "q_state_base": [float(v) for v in q_state_base],
        "q_ctrl_base": [float(v) for v in q_ctrl_base],
        "p0_control_block": [float(v) for v in ctrl_var],
        "raw_train_ade": raw_ade,
        "grid": [float(g) for g in grid],
        "curve": curve,
        "selected_q_scale": best["q_scale"],
        "selection_criterion": "train-split ADE",
        "selected_train_ade": best["train_ade"],
        "selected_train_known_model_residual": best["train_known_model_residual"],
        "raw_train_known_model_residual": raw_dynk,
        "CRITERION_SENSITIVITY": {
            "what": "the q_scale that would have been selected by the KNOWN-MODEL residual "
                    "instead of by ADE. Recorded, not used. The smoother is selected on ADE "
                    "and then reported on Dyn.-K and the inequality metrics, so how far apart "
                    "these two sit is the measure of whether that choice is doing real work.",
            "q_scale_by_known_model_residual": best_dynk["q_scale"] if best_dynk else None,
            "its_train_ade": best_dynk["train_ade"] if best_dynk else None,
            "its_known_model_residual":
                best_dynk["train_known_model_residual"] if best_dynk else None,
            "same_q_scale_as_ade":
                bool(best_dynk and best_dynk["q_scale"] == best["q_scale"]),
            "decades_apart":
                (abs(jnp.log10(best_dynk["q_scale"] / best["q_scale"])).item()
                 if best_dynk and best_dynk["q_scale"] > 0 and best["q_scale"] > 0 else None),
        },
        "train_ade_improvement_over_raw": raw_ade - best["train_ade"],
        "minimum_at_grid_edge": bool(at_edge),
        "q_scales_within_1pct_of_best": within_1pct,
        "flat_minimum": len(within_1pct) > 1,
    }


def main() -> int:
    """Tune the smoother covariances on the training split and write the artifact.

    Returns:
        Process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "inD-preprocessed" / "v1"))
    ap.add_argument("--predictor-root", default=str(ROOT / "outputs" / "ind" / "predictors"))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--max-windows", type=int, default=2000,
                    help="cap on train windows per seed before pooling")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--grid-lo", type=float, default=-6.0)
    ap.add_argument("--grid-hi", type=float, default=2.0)
    ap.add_argument("--grid-n", type=int, default=17)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ind" / "smoother_noise.json"))
    a = ap.parse_args()

    grid = jnp.logspace(a.grid_lo, a.grid_hi, a.grid_n)
    panels_cfg = _panels(a.data_dir, a.predictor_root)
    panels = list(panels_cfg)
    fams = [f for f in a.families.split(",") if f in FAMILIES]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    started = time.time()
    try:
        with claim_output(out):
            for panel in panels:
                for family in fams:
                    print(f"[{panel}/{family}] tuning on the TRAIN split", flush=True)
                    t = time.time()
                    rec = _tune_family(panel, panels_cfg[panel], family, grid,
                                       a.max_windows, a.chunk, not a.quiet)
                    rec["seconds"] = time.time() - t
                    records.append(rec)
                    print(f"  selected q_scale {rec['selected_q_scale']:.3e}, "
                          f"train ADE {rec['selected_train_ade']:.6f} "
                          f"(raw {rec['raw_train_ade']:.6f}), "
                          f"edge={rec['minimum_at_grid_edge']}, "
                          f"flat={rec['flat_minimum']}, {rec['seconds'] / 60:.1f} min",
                          flush=True)
                    out.write_text(json.dumps({
                        "artifact": str(out),
                        "what": "EKF/RTS smoother noise covariances, tuned on the TRAINING "
                                "SPLIT ONLY. The evaluation reads this file and tunes nothing.",
                        "procedure": (
                            "Diagonal shapes are measured on the train split: r_base is the "
                            "predictor's per-channel MSE against ground truth, q_state_base is "
                            "the per-channel one-step known-model residual on ground-truth "
                            "trajectories, q_ctrl_base is the per-channel one-step increment of "
                            "the known model's own recovered controls. A COMMON rescaling of "
                            "P0, Q and R leaves the Kalman gain exactly unchanged, so the "
                            "estimator depends on the ratio alone and the single swept scalar "
                            "q_scale covers the whole space; P0's state block is pinned at "
                            "r_base for that reason. P0's control block is a prior outside the "
                            "invariance group, held at the train-split control variance and NOT "
                            "tuned. Selection minimises train-split ADE. One q_scale per "
                            "(panel, family), pooled over all five predictor seeds."),
                        "predictor_roots": {p: str(panels_cfg[p]["predictor_root"]) for p in panels},
                        "split": "train",
                        "elapsed_minutes": (time.time() - started) / 60.0,
                        "records": records,
                    }, indent=2) + "\n")
    except OutputBusy:
        print("another driver holds this output; nothing done", flush=True)
        return 0

    edge = [f"{r['panel']}/{r['family']}" for r in records if r["minimum_at_grid_edge"]]
    if edge:
        print(f"\nMINIMUM AT A GRID EDGE for {edge} -- the grid does not bracket the optimum "
              f"and must be widened before these values are used.", flush=True)
    print(f"\n{len(records)} (panel, family) pairs tuned in "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    print(f"wrote {out}", flush=True)
    return 1 if edge else 0


if __name__ == "__main__":
    sys.exit(main())
