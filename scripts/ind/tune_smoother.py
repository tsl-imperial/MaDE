"""Tune the EKF/RTS smoother's process and measurement covariances. TRAINING SPLIT ONLY.

The spec is explicit: *"Process and measurement noise covariances tuned on the training split
only, never on test. Report the tuning procedure and the selected values."* This script is that
tuning, and it is the only place the smoother sees data before evaluation. It writes the
selected values to a JSON that the evaluation reads; the evaluation tunes nothing.

## The procedure, stated in full so the paper can quote it

**The diagonals' SHAPES come from training-split statistics, not from a guess.** Three vectors,
all measured on the training split and all with an interpretation:

| vector | what it is measured as | what it means |
|---|---|---|
| `r_base` | per-channel mean squared error of the predictor against ground truth | the measurement noise the smoother actually faces -- the "measurement" IS a prediction |
| `q_state_base` | per-channel mean squared ONE-STEP known-model residual on ground-truth trajectories | the model mismatch between the known kinematic-bicycle model and the recorded inD trajectories |
| `q_ctrl_base` | per-channel mean squared one-step increment of the known model's own recovered controls | the random walk's step size, so the control block's process noise is the rate the controls really move at |

**Only ONE scalar is then swept, and that is not a shortcut -- it is the whole search space.**
Scaling `P0`, `Q` and `R` by a common factor leaves the Kalman gain exactly unchanged: if
`P0 -> aP0`, `Q -> aQ`, `R -> aR` then `P^- -> aP^-` by induction and
`K = aP^-H^T(a(HP^-H^T+R))^{-1}` is invariant, so the state estimate is invariant, so the
Jacobian evaluated at that estimate is invariant and the induction closes. The estimator
therefore depends on the RATIO alone. Sweeping `q_scale` with `R` pinned at `r_base` and `P0`'s
state block pinned at `r_base` covers that ratio completely.

**The one exception, stated rather than hidden:** `P0`'s control block has no `r_base`
counterpart -- it is a prior on the initial steering and acceleration, not a measurement
covariance -- so it sits outside the invariance group. It is held at the training-split
variance of the recovered controls and is NOT tuned. It affects only the first few steps of
each window.

**Selection criterion: training-split ADE.** Reported as a full curve over the grid, not as a
single winner, so a reader can see whether the minimum is sharp or the curve is flat -- which
is the difference between "this value matters" and "anything in this decade would do".

**The known-model residual is recorded along the same curve, and is NOT used to select.** The
smoother is selected on ADE and then reported on Dyn.-K and the inequality metrics, so a reader
is entitled to ask whether a different criterion would have flattered it. `CRITERION_SENSITIVITY`
answers that with the q_scale the residual would have chosen and how many decades away it sits.
Measured rather than argued, and recorded whichever way it falls.

**Pooled over all five predictor seeds, per (panel, family).** The noise covariance describes a
predictor family's error statistics, and those differ by family, so one value per family is the
honest granularity. Pooling the seeds rather than tuning on seed 0 removes the question of why
seed 0. No per-seed tuning: one `q_scale` per (panel, family) is applied to all five seeds.

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
    return {
        "ind": {
            "data_dir": data_dir,
            "predictor_root": predictor_root,
            "dt": 0.2,
        },
    }


def _physics(panel: str):
    """The kinematic bicycle known model."""
    del panel
    return KinematicBicycle()


def _params() -> jax.Array:
    """`L_REF`, the same reference wheelbase the metric stencil scores every row against."""
    return jnp.asarray([L_REF], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# training-split windows
# ---------------------------------------------------------------------------


def _train_windows(panel: str, cfg: dict, spec: dict, cap: int) -> dict[str, jax.Array]:
    """Build TRAIN-split windows. The split name is hardcoded; there is no flag to change it."""
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
        # A fixed stride rather than a random draw: the subsample is then reproducible without
        # carrying a key, and it spans the split rather than clustering at its start.
        idx = jnp.arange(0, n, max(1, n // cap))[:cap]
        w = {k: v[idx] for k, v in w.items() if hasattr(v, "shape") and v.shape[0] == n}
    return w


def _context(panel: str, w: dict[str, jax.Array]) -> jax.Array:
    del panel
    return jax.vmap(assemble_context)(w["context"], w["metadata"])


def _forward(predictor, context: jax.Array, chunk: int) -> jax.Array:
    out = [jax.vmap(predictor)(context[i:i + chunk]) for i in range(0, context.shape[0], chunk)]
    return jnp.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# the three base diagonals
# ---------------------------------------------------------------------------


def _known_step(physics, x: jax.Array, u: jax.Array, params: jax.Array, dt: float) -> jax.Array:
    """One Heun step of the KNOWN model -- the same step the filter propagates with."""
    k1 = physics.vector_field(x, u, params, 0.0)
    k2 = physics.vector_field(x + dt * k1, u, params, 0.0)
    return x + 0.5 * dt * (k1 + k2)


def _model_mismatch(physics, gt: jax.Array, params: jax.Array, dt: float
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-channel one-step known-model residual, control increments, and control variance.

    Measured on GROUND-TRUTH training trajectories under the known model's OWN analytic control
    inverse -- the same inverse the filter initialises from.
    """
    def per_window(x: jax.Array):
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
    return float(jnp.mean(jnp.linalg.norm(x[..., :2] - gt[..., :2], axis=-1)))


def _known_residual(physics, x: jax.Array, params: jax.Array, dt: float) -> float:
    """Mean one-step KNOWN-model residual of a trajectory batch, under the model's own inverse.

    This is Dyn.-K in the shape the smoother actually targets, and it is recorded ALONGSIDE the
    ADE curve rather than used to select. **Why it is here at all:** the selection criterion is
    ADE, and the smoother is then reported on Dyn.-K and the inequality metrics. If the
    ADE-optimal covariance were far from the Dyn.-K-optimal one, that presentation would need
    defending. Recording both settles whether the question is live instead of arguing it.
    """
    def per_window(t: jax.Array) -> jax.Array:
        u = jax.vmap(physics.known_control_prior, in_axes=(0, 0, None, None))(
            t[:-1], t[1:], params, dt)
        nxt = jax.vmap(_known_step, in_axes=(None, 0, 0, None, None))(
            physics, t[:-1], u, params, dt)
        return jnp.linalg.norm(nxt - t[1:], axis=-1)

    return float(jnp.mean(jax.vmap(per_window)(x)))


def _build(physics, params, dt, q_state, q_ctrl, r, p0) -> KinodynamicSmoother:
    return KinodynamicSmoother(
        physics=physics, params=params, dt=dt,
        q_diag=jnp.concatenate([q_state, q_ctrl]), r_diag=r, p0_diag=p0)


def _floor(v: jax.Array) -> jax.Array:
    """Keep a channel that moved by exactly zero on the training split from zeroing the matrix.

    This is a POSITIVE-DEFINITENESS guard on a covariance diagonal, not a tolerance and not a
    convergence test. It fires only on a channel with no measured variation at all.
    """
    return jnp.maximum(v, 1e-12)


def _tune_family(panel: str, cfg: dict, family: str, grid: jax.Array,
                 cap: int, chunk: int, verbose: bool) -> dict:
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
    # The smoother, like every other correction row, sees the anchor state prepended.
    meas = jnp.concatenate([x0_all[:, None, :], pred], axis=1)
    gt_full = jnp.concatenate([x0_all[:, None, :], gt_all], axis=1)

    r_base = _floor(jnp.mean((pred - gt_all) ** 2, axis=(0, 1)))
    q_state_base, q_ctrl_base, ctrl_var = _model_mismatch(physics, gt_full, params, dt)
    q_state_base, q_ctrl_base, ctrl_var = _floor(q_state_base), _floor(q_ctrl_base), _floor(ctrl_var)
    # P0's state block sits at the measurement scale, which is what makes the single-scalar
    # sweep complete; its control block is the prior described in the module docstring.
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

    # The Dyn.-K-optimal setting, recorded but NOT used to select. If it sits far from the
    # ADE-optimal one, the choice of criterion is doing real work and has to be defended; if
    # the two are close, the question dissolves. Either way it is measured rather than argued.
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "inD-preprocessed" / "v1"))
    ap.add_argument("--predictor-root", default=str(ROOT / "outputs" / "ind" / "predictors"))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--max-windows", type=int, default=2000,
                    help="cap on TRAIN windows per seed before pooling")
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
