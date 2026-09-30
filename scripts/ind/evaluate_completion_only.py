"""Completion-only appendix: attributing the inD ADE rise between the COMPLETION step and
the CORRECTION step.

This is an attribution run, not a stopping-tolerance sweep. Three arms on frozen models,
one flag apart:

  1. full      -- MaDE plug-and-play, correction_mode="eval_adaptive" (the published row)
  2. completion-- correction_mode="it_only": x is re-derived through T_theta from the
                  inferred control and then NEVER steered toward feasibility
  3. raw       -- the predictor's own output, the floor

"it_only" is the exact arm needed, not an approximation: MaDECell.__call__ computes
x_pred = augmented_dynamics.integrate(...) and passes THAT into the corrector, and
Corrector.__call__ returns (x_pred, u) unchanged under that mode. So arm 2 is I -> T with C
removed, not C run to a loose tolerance.

THE ATTRIBUTION. With ADE_raw <= ADE_completion <= ADE_full, the share of the raw->full
rise carried by completion is (ADE_completion - ADE_raw) / (ADE_full - ADE_raw). If that is
most of it, the cost is intrinsic to projecting onto the kinematic manifold and no corrector
tolerance can recover it. If it is small, the corrector is the lever. The ordering is NOT
assumed -- it is measured and reported, and a violation of it is reported rather than hidden.

BOTH WINDOW SETS are computed: unfiltered (the full test split) and filtered (moving agents
only, 0.5 m net displacement over the horizon, the population the paper reports on). Neither
is presented as a correction of the other.

Usage:
  python scripts/ind/evaluate_completion_only.py --out outputs/ind/completion_only.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import jax, numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "src"))

WINDOW_SPEC = {"history": 10, "horizon": 15, "stride": 5, "dt": 0.2}
MADE_ROOTS = {0: "eval", 1: "eval_made1", 2: "eval_made2"}
THRESH_M = 0.5
_PCTS = [25, 50, 75, 95, 99]
_TRIM = 0.05


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


E = _load("_e05x3", Path(__file__).resolve().parent / "eval_lib.py")
from made.evaluation import metrics as M  # noqa: E402
from made.upstream.stage_training import apply_made_trajectory_with_controls  # noqa: E402


def stats(a):
    a = np.asarray(a).reshape(-1); f = a[np.isfinite(a)]
    n = f.size; k = int(np.floor(n * _TRIM))
    tr = np.sort(f)[k:n - k] if n - 2 * k > 0 else f
    return {"n": int(a.size), "mean": float(np.mean(f)),
            **{f"p{p}": float(np.percentile(f, p)) for p in _PCTS},
            "trimmed_mean_5pct": float(np.mean(tr))}


def made_rows(made_model, x_pred, metadata, x0, dt, mode):
    """As eval_lib._made_pnp_rows, but with the correction mode exposed."""
    params = made_model.params_from_metadata(metadata)
    return jax.vmap(
        lambda xp, p, z: apply_made_trajectory_with_controls(
            made_model.cell, xp, p, dt, correction_mode=mode, x0=z)
    )(x_pred, params, x0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data" / "inD-preprocessed" / "v1"))
    ap.add_argument("--families", default="lstm,ssm,transformer")
    ap.add_argument("--pred-seeds", default="0,1,2,3,4")
    ap.add_argument("--made-seeds", default="0,1,2")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--threshold-m", type=float, default=THRESH_M)
    ap.add_argument(
        "--predictor-root", default=str(HERE / "outputs" / "ind" / "predictors"),
        help="Predictor run directory.",
    )
    ap.add_argument(
        "--made-root", default=str(HERE / "outputs" / "ind" / "made" / "seed{ms}" / "checkpoints"),
        help="MaDE checkpoint template with {ms} for the seed.",
    )
    ap.add_argument("--out", default=str(HERE / "outputs" / "ind" / "completion_only.json"))
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    pseeds = [int(s) for s in a.pred_seeds.split(",") if s]
    mseeds = [int(s) for s in a.made_seeds.split(",") if s]

    from made.baselines.clamp_baseline import ClampBaseline
    from made.data.ind_data import create_ind_data_source
    from made.data.window_dataset import make_prediction_windows
    from made.evaluation.real_data_eval import load_train_envelope_and_residual
    from made.physics.constraints import inD_physical_constraints
    from made.upstream.factory import load_predictor

    dt = float(WINDOW_SPEC["dt"])
    data_dir = str(Path(a.data_dir).expanduser().resolve())
    states, metadata, lengths = create_ind_data_source(
        data_dir, "test", use_stub=False, stub_num_trajectories=0,
        stub_trajectory_length=0, stub_seed=0)
    kw = dict(history=WINDOW_SPEC["history"], horizon=WINDOW_SPEC["horizon"],
              stride=WINDOW_SPEC["stride"])
    sets = {
        "unfiltered": make_prediction_windows(states, lengths, metadata, **kw),
        "filtered": make_prediction_windows(states, lengths, metadata,
                                            min_displacement_m=a.threshold_m, **kw),
    }
    envelope, gt_residual, _ = load_train_envelope_and_residual(
        data_dir, use_stub=False, dt=dt, smoke_seed=0, split="train")
    # One constraint object, read by the clamp projection AND the metric.
    phys = inD_physical_constraints()
    clamp_box = phys
    clamp_model = ClampBaseline(constraints=clamp_box)
    for k, w in sets.items():
        print(f"{k}: {int(w['context'].shape[0])} windows", flush=True)

    res = {
        "artifact": "Completion-only appendix -- ADE attribution between the completion and "
                    "correction steps",
        "ade_fde_definition": "position: Euclidean distance between returned and recorded (x, y), "
                               "state indices 0 and 1, in metres; FDE at the last step",
        "design": ("An attribution run, not a stopping-tolerance sweep. Arm 2 removes the "
                   "corrector (correction_mode='it_only'), so the state is re-derived through "
                   "T_theta from the inferred control and never steered toward feasibility."),
        "window_sets": {"unfiltered": "the full test split",
                        "filtered": (f"moving agents only, GT net displacement > "
                                     f"{a.threshold_m} m over the horizon; what the paper "
                                     "reports")},
        "no_directional_claim": ("Filtered figures are NOT conservative, NOT a lower bound "
                                 "and NOT a floor a retrain would beat. Both sides of every "
                                 "comparison shift under the filter."),
        "threshold_m": a.threshold_m,
        "constraint_set": "published",
        "constraint_set_constructor": "inD_physical_constraints()",
        "clamp_and_metric_share_one_object": clamp_box is phys,
        "predictor_root": a.predictor_root,
        "made_root": a.made_root,
        "cells": [],
    }
    t0 = time.time()
    made_cache: dict[int, object] = {}

    for wname, win in sets.items():
        ctx = E._assemble_batch_context(win["context"], win["metadata"])
        x_gt, x0, meta = win["future"], win["context"][:, -1, :], win["metadata"]
        nw = int(win["context"].shape[0])
        for fam in fams:
            for ps in pseeds:
                pred, _ = load_predictor(f"{a.predictor_root}/{fam}_stage1_seed{ps}")
                x_pred = E._chunked_predictor_forward(pred, ctx, a.chunk_size)
                arms = [("raw", None, None, x_pred, None),
                        ("clamp", None, None, E._clamp_rows(clamp_model, x_pred), None)]
                for ms in mseeds:
                    if ms not in made_cache:
                        made_path = a.made_root.format(ms=ms)
                        made_cache[ms] = E._load_frozen_made(made_path, False, 0)
                    xf, uf = made_rows(made_cache[ms], x_pred, meta, x0, dt, "eval_adaptive")
                    xc, uc = made_rows(made_cache[ms], x_pred, meta, x0, dt, "it_only")
                    arms.append(("made_full", ms, MADE_ROOTS[ms], xf, uf))
                    arms.append(("made_completion_only", ms, MADE_ROOTS[ms], xc, uc))
                for row, ms, root, x_row, u_row in arms:
                    t = time.time()
                    x_full = E._full_sequence(x0, x_row)
                    u_kb = E._derive_dynamics_controls(x_full, dt)
                    # A row that emits controls is scored on its own; a row that emits none
                    # is scored on recovered ones. Position-only (x, y) ADE/FDE.
                    m = E._row_metrics(
                        x_row, x_gt, u_row if u_row is not None else u_kb, x_full,
                        u_row if u_row is not None else u_kb, envelope, phys, gt_residual,
                        dt, u_dyn_kb=u_kb if u_row is not None else None,
                        position_displacement=True)
                    # The bare metric key above is the EMITTED figure for rows that emit
                    # controls; the RECOVERED figure is kept beside it as a labelled
                    # diagnostic.
                    if u_row is not None:
                        from made.evaluation.metrics import compute_inequality_dual
                        recovered = compute_inequality_dual(x_row, u_kb, envelope, phys)
                        for k, v in recovered.items():
                            if "inequality" in k:
                                m[f"{k}__recovered_controls"] = v
                    per_ade = np.asarray(jax.device_get(
                        jax.vmap(M._trajectory_position_ade)(x_row, x_gt)))
                    label = (f"{wname}/{row}/{fam}/pseed{ps}"
                             + (f"/mseed{ms}" if ms is not None else ""))
                    res["cells"].append({
                        "label": label, "window_set": wname, "row": row, "family": fam,
                        "predictor_seed": ps, "made_seed": ms, "n_windows": nw,
                        "scalar_metrics": {k: float(v) for k, v in m.items()},
                        "ade_distribution": stats(per_ade)})
                    print(f"  {label}: ADE {m['ade']:.4f} FDE {m['fde']:.4f} "
                          f"DynK {m['dynamics_violation']:.5f} "
                          f"ineq {m['inequality_violation_rate_physical']:.4f} "
                          f"({time.time()-t:.0f}s)", flush=True)

    # Attribution, computed per (window set, family, pseed, made seed).
    by = {c["label"]: c for c in res["cells"]}
    attr = []
    for wname in sets:
        for fam in fams:
            for ps in pseeds:
                raw = by.get(f"{wname}/raw/{fam}/pseed{ps}")
                for ms in mseeds:
                    full = by.get(f"{wname}/made_full/{fam}/pseed{ps}/mseed{ms}")
                    comp = by.get(f"{wname}/made_completion_only/{fam}/pseed{ps}/mseed{ms}")
                    if not (raw and full and comp):
                        continue
                    ar = raw["scalar_metrics"]["ade"]
                    af = full["scalar_metrics"]["ade"]
                    ac = comp["scalar_metrics"]["ade"]
                    total = af - ar
                    ordered = ar <= ac <= af
                    attr.append({
                        "window_set": wname, "family": fam, "predictor_seed": ps,
                        "made_seed": ms,
                        "ade_raw": ar, "ade_completion_only": ac, "ade_full": af,
                        "rise_total": total,
                        "rise_from_completion": ac - ar,
                        "rise_from_correction": af - ac,
                        "completion_share_of_rise": (
                            float((ac - ar) / total) if total != 0 else None),
                        "monotone_raw_le_completion_le_full": bool(ordered),
                        "note": (None if ordered else
                                 "ORDERING VIOLATED: the decomposition into two "
                                 "non-negative parts does not hold for this cell; the "
                                 "share is reported but must not be read as a fraction"),
                    })
    res["attribution"] = attr

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {a.out} -- {len(res['cells'])} cells, {time.time()-t0:.0f}s")
    for r in attr:
        s = r["completion_share_of_rise"]
        print(f"  {r['window_set']}/{r['family']}/pseed{r['predictor_seed']}"
              f"/mseed{r['made_seed']}: raw {r['ade_raw']:.4f} -> completion "
              f"{r['ade_completion_only']:.4f} -> full {r['ade_full']:.4f}; completion "
              f"carries {'n/a' if s is None else f'{s*100:.1f}%'}"
              f"{'' if r['monotone_raw_le_completion_le_full'] else '  [ORDERING VIOLATED]'}")


if __name__ == "__main__":
    main()
