"""Filtered inD evaluation on existing frozen checkpoints — a deliverable, not a diagnostic.

PROTOCOL, which must travel with every number this produces:
    Both the upstream predictors and the MaDE checkpoints were TRAINED on the UNFILTERED
    inD distribution, which is about 85% stationary windows. They are EVALUATED here on
    MOVING AGENTS ONLY (ground-truth net displacement > 0.5 m over the 3.0 s horizon).
    Training broad and evaluating on the slice of interest; nothing is retrained.

NO DIRECTIONAL CLAIM is made or implied. These figures are not conservative, not a lower
bound, and not a floor a retrained model would beat. Both sides of every comparison shift
under the filter and the direction is unknown until a retrain runs.

Usage:
  python scripts/ind/evaluate.py --out outputs/ind/panel.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "src"))

WINDOW_SPEC = {"history": 10, "horizon": 15, "stride": 5, "dt": 0.2}
MADE_ROOTS = {0: "eval", 1: "eval_made1", 2: "eval_made2"}
THRESH_M = 0.5
PCTS = [25, 50, 75, 95, 99]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


E = _load("_e05ind", Path(__file__).resolve().parent / "eval_lib.py")
from made.evaluation import metrics as M  # noqa: E402


def stats(a):
    a = np.asarray(a).reshape(-1)
    f = a[np.isfinite(a)]
    n = f.size
    k = int(np.floor(n * 0.05))
    trimmed = np.sort(f)[k:n - k] if n - 2 * k > 0 else f
    return {"n": int(a.size), "mean": float(np.mean(f)),
            **{f"p{p}": float(np.percentile(f, p)) for p in PCTS},
            "trimmed_mean_5pct": float(np.mean(trimmed))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data" / "inD-preprocessed" / "v1"))
    ap.add_argument("--families", default="lstm,ssm,transformer")
    ap.add_argument("--pred-seeds", default="0,1,2,3,4")
    ap.add_argument("--made-seeds", default="0,1,2")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument(
        "--threshold-m",
        type=float,
        default=THRESH_M,
        help=(
            "Track-level displacement threshold in metres. A NEGATIVE VALUE DISABLES THE "
            "FILTER ENTIRELY and evaluates every window. 0.0 does NOT disable it: "
            "make_prediction_windows still takes its filtering path and drops windows with "
            "exactly zero displacement, leaving only a fraction of the full split."
        ),
    )
    ap.add_argument(
        "--predictor-root",
        default=str(HERE / "outputs" / "ind" / "predictors"),
        help="Directory holding <family>_stage1_seed<N> predictor runs.",
    )
    ap.add_argument(
        "--made-root",
        default=str(HERE / "outputs" / "ind" / "made" / "seed{ms}" / "checkpoints"),
        help="Template for the MaDE checkpoint directory, with {ms} for the seed.",
    )
    ap.add_argument(
        "--split", default="test", choices=("train", "val", "test"),
        help="Which inD split to evaluate. The envelope and residual reference stay pinned "
             "to the train split either way.",
    )
    ap.add_argument("--smoother-noise", default=None,
                    help="Path to the TRAIN-SPLIT tuning artifact from "
                         "scripts/ind/tune_smoother.py. When given, the panel gains the "
                         "`smoother` (ADE-tuned) and `smoother_dyn` (known-model-residual-"
                         "tuned) rows. Omitted, the panel is exactly what it was.")
    ap.add_argument("--out", default=str(HERE / "outputs" / "ind" / "panel.json"))
    ap.add_argument("--per-window-dir", default=None,
                    help="Directory for the per-window vector npz of every cell. Defaults to "
                         "<out>.per_window/ beside the artifact; pass an empty string to disable.")
    a = ap.parse_args()
    fams = a.families.split(","); pseeds = [int(s) for s in a.pred_seeds.split(",")]
    mseeds = [int(s) for s in a.made_seeds.split(",")]

    from made.baselines.clamp_baseline import ClampBaseline
    from made.data.window_dataset import make_prediction_windows
    from made.data.ind_data import create_ind_data_source
    from made.evaluation.real_data_eval import load_train_envelope_and_residual
    from made.physics.constraints import inD_physical_constraints
    from made.physics.kinematic_bicycle import KinematicBicycle
    from made.upstream.factory import load_predictor
    from scripts.common.run_guard import claim_output

    dt = float(WINDOW_SPEC["dt"])
    if a.per_window_dir == "":
        per_window_dir = None
    else:
        per_window_dir = Path(a.per_window_dir or (str(a.out) + ".per_window"))
        per_window_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(Path(a.data_dir).expanduser().resolve())
    states, metadata, lengths = create_ind_data_source(data_dir, a.split, use_stub=False,
        stub_num_trajectories=0, stub_trajectory_length=0, stub_seed=0)

    unf = make_prediction_windows(states, lengths, metadata, history=WINDOW_SPEC["history"],
        horizon=WINDOW_SPEC["horizon"], stride=WINDOW_SPEC["stride"])
    # A negative threshold means no filter at all -- min_displacement_m=None -- rather than a
    # threshold of zero, which still takes the filtering path.
    win = (
        unf
        if a.threshold_m < 0
        else make_prediction_windows(
            states, lengths, metadata, history=WINDOW_SPEC["history"],
            horizon=WINDOW_SPEC["horizon"], stride=WINDOW_SPEC["stride"],
            min_displacement_m=a.threshold_m)
    )
    n_unf, n_fil = int(unf["context"].shape[0]), int(win["context"].shape[0])
    # Assert the POPULATION before any number is computed from it. --threshold-m 0.0 would
    # otherwise silently produce a small filtered population labelled unfiltered.
    if a.threshold_m < 0 and n_fil != n_unf:
        raise SystemExit(
            f"POPULATION ASSERTION FAILED: --threshold-m {a.threshold_m} means unfiltered, but "
            f"{n_fil} of {n_unf} windows survived. Refusing to produce a mislabelled artifact."
        )
    tracks = np.unique(np.asarray(win["traj_index"]))
    _, per_track = np.unique(np.asarray(win["traj_index"]), return_counts=True)
    pt = np.sort(per_track)[::-1]
    print(f"windows {n_unf} -> {n_fil} ({(1-n_fil/n_unf)*100:.1f}% removed); "
          f"{tracks.size} distinct tracks; per-track median {np.median(pt):.0f} max {pt.max()}",
          flush=True)

    ctx = E._assemble_batch_context(win["context"], win["metadata"])
    x_gt, x0, meta = win["future"], win["context"][:, -1, :], win["metadata"]
    envelope, gt_residual, _ = load_train_envelope_and_residual(
        data_dir, use_stub=False, dt=dt, smoke_seed=0, split="train")
    # One constraint object, read by the clamp projection AND the metric, so
    # clamp_and_metric_share_one_object is true by construction, not just of the values.
    scored = inD_physical_constraints()
    clamp_box = scored          # THE SAME OBJECT, deliberately, not an equal copy
    phys = scored
    clamp_model = ClampBaseline(constraints=clamp_box)
    clamp_box_note = (
        "clamp projects onto inD_physical_constraints(), the same set the inequality "
        "metric scores against."
    )

    # Filled by the cell loop below; declared here because the payload that records it is built first.
    _resolved_made_roots: dict = {}
    res = {
        "artifact": "Filtered inD evaluation on existing frozen checkpoints",
        "ade_fde_definition": "position: Euclidean distance between returned and recorded (x, y), "
                               "state indices 0 and 1, in metres; FDE at the last step",
        "split": a.split,
        "proximity_gamma": 0.0,
        "constraint_set": "published",
        "constraint_set_constructor": "inD_physical_constraints()",
        "clamp_and_metric_share_one_object": clamp_box is scored,
        "clamp_box": "ind_physical",
        "clamp_box_note": clamp_box_note,
        "status": "DELIVERABLE. A publishable configuration, not a diagnostic.",
        "protocol": ("Predictors AND MaDE checkpoints were TRAINED on the UNFILTERED inD "
                     "distribution, ~85% stationary windows. They are EVALUATED here on "
                     "MOVING AGENTS ONLY: ground-truth net displacement > "
                     f"{a.threshold_m} m over the {WINDOW_SPEC['horizon']*dt:.1f} s horizon. "
                     "Nothing was retrained. Train broad, evaluate on the slice of interest."),
        "no_directional_claim": ("These figures are NOT conservative, NOT a lower bound, and "
                                 "NOT a floor a retrained model would beat. Both sides of every "
                                 "comparison shift under the filter; the direction is unknown "
                                 "until a retrain runs."),
        "threshold_m": a.threshold_m,
        "predictor_root": a.predictor_root,
        "made_root": a.made_root,
        "made_roots_resolved": _resolved_made_roots,
        "inequality_controls": "derived (single convention: recovered from the states through "
                               "the inverse known model when the row emits none)",
        "inequality_controls_note": (
            "'none' takes compute_inequality_dual's documented state-only branch for rows that "
            "emit no controls. 'derived' scores raw and clamp against control bounds they "
            "never emit."
        ),
        "checkpoint_selection": "best-validation; the loader restores early_stop_best_step "
                                "rather than the highest step",
        "windows_unfiltered": n_unf, "windows_filtered": n_fil,
        "distinct_tracks_filtered": int(tracks.size),
        "windows_per_track_filtered": {"median": float(np.median(pt)), "max": int(pt.max()),
            "top1_share_pct": float(pt[0]/pt.sum()*100),
            "top5_share_pct": float(pt[:5].sum()/pt.sum()*100),
            "tracks_for_half_the_windows": int(np.searchsorted(np.cumsum(pt), pt.sum()/2)+1)},
        "cells": [],
    }

    made_cache: dict[int, object] = {}
    t0 = time.time()
    for fam in fams:
        for ps in pseeds:
            pred_root = a.predictor_root
            pred, _ = load_predictor(f"{pred_root}/{fam}_stage1_seed{ps}")
            x_pred = E._chunked_predictor_forward(pred, ctx, a.chunk_size)
            arms = [("raw", "eval", None, x_pred, None),
                    ("clamp", "eval", None, E._clamp_rows(clamp_model, x_pred), None)]
            # Two smoother arms: training-split tuning put the ADE-optimal and the
            # known-model-residual-optimal covariances several decades apart in q_scale, with
            # Dyn.-K differing between them by a large factor. Built per family since the
            # covariances describe a predictor family's error statistics; `u_row` is `None`
            # here since, like raw and clamp, the smoother emits no controls until scored below.
            if a.smoother_noise:
                for _r, (_sm, _t) in E._load_smoother_arms(
                        a.smoother_noise, fam, "ind", dt,
                        ("smoother", "smoother_dyn")).items():
                    arms.append((_r, f"eval:{_t['criterion']}", None,
                                 *E._smoother_rows(_sm, x0, x_pred)))
            for ms in mseeds:
                if ms not in made_cache:
                    made_path = a.made_root.format(ms=ms)
                    # NOT MADE_ROOTS: that is the cell-LABEL map ("eval", "eval_made1", ...)
                    # and writing a path into it would relabel every made_pnp cell.
                    _resolved_made_roots[ms] = made_path
                    made_cache[ms] = E._load_frozen_made(made_path, False, 0)
                xc, uc = E._made_pnp_rows(made_cache[ms], x_pred, meta, x0, dt)
                arms.append(("made_pnp", MADE_ROOTS[ms], ms, xc, uc))
            for row, root, ms, x_row, u_row in arms:
                t = time.time()
                x_full = E._full_sequence(x0, x_row)
                u_kb = E._derive_dynamics_controls(x_full, dt)
                # Convention: a row that emits controls is scored on its own; a row that
                # emits none is scored on controls recovered through the inverse known model.
                # MaDE always emits. The smoother also emits (its augmented state carries the
                # controls, smoothed by the RTS pass), so both smoother arms pass their own
                # `u_row`. Raw and clamp emit nothing and keep the recovered controls.
                u_for_ineq = u_row if u_row is not None else u_kb
                # Smoother controls feed the inequality scorer only; dynamics inputs stay on
                # controls recovered through the known model, so the dynamics column stays
                # labelled Dyn.-K correctly.
                is_smoother = row in ("smoother", "smoother_dyn")
                u_for_dyn = u_kb if (u_row is None or is_smoother) else u_row
                u_dyn_kb = None if (u_row is None or is_smoother) else u_kb
                # Position-only (x, y) ADE/FDE.
                m = E._row_metrics(x_row, x_gt, u_for_ineq, x_full,
                                   u_for_dyn, envelope, phys,
                                   gt_residual, dt, u_dyn_kb=u_dyn_kb,
                                   position_displacement=True)
                per_ade = np.asarray(
                    jax.device_get(jax.vmap(M._trajectory_position_ade)(x_row, x_gt)))
                per_fde = np.asarray(
                    jax.device_get(jax.vmap(M._trajectory_position_fde)(x_row, x_gt)))
                label = f"{row}/{fam}/pseed{ps}" + (f"/mseed{ms}" if ms is not None else "")
                # Save the per-window vector for every metric the panel reports, so a later
                # question about their distribution needs no re-run. These are the same
                # vectors `scalar_metrics` reduces from (same per-trajectory primitives, not
                # a second measurement).
                if per_window_dir is not None:
                    kb_phys = KinematicBicycle()
                    kb_par = jnp.asarray([E.L_REF], dtype=jnp.float64)
                    u_ineq = (u_for_ineq if u_for_ineq is not None
                              else jnp.zeros(x_row.shape[:-1] + phys.control_min.shape,
                                             dtype=x_row.dtype))
                    vecs = {
                        "ade": per_ade,
                        "fde": per_fde,
                        # Same control `_row_metrics` reduces this scalar from: for a row
                        # that emits controls, Dyn.-K is still scored on the KB-recovered
                        # ones (`u_dyn_kb`), not the emitted ones.
                        "dynamics_violation": np.asarray(jax.device_get(jax.vmap(
                            lambda xf, uu: M._trajectory_dynamics_violation_known(
                                xf, uu, kb_phys, kb_par, dt))(
                            x_full, u_dyn_kb if u_dyn_kb is not None else u_for_dyn))),
                        "inequality_violation_rate_physical": np.asarray(jax.device_get(jax.vmap(
                            lambda xx, uu: M._trajectory_inequality_violation_rate(
                                xx, uu, phys))(x_row, u_ineq))),
                        "inequality_violation_magnitude_physical": np.asarray(jax.device_get(
                            jax.vmap(lambda xx, uu: M._trajectory_inequality_violation_magnitude(
                                xx, uu, phys))(x_row, u_ineq))),
                    }
                    out_npz = per_window_dir / (label.replace("/", "__") + ".npz")
                    np.savez_compressed(out_npz, **vecs)
                res["cells"].append({"label": label, "row": row, "family": fam,
                    "predictor_seed": ps, "made_seed": ms, "n_windows": n_fil,
                    "scalar_metrics": {k: float(v) for k, v in m.items()},
                    "ade": stats(per_ade), "fde": stats(per_fde)})
                print(f"  {label}: ADE mean {m['ade']:.4f} p50 {stats(per_ade)['p50']:.4f} "
                      f"trim {stats(per_ade)['trimmed_mean_5pct']:.4f} ({time.time()-t:.0f}s)",
                      flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Refuse rather than overwrite a live writer's artifact.
    with claim_output(a.out):
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {a.out} ({len(res['cells'])} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
