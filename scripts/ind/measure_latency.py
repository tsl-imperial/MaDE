# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Latency-only harness: timing rows for raw / clamp / smoother / made_pnp on the
trained predictors and MaDE checkpoints.

Adds only the latency columns and a provenance block to a full inD evaluation pass.

Reuses ``eval_lib.py``'s ``_time_pipeline`` with the same arguments and defaults as the
full pass (``timing_batch_size`` 16, ``timing_repeats`` 5), so figures are comparable.
``made_pnp`` is reconstructed from the same three module-level pieces the full pass closes
over.

Provenance recorded: git commit, hardware string, timing batch size, repeat count, and
number of windows timed over.

Writes only the requested ``--out`` artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from made.utils.jax_setup import configure  # noqa: E402

configure()

import numpy as np  # noqa: E402

DATA_DIR = ROOT / "data/inD-preprocessed/v1"
FAMILIES = ("lstm", "ssm", "transformer")
PRED_SEEDS = (0, 1, 2, 3, 4)
MADE_SEEDS = (0, 1, 2)
OUT = ROOT / "outputs/table_6/latency.json"
# The corrector's eval loop runs to a tolerance, so step size affects iteration count and
# therefore latency. Import the single definition from evaluate.py to avoid mismatch.
from scripts.ind.evaluate import WINDOW_SPEC, THRESH_M as THRESHOLD_M  # noqa: E402

# Panel's step size, checked against WINDOW_SPEC's below.
PANEL_DT: float = 0.2
TIMING_BATCH_SIZE = 16
TIMING_REPEATS = 5


from scripts.common.run_guard import claim_output, require_idle_devices  # noqa: E402



def _machine_state(idle_scope: tuple[int, ...] | None) -> dict:
    """Record what was on the cards when the measurement started (not just a pass/fail flag).

    Captures compute processes, their cards, and each card's sustained utilisation, so a
    reader can judge the figure rather than trust an idle-check boolean.

    Args:
        idle_scope: Device indices the idle check covered, or None for the blanket check.

    Returns:
        Dict with the check scope, compute processes at start, bus ids and utilisation samples.
    """
    import subprocess

    def _q(query: str) -> list[str]:
        """Run one ``nvidia-smi`` query.

        Args:
            query: Query string after ``--query-``.

        Returns:
            The output lines, or a single ``(unavailable: ...)`` line on failure.
        """
        try:
            return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        except Exception as exc:  # pragma: no cover
            return [f"(unavailable: {exc!r})"]

    util = [_q("gpu=index,utilization.gpu,memory.used") for _ in range(5)]
    procs = _q("compute-apps=pid,gpu_bus_id,used_memory")
    named = []
    for line in procs:
        pid = line.split(",", 1)[0].strip()
        try:
            cmd = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except Exception:
            cmd = "(gone)"
        named.append({"row": line, "cmdline": cmd.strip()})
    return {
        "idle_check_scope": list(idle_scope) if idle_scope else "ALL DEVICES (blanket check)",
        "blanket_standard_met": idle_scope is None,
        "compute_processes_at_start": named,
        "bus_id_to_index": _q("gpu=index,gpu_bus_id"),
        "utilisation_samples": util,
    }


def _load_module(name: str, path: Path) -> "ModuleType":
    """Import a Python file as a module registered under ``name``.

    Args:
        name: Module name to register in ``sys.modules``.
        path: Path of the source file.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _display(p: Path) -> str:
    """Format a path relative to the repository root when possible.

    Args:
        p: Path to format.

    Returns:
        The root-relative path, or the path unchanged when outside the root.
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _git_sha() -> str:
    """Current git commit hash.

    Returns:
        The ``HEAD`` sha, or ``unknown`` when git fails.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    """Measure per-trajectory latency of every row and write the JSON report.

    Raises:
        SystemExit: If a precondition for a valid measurement is not met.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--pred-seeds", default=",".join(str(s) for s in PRED_SEEDS))
    ap.add_argument("--made-seeds", default=",".join(str(s) for s in MADE_SEEDS))
    ap.add_argument("--timing-batch-size", type=int, default=TIMING_BATCH_SIZE)
    ap.add_argument("--timing-repeats", type=int, default=TIMING_REPEATS)
    ap.add_argument("--smoother-noise", default=None,
                    help="TRAIN-SPLIT tuning artifact. When given, both smoother arms are "
                         "timed alongside the other rows.")
    ap.add_argument("--idle-devices", default=None,
                    help="Narrow the blanket idle check to these physical device indices, "
                         "e.g. '1'. A disclosed departure from the standard protocol: use "
                         "only when a non-MaDE process that cannot be stopped holds another "
                         "card. What was running is recorded in the artifact either way.")
    ap.add_argument("--predictor-root", default=str(ROOT / "outputs" / "ind" / "predictors"),
                    help="Directory holding <family>_stage1_seed<N> predictor runs.")
    ap.add_argument(
        "--made-root", default=str(ROOT / "outputs" / "ind" / "made" / "seed{ms}" / "checkpoints"),
        help="Template for the MaDE checkpoint directory, with {ms} for the seed.",
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    predictor_root = ROOT / args.predictor_root
    def made_checkpoints(seed: int) -> Path:
        """Checkpoint directory for a MaDE seed.

        Args:
            seed: MaDE seed.

        Returns:
            The checkpoint directory under the repository root.
        """
        return ROOT / args.made_root.format(ms=seed)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    pseeds = [int(s) for s in args.pred_seeds.split(",") if s.strip()]
    mseeds = [int(s) for s in args.made_seeds.split(",") if s.strip()]

    # Refuse rather than warn, checked before anything loads, so a busy machine costs a
    # second rather than ten minutes and a wrong table.
    #
    # `--idle-devices` narrows the blanket check (disclosed departure from standard protocol);
    # leave unset for the full check the paper's protocol requires. Meant for a non-MaDE
    # process that cannot be stopped holding another card. Either way what was running is
    # recorded in the artifact.
    idle_scope = (tuple(int(d) for d in args.idle_devices.split(",") if d.strip())
                  if args.idle_devices else None)
    require_idle_devices(devices=idle_scope)
    machine_state = _machine_state(idle_scope)

    E = _load_module("_e05ind_latency", ROOT / "scripts" / "ind" / "eval_lib.py")

    from made.baselines.clamp_baseline import ClampBaseline
    from made.data.ind_data import create_ind_data_source
    from made.data.window_dataset import make_prediction_windows
    from made.models import MaDEModel
    from made.physics.constraints import (
        drop_position_bounds,
        kinematic_bicycle_constraints,
    )
    from made.upstream.factory import load_predictor

    dt = float(WINDOW_SPEC["dt"])
    # A measurement at the wrong step size would silently describe a different configuration.
    if abs(dt - PANEL_DT) > 1e-12:
        raise SystemExit(
            f"STEP-SIZE ASSERTION FAILED: measuring at dt={dt} against a panel at "
            f"dt={PANEL_DT}. This is the defect that withdrew the previous latency figures. "
            f"Refusing to produce a number that describes a different configuration."
        )
    states, metadata, lengths = create_ind_data_source(
        str(data_dir), "test", use_stub=False,
        stub_num_trajectories=0, stub_trajectory_length=0, stub_seed=0,
    )
    win = make_prediction_windows(
        states, lengths, metadata,
        history=WINDOW_SPEC["history"], horizon=WINDOW_SPEC["horizon"],
        stride=WINDOW_SPEC["stride"], min_displacement_m=THRESHOLD_M,
    )
    context_all = E._assemble_batch_context(win["context"], win["metadata"])
    metadata_all = win["metadata"]
    x0_all = win["context"][:, -1, :]
    n_windows = int(context_all.shape[0])
    timing_batch = min(args.timing_batch_size, n_windows)
    clamp_model = ClampBaseline(
        constraints=drop_position_bounds(kinematic_bicycle_constraints())
    )

    # Built once per family, outside the timing loop.
    smoother_arms = {}
    if args.smoother_noise:
        for family in families:
            smoother_arms[family] = E._load_smoother_arms(
                args.smoother_noise, family, "ind", dt, ("smoother", "smoother_dyn"))

    print(f"{n_windows} filtered windows; timing batch {timing_batch}, "
          f"{args.timing_repeats} repeats"
          f"{'; smoother arms included' if smoother_arms else ''}", flush=True)

    rows: list[dict] = []
    started = time.time()
    for family in families:
        for pseed in pseeds:
            predictor, _ = load_predictor(f"{predictor_root}/{family}_stage1_seed{pseed}")

            timing = E._time_pipeline(
                lambda ctx: predictor(ctx),
                (context_all[0],), (context_all[:timing_batch],),
                repeats=args.timing_repeats,
            )
            rows.append({"row": "raw", "family": family, "predictor_seed": pseed,
                         "made_seed": None, **timing})

            timing = E._time_pipeline(
                lambda ctx: E._clamp_trajectory(clamp_model, predictor(ctx)),
                (context_all[0],), (context_all[:timing_batch],),
                repeats=args.timing_repeats,
            )
            rows.append({"row": "clamp", "family": family, "predictor_seed": pseed,
                         "made_seed": None, **timing})

            # Both tuned arms differ only in covariances; a reader will want to know
            # whether that costs anything.
            if smoother_arms:
                for row_name, (sm, tuning) in smoother_arms[family].items():
                    timing = E._time_pipeline(
                        lambda ctx, _s=sm: E._smoother_trajectory(
                            _s, ctx[-1, :E._SMOOTHER_STATE_DIM], predictor(ctx)),
                        (context_all[0],), (context_all[:timing_batch],),
                        repeats=args.timing_repeats,
                    )
                    rows.append({"row": row_name, "family": family, "predictor_seed": pseed,
                                 "made_seed": None,
                                 "smoother_criterion": tuning["criterion"],
                                 "smoother_q_scale": tuning["q_scale"], **timing})

            for mseed in mseeds:
                made_model = MaDEModel.from_checkpoint(str(made_checkpoints(mseed)))

                # Same three lines the full pass closes over.
                def _made_pnp_pipeline(
                    ctx: jax.Array, meta: jax.Array, x0: jax.Array, _m: Any = made_model
                ) -> jax.Array:
                    """Predict one window and apply the frozen MaDE correction.

                    Args:
                        ctx: Assembled context for the window.
                        meta: Window metadata.
                        x0: Seed state.
                        _m: Bound MaDE model.

                    Returns:
                        The corrected future states.
                    """
                    x_pred = predictor(ctx)
                    params = _m.params_from_metadata(meta)
                    x_corr, _u = E._made_pnp_single(_m.cell, x_pred, params, x0, dt)
                    return x_corr

                timing = E._time_pipeline(
                    _made_pnp_pipeline,
                    (context_all[0], metadata_all[0], x0_all[0]),
                    (context_all[:timing_batch], metadata_all[:timing_batch],
                     x0_all[:timing_batch]),
                    repeats=args.timing_repeats,
                )
                rows.append({"row": "made_pnp", "family": family, "predictor_seed": pseed,
                             "made_seed": mseed, **timing})
            print(f"  {family}/pseed{pseed} done ({time.time()-started:.0f}s)", flush=True)

    def _agg(row: str, key: str) -> dict:
        """Summarise one latency key over the cells of a row.

        Args:
            row: Row name.
            key: Latency field to aggregate.

        Returns:
            Dict with count, mean, population std, median, min and max.
        """
        vals = [r[key] for r in rows if r["row"] == row]
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std_population": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    report = {
        # Derived from the output path.
        "artifact": _display(Path(args.out)),
        "what": f"Latency only, measured on predictors at {_display(predictor_root)} and "
                f"the frozen MaDE models.",
        "timing_code": "scripts/ind/eval_lib.py::_time_pipeline, imported and called with the "
                       "same arguments and defaults the full pass uses, so figures are "
                       "comparable across the two",
        "provenance": {
            "git_sha": _git_sha(),
            "hardware": f"{platform.system()} {platform.release()} | {platform.machine()} | "
                        f"jax_devices={jax.devices()}",
            # `jax.devices()` reports the logical id, always 0 under CUDA_VISIBLE_DEVICES
            # remapping, so the physical index is recorded explicitly.
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu": (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0] or None,
            "device_rule": "the paper's latency figures were measured on a single idle "
                           "physical GPU with no other process resident on it",
            "timing_batch_size": timing_batch,
            "timing_repeats": args.timing_repeats,
            "num_windows_timed_over": n_windows,
            "window_spec": WINDOW_SPEC,
            "dt_asserted_against_panel": {"measured_at": dt, "panel": PANEL_DT},
            "MACHINE_STATE_AT_START": machine_state,
            "baseline_figures_note": (
                "The baselines ARE re-timed here, in the same single-GPU run as the MaDE "
                "rows, so every row in this file comes from one measurement session."
            ),
            "threshold_m": THRESHOLD_M,
            "predictor_root": _display(predictor_root),
            "made_root": args.made_root,
            "made_roots_resolved": {str(m): _display(made_checkpoints(m))
                                    for m in mseeds},
            "checkpoint_selection": "best-validation",
        },
        "population": {
            "predictor_runs": len(families) * len(pseeds),
            "made_checkpoints": len(mseeds),
            "cells_raw": len(families) * len(pseeds),
            "cells_clamp": len(families) * len(pseeds),
            "cells_made_pnp": len(families) * len(pseeds) * len(mseeds),
            "note": "the published amortised range was stated over fifteen predictor runs; "
                    "raw and clamp here are over that same count, and made_pnp is over "
                    "predictor runs x MaDE checkpoints",
        },
        "aggregates": {
            row: {
                "latency_batch1_median_s": _agg(row, "latency_batch1_median_s"),
                "latency_batched_amortised_s": _agg(row, "latency_batched_amortised_s"),
            }
            for row in sorted({r["row"] for r in rows})
        },
        "wall_clock_minutes": (time.time() - started) / 60.0,
        "cells": rows,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    # A second writer refuses here rather than overwriting silently.
    with claim_output(out_path):
        out_path.write_text(json.dumps(report, indent=2) + "\n")

    print()
    for row in sorted({r["row"] for r in rows}):
        a = report["aggregates"][row]
        b1, ba = a["latency_batch1_median_s"], a["latency_batched_amortised_s"]
        print(f"  {row:9s} n={b1['n']:2d}  batch-1 median {b1['mean']:.6f}s "
              f"(range {b1['min']:.6f}-{b1['max']:.6f})  "
              f"amortised {ba['mean']:.6f}s (range {ba['min']:.6f}-{ba['max']:.6f})")
    print(f"\nwrote {_display(out_path)}")


if __name__ == "__main__":
    main()
