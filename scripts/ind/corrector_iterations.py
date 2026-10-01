# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Corrector-iteration counts: how many gradient steps the eval-time adaptive corrector takes
per timestep, on the same windows and models as the latency run (`measure_latency.py`), so the
two describe one configuration: the filtered inD windows (`WINDOW_SPEC`, `THRESH_M`), the
MSE-only predictors, and the frozen MaDE models.

`CorrectorDiagnostics` is only produced by `mode="eval_adaptive"`, which is the mode the
plug-and-play evaluation uses, so nothing here is a special path: `apply_made_trajectory_with_controls`
is called with `return_diagnostics=True` and is otherwise the same call the panel makes.

## What is reported

Per MaDE seed and pooled: total calls, median, 95th percentile and maximum iterations, and
the cap-hit share, together with the cap (`eval_max_steps`) and tolerance (`eval_tol`) those
numbers are relative to — a cap-hit share means nothing without the cap it is hitting.

**One call is one timestep of one window**, not one window: the corrector runs once per row of
the horizon, and the diagnostics leaves have shape `[T]` per window. Counting windows would
undercount by a factor of the horizon.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

from made.utils.jax_setup import configure  # noqa: E402

configure()

from scripts.ind.evaluate import WINDOW_SPEC, THRESH_M  # noqa: E402

DATA_DIR = ROOT / "data/inD-preprocessed/v1"
OUT = ROOT / "outputs/appendix/corrector_iterations.json"
FAMILIES = ("lstm", "ssm", "transformer")
PRED_SEEDS = (0, 1, 2, 3, 4)
MADE_SEEDS = (0, 1, 2)


def _git_sha() -> str:
    """Current git commit hash.

    Returns:
        The ``HEAD`` sha, or ``unknown`` when git is unavailable.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _load_module(name: str, path: Path) -> "ModuleType":
    """Import a Python file as a module registered under ``name``.

    Args:
        name: Module name to register in ``sys.modules``.
        path: Path of the source file.

    Returns:
        The loaded module.
    """
    import importlib.util
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


def _summary(counts: np.ndarray, caps: np.ndarray) -> dict:
    """Summarise corrector iteration counts.

    Args:
        counts: Iteration count per call.
        caps: Boolean flag per call, True when the iteration cap was hit.

    Returns:
        Dict with call count, median, p95, max, mean and cap-hit statistics.
    """
    if counts.size == 0:
        return {"calls": 0}
    return {
        "calls": int(counts.size),
        "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "cap_hits": int(caps.sum()),
        "cap_hit_share": float(caps.mean()),
    }


def main() -> int:
    """Measure corrector iteration counts across families and seeds and write the JSON.

    Returns:
        Process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--pred-seeds", default=",".join(str(s) for s in PRED_SEEDS))
    ap.add_argument("--made-seeds", default=",".join(str(s) for s in MADE_SEEDS))
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="probe only: cap the window count")
    ap.add_argument("--predictor-root", default=str(ROOT / "outputs" / "ind" / "predictors"),
                    help="Directory holding <family>_stage1_seed<N> predictor runs.")
    ap.add_argument(
        "--made-root", default=str(ROOT / "outputs" / "ind" / "made" / "seed{ms}" / "checkpoints"),
        help="Template for the MaDE checkpoint directory, with {ms} for the seed.",
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    predictor_root = ROOT / a.predictor_root
    def made_checkpoints(seed: int) -> Path:
        """Checkpoint directory for a MaDE seed.

        Args:
            seed: MaDE seed.

        Returns:
            The checkpoint directory under the repository root.
        """
        return ROOT / a.made_root.format(ms=seed)

    families = [f.strip() for f in a.families.split(",") if f.strip()]
    pseeds = [int(s) for s in a.pred_seeds.split(",") if s.strip()]
    mseeds = [int(s) for s in a.made_seeds.split(",") if s.strip()]

    E = _load_module("_e05ind_iters", ROOT / "scripts" / "ind" / "eval_lib.py")
    from made.data.ind_data import create_ind_data_source
    from made.data.window_dataset import make_prediction_windows
    from made.models import MaDEModel
    from made.upstream.factory import load_predictor
    from made.upstream.stage_training import apply_made_trajectory_with_controls

    dt = float(WINDOW_SPEC["dt"])
    states, metadata, lengths = create_ind_data_source(
        str(data_dir), "test", use_stub=False,
        stub_num_trajectories=0, stub_trajectory_length=0, stub_seed=0,
    )
    win = make_prediction_windows(
        states, lengths, metadata,
        history=WINDOW_SPEC["history"], horizon=WINDOW_SPEC["horizon"],
        stride=WINDOW_SPEC["stride"], min_displacement_m=THRESH_M,
    )
    context_all = E._assemble_batch_context(win["context"], win["metadata"])
    metadata_all = win["metadata"]
    x0_all = win["context"][:, -1, :]
    if a.limit_windows:
        context_all = context_all[: a.limit_windows]
        metadata_all = metadata_all[: a.limit_windows]
        x0_all = x0_all[: a.limit_windows]
    n_windows = int(context_all.shape[0])
    print(f"{n_windows} filtered windows, dt={dt}", flush=True)

    cells, by_seed, t0 = [], {}, time.time()
    cap = tol = None
    for mseed in mseeds:
        made_model = MaDEModel.from_checkpoint(str(made_checkpoints(mseed)))
        if cap is None:
            corr = made_model.cell.corrector
            cap = int(getattr(corr, "eval_max_steps", -1))
            tol = float(getattr(corr, "eval_tol", float("nan")))
        for family in families:
            for pseed in pseeds:
                predictor, _ = load_predictor(f"{predictor_root}/{family}_stage1_seed{pseed}")
                # The predictor is a single-window callable; batch it through
                # `_chunked_predictor_forward` rather than call it on the whole array (which
                # silently produces the wrong shape instead of failing cleanly).
                x_pred_all = E._chunked_predictor_forward(predictor, context_all, 4096)
                params_all = made_model.params_from_metadata(metadata_all)

                def _one(
                    x_pred: jax.Array, params: Any, x0: jax.Array, _m: Any = made_model
                ) -> tuple[jax.Array, jax.Array]:
                    """Run one trajectory and report its corrector iterations.

                    Args:
                        x_pred: Predicted future states.
                        params: Physics parameters for this window.
                        x0: Seed state.
                        _m: Bound MaDE model.

                    Returns:
                        The iteration count and the cap-hit flag.
                    """
                    _x, _u, diag = apply_made_trajectory_with_controls(
                        _m.cell, x_pred, params, dt,
                        correction_mode="eval_adaptive", x0=x0, return_diagnostics=True)
                    return diag.n_iterations, diag.cap_hit

                n_it, cap_hit = jax.vmap(_one)(x_pred_all, params_all, x0_all)
                counts = np.asarray(n_it).reshape(-1)
                caps = np.asarray(cap_hit).reshape(-1).astype(bool)
                rec = {"family": family, "predictor_seed": pseed, "made_seed": mseed,
                       "windows": n_windows, **_summary(counts, caps)}
                cells.append(rec)
                by_seed.setdefault(mseed, []).append((counts, caps))
                print(f"  mseed{mseed} {family}/pseed{pseed}: calls={rec['calls']} "
                      f"median={rec['median']} p95={rec['p95']} max={rec['max']} "
                      f"cap_hits={rec['cap_hits']}", flush=True)

    per_seed = {}
    all_counts, all_caps = [], []
    for mseed, parts in by_seed.items():
        c = np.concatenate([p[0] for p in parts])
        k = np.concatenate([p[1] for p in parts])
        per_seed[str(mseed)] = _summary(c, k)
        all_counts.append(c)
        all_caps.append(k)
    pooled = _summary(np.concatenate(all_counts), np.concatenate(all_caps))

    payload = {
        "artifact": str(Path(a.out)),
        "what": "Eval-corrector iteration counts and cap hits on the same windows and models "
                "as the latency run.",
        "why": "The correction loop's iteration count is data-dependent, so the batch-1 "
               "latency spread reported alongside it can only be explained by directly "
               "counting how many steps the corrector takes.",
        "one_call_is": "ONE TIMESTEP of one window, not one window: the corrector runs once per "
                       "row of the horizon and the diagnostics leaves have shape [T] per window",
        "cap": cap,
        "tolerance": tol,
        "cap_hit_means": "the adaptive loop exited at eval_max_steps with residual violation "
                         "still above eval_tol, i.e. it was cut off rather than converging",
        "mode": "eval_adaptive -- the only mode that computes these diagnostics, and the mode "
                "the plug-and-play evaluation already uses, so this is not a special path",
        "windows": n_windows,
        "predictor_root": _display(predictor_root),
        "made_roots_resolved": {str(m): _display(made_checkpoints(m))
                                for m in mseeds},
        "provenance": {"git_sha": _git_sha(),
                       "hardware": f"{platform.system()} {platform.release()} | "
                                   f"{platform.machine()} | jax_devices={jax.devices()}",
                       "minutes": round((time.time() - t0) / 60, 1)},
        "pooled": pooled,
        "per_made_seed": per_seed,
        "cells": cells,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\ncap={cap} tol={tol}")
    print(f"pooled: {pooled}")
    for k, v in sorted(per_seed.items()):
        print(f"  seed {k}: {v}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
