"""Score the E01 ladder's inequality and dynamics metrics. No retrain.

Each cell's checkpoint is evaluated with the control routing described in
`scripts/sim/evaluate.py::main_programmatic`:

| metric | controls |
|---|---|
| inequality | emitted where the row emits (all MaDE variants); recovered through the condition's **known** model where it does not |
| Dyn.-K | recovered through the **known** model, every row |
| Dyn.-T | recovered through the **true** model, every row (except the underspecified dynamic bicycle; see `evaluate.py`) |
| Dyn.-L | computed for every MaDE row; for rows with no learned model of their own (clamp, mlp, fab) it is borrowed from the MaDE model for the same system/condition/seed, when available |
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The rows that have no learned model of their own and so borrow MaDE's.
E01_BASELINE_VARIANTS_FOR_DYN_L = ("clamp", "mlp", "fab")

SYSTEMS = ("double_integrator", "unicycle", "kinematic_bicycle", "dynamic_bicycle")
VARIANTS = (
    "made", "made-no-residual", "made-no-corrector", "made-fixed-i", "made-supervised-i",
    "mlp", "fab", "clamp",
)
PRIOR_ONLY_SYSTEM = "dynamic_bicycle"
PRIOR_ONLY_VARIANT = "made-prior-only"

PREFIX = {"double_integrator": "di", "unicycle": "unicycle",
          "kinematic_bicycle": "kinbicycle", "dynamic_bicycle": "dynbicycle_underspecified"}


def _cells(systems, variants, seeds):
    """The fixed E01 scoring matrix: 4 systems x 8 variants, plus made-prior-only for DB only."""
    for system in SYSTEMS:
        if system not in systems:
            continue
        for variant in VARIANTS:
            if variant not in variants:
                continue
            for seed in seeds:
                yield (system, variant, seed)
    if PRIOR_ONLY_SYSTEM in systems and PRIOR_ONLY_VARIANT in variants:
        for seed in seeds:
            yield (PRIOR_ONLY_SYSTEM, PRIOR_ONLY_VARIANT, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default=None, help="Comma-separated systems (default: all)")
    ap.add_argument("--variants", default=None, help="Comma-separated variants (default: all)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--config-dir", default=str(ROOT / "configs/sim"))
    ap.add_argument("--runs-root", default=str(ROOT / "outputs/sim/runs"))
    ap.add_argument("--scores-root", default=str(ROOT / "outputs/sim/scores"))
    ap.add_argument("--data-dir", default=str(ROOT / "data/generated"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from made.utils.config import load_config
    from made.utils import e01_condition
    from scripts.sim import evaluate

    wanted_systems = set(SYSTEMS) if a.systems is None else set(a.systems.split(","))
    wanted_variants = (set(VARIANTS) | {PRIOR_ONLY_VARIANT}) if a.variants is None \
        else set(a.variants.split(","))
    seeds = [int(s) for s in a.seeds.split(",")]

    cells = list(_cells(wanted_systems, wanted_variants, seeds))
    if a.dry_run:
        for c in cells:
            print("  ", c)
        print(len(cells))
        return 0

    config_dir = Path(a.config_dir)
    runs_root = Path(a.runs_root)
    scores_root = Path(a.scores_root)
    data_dir = Path(a.data_dir)

    records, t0 = [], time.time()
    for i, (system, variant, seed) in enumerate(cells, 1):
        cfg_path = config_dir / f"{PREFIX[system]}_{variant}.json"
        cfg = load_config(str(cfg_path))
        cfg = replace(cfg, training=replace(cfg.training, seed=seed),
                      evaluation=replace(cfg.evaluation,
                                         perturbation_seed=cfg.evaluation.perturbation_seed + seed))
        condition = e01_condition(cfg)
        run = runs_root / system / condition / variant / f"seed{seed}"
        checkpoint = None if variant in ("clamp", "made-prior-only") else str(run / "checkpoints")
        out = scores_root / system / condition / variant / f"seed{seed}" / "metrics.json"
        cell_label = f"{system}/{condition}/{variant}/seed{seed}"

        if checkpoint is not None and not Path(checkpoint).is_dir():
            records.append({"cell": cell_label, "out": str(out),
                            "ok": False, "error": "missing checkpoint"})
            continue

        # A row with no learned model borrows the MaDE model for THIS system, condition and
        # seed for its Dyn.-L.
        made_ckpt = runs_root / system / condition / "made" / f"seed{seed}" / "checkpoints"
        dyn_learned_from = str(made_ckpt) if variant in E01_BASELINE_VARIANTS_FOR_DYN_L \
            and made_ckpt.is_dir() else None

        try:
            evaluate.main_programmatic(
                cfg,
                checkpoint=checkpoint,
                variant=variant,
                test_data=str(data_dir / system),
                output=str(out),
                eval_regime=condition,
                dyn_learned_from=dyn_learned_from,
            )
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001 - recorded per cell, the sweep continues
            ok, err = False, f"{type(exc).__name__}: {exc}"
        records.append({"cell": cell_label, "out": str(out), "ok": ok, "error": err})
        if i % 10 == 0 or not ok:
            print(f"  [{i}/{len(cells)}] {cell_label} ok={ok}" + (f"  {err}" if err else ""))

    n_ok = sum(1 for r in records if r.get("ok"))
    manifest_path = scores_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "n_cells": len(records), "n_ok": n_ok,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
        "cells": records,
    }, indent=2) + "\n")
    print(f"\n{n_ok}/{len(records)} cells scored in {(time.time() - t0) / 60:.1f} min")
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
