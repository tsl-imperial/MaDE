# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Render the completion-only appendix table (`tab_completion_only`).

## What the table is

- **Rows:** raw, completion-only, full MaDE, per predictor family. Public labels, no harness keys.
- **Columns:** Ineq. rate and Ineq. mag. on the **filtered** evaluation windows, and Ineq. mag. on
  **all test** windows.
- **Means with population sd over cells, means only.**
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common.tex_check import assert_compiles, overfull_pt  # noqa: E402

SOURCE = ROOT / "outputs/ind/completion_only.json"
OUT_DIR = ROOT / "outputs/table_11"

PREDICTORS = ("lstm", "ssm", "transformer")
PREDICTOR_LABEL = {"lstm": "Recurrent", "ssm": "State-space", "transformer": "Transformer"}
ROWS = ("raw", "made_completion_only", "made_full")
ROW_LABEL = {"raw": "raw", "made_completion_only": "completion only", "made_full": "MaDE"}

# (column label, window set, metric key)
COLUMNS = [
    (r"Ineq.\ rate (filt.)", "filtered", "inequality_violation_rate_physical"),
    (r"Ineq.\ mag.\ (filt.)", "filtered", "inequality_violation_magnitude_physical"),
    (r"Ineq.\ mag.\ (all)", "unfiltered", "inequality_violation_magnitude_physical"),
]
ALIGNMENT = "llccc"
TABCOLSEP_PT = 4
TEXTWIDTH_PT = 458.74


def _display(p: Path) -> str:
    """Format a path relative to the repository root when possible.

    Args:
        p: Path to format.

    Returns:
        The root-relative path as a string, or the path unchanged when outside the root.
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _pop_sd(v: list[float]) -> float:
    """Population standard deviation (divides by n).

    Args:
        v: Non-empty sequence of values.

    Returns:
        The population standard deviation.
    """
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _primary(cell: dict, metric: str) -> float | None:
    """The figure for this row: emitted controls where the row emits them.

    Two artifact vintages exist:

    * older -- bare key is the RECOVERED scoring, emitted sits under `__emitted_controls`;
    * current -- bare key is the EMITTED scoring, recovered sits under `__recovered_controls`.

    Taking `__emitted_controls` where it exists and the bare key otherwise is correct for either.

    Args:
        cell: One evaluation cell from the source artifact.
        metric: Scalar metric name.

    Returns:
        The metric value, or None when the cell lacks it.
    """
    sm = cell.get("scalar_metrics", {})
    if cell["row"].startswith("made"):
        v = sm.get(metric + "__emitted_controls")
        if v is not None:
            return float(v)
    v = sm.get(metric)
    return None if v is None else float(v)


def _grid(cells: list[dict]) -> dict:
    """Aggregate cells into mean, population sd and count per (family, row, window set).

    Args:
        cells: Evaluation cells from the source artifact.

    Returns:
        Mapping from (family, row, window set) to per-metric mean/sd/n.
    """
    acc: dict = {}
    for c in cells:
        gkey = (c["family"], c["row"], c["window_set"])
        for _, _ws, metric in COLUMNS:
            # A row that EMITS controls is scored on its own emitted controls; see _primary.
            v = _primary(c, metric)
            if v is not None:
                acc.setdefault(gkey, {}).setdefault(metric, []).append(float(v))
    out = {}
    for key, per in acc.items():
        out[key] = {m: {"mean": sum(v) / len(v), "sd": _pop_sd(v), "n": len(v)}
                    for m, v in per.items()}
    return out


def _fmt(cell: dict | None, tex: bool) -> str:
    """Format one table cell as mean plus-minus sd.

    Args:
        cell: Mean/sd/n record, or None for a missing cell.
        tex: Emit LaTeX math when True, plain text otherwise.

    Returns:
        The formatted cell string (``--`` when missing).
    """
    if cell is None:
        return "--"
    body = f"{cell['mean']:.4f}\\,{{\\scriptscriptstyle\\pm}}\\,{cell['sd']:.4f}"
    return f"${body}$" if tex else f"{cell['mean']:.4f} ± {cell['sd']:.4f}"


def main() -> int:
    """Render the table from the source artifact and write the outputs.

    Returns:
        Process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE),
                    help="x3-attribution artifact on the MSE-only predictors.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    payload = json.loads(Path(a.source).read_text())
    grid = _grid(payload["cells"])
    pooled: dict = {}
    for label, ws, metric in COLUMNS:
        for row in ROWS:
            rv = [v for v in (_primary(c, metric) for c in payload["cells"]
                              if c["window_set"] == ws and c["row"] == row)
                  if v is not None]
            if rv:
                pooled[f"{row}/{ws}/{metric}"] = {"mean": sum(rv) / len(rv), "n": len(rv)}
    # completion-only against ITS OWN raw forecast, per cell, filtered windows
    raw_by = {(c["family"], c["predictor_seed"]): c for c in payload["cells"]
              if c["row"] == "raw" and c["window_set"] == "filtered"}
    wins: dict = {}
    for metric in ("inequality_violation_rate_physical",
                   "inequality_violation_magnitude_physical"):
        better = total = 0
        for c in payload["cells"]:
            if c["row"] != "made_completion_only" or c["window_set"] != "filtered":
                continue
            r = raw_by.get((c["family"], c["predictor_seed"]))
            cv = _primary(c, metric)
            rvv = _primary(r, metric) if r is not None else None
            if cv is None or rvv is None:
                continue
            total += 1
            # completion-only on ITS OWN controls against raw on RECOVERED ones, applying the
            # emitted-vs-recovered rule to each row rather than one rule for both.
            if cv < rvv:
                better += 1
        wins[metric] = {"completion_only_better_than_its_own_raw": better, "of": total}

    caption = (
        "Completion-only attribution on the MSE-only recorded predictors. Rows are the raw "
        "forecast, MaDE with the corrector skipped (completion only), and the full model, "
        "scored on the same cells in one pass. Means with population standard deviations over "
        "cells. Filtered columns use the evaluation windows; the last column uses all test "
        "windows. Inequality scoring uses emitted controls where the row emits them, "
        "known-model-recovered otherwise.")

    md = [f"_{caption}_", "",
          "| Predictor | Row | " + " | ".join(la.replace(r"\ ", " ") for la, _, _ in COLUMNS)
          + " |",
          "|---|---|" + "---|" * len(COLUMNS)]
    tex = [r"{\setlength{\tabcolsep}{" + str(TABCOLSEP_PT) + r"pt}",
           r"\begin{tabular}{" + ALIGNMENT + "}", r"\toprule",
           r" & & \multicolumn{2}{c}{Filtered windows} & \multicolumn{1}{c}{All test windows} \\",
           r"\cmidrule(lr){3-4} \cmidrule(lr){5-5}",
           "Predictor & Row & " + " & ".join(la for la, _, _ in COLUMNS) + r" \\",
           r"\midrule"]
    for family in PREDICTORS:
        for i, row in enumerate(ROWS):
            md_c, tex_c = [], []
            for _, ws, metric in COLUMNS:
                cell = grid.get((family, row, ws), {}).get(metric)
                md_c.append(_fmt(cell, False))
                tex_c.append(_fmt(cell, True))
            md.append(f"| {PREDICTOR_LABEL[family] if i == 0 else ''} | {ROW_LABEL[row]} | "
                      + " | ".join(md_c) + " |")
            first = (r"\multirow{" + str(len(ROWS)) + r"}{*}{" + PREDICTOR_LABEL[family] + "}"
                     ) if i == 0 else ""
            tex.append(f"{first} & {ROW_LABEL[row]} & " + " & ".join(tex_c) + r" \\")
        if family != PREDICTORS[-1]:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", "}"]

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_text = "\n".join(tex) + "\n"
    assert_compiles(tex_text, name="tab_completion_only.tex")
    over = overfull_pt(tex_text, textwidth=f"{TEXTWIDTH_PT}pt")
    if over > 0.0:
        raise SystemExit(f"tab_completion_only.tex is {over:.2f}pt too wide for "
                         f"{TEXTWIDTH_PT}pt.")
    (out_dir / "tab_completion_only.tex").write_text(tex_text)
    (out_dir / "tab_completion_only.md").write_text("\n".join(md) + "\n")
    (out_dir / "tab_completion_only.audit.json").write_text(json.dumps({
        "artifact": _display(out_dir / "tab_completion_only.audit.json"),
        "what": "The completion-only appendix table.",
        "source": _display(Path(a.source)),
        "run": (f"evaluate_completion_only.py on {payload.get('predictor_root')} against "
                f"{payload.get('made_root')}."),
        "rows": {k: ROW_LABEL[k] for k in ROWS},
        "columns": [{"label": la.replace("\\ ", " "), "window_set": ws, "metric": m}
                    for la, ws, m in COLUMNS],
        "aggregation": "mean over windows per cell, then mean with POPULATION sd over cells; "
                       "means only",
        "expected_cell_counts_per_family": {"raw": 5, "made_completion_only": 15, "made_full": 15},
        "cell_count_note": "raw has no MaDE-seed axis (5 predictor seeds); the two MaDE rows "
                           "span 5 predictor seeds x 3 MaDE seeds",
        "inequality_convention": "a row that emits controls is scored on its own "
                                 "(completion-only and full MaDE); raw is scored on controls "
                                 "recovered through the inverse known model",
        "width_pt_target": TEXTWIDTH_PT,
        "overfull_pt": over,
        "pooled_means": pooled,
        "completion_only_vs_its_own_raw_filtered": wins,
        "cells": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in grid.items()},
    }, indent=2) + "\n")

    print("\n".join(md))
    print(f"width: {over:.2f}pt overfull at {TEXTWIDTH_PT}pt")
    for m, w in wins.items():
        print(f"  {m}: completion-only beats its own raw in "
              f"{w['completion_only_better_than_its_own_raw']}/{w['of']} cells")
    print(f"wrote 3 files under {_display(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
