# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Render the appendix inequality breakdown tables (`tab_ineq_breakdown_ind_rate`,
`tab_ineq_breakdown_ind_magnitude`): rate and magnitude, each split into the state-bound and
control-bound components plus the combined figure, from the inD panel.

## How the components combine

The combination rule is verified on the numbers rather than asserted.

**Unit note: the code counts trajectory STEPS, not windows.** `inequality_violation_rate` is
the fraction of trajectory steps that violate at least one constraint, so every "window" below
is a step.

**The rate rule holds.** The combined rate lies between `max(state, control)` and
`state + control` on every cell -- the signature of a union: a step violating both channels is
counted once, so the combined figure falls short of the sum by the overlap.

**The magnitude rule does NOT hold exactly on the reported numbers, and the reason is
averaging.** Quadrature is a per-step identity: for one trajectory step,
`||v_all|| = sqrt(||v_state||^2 + ||v_control||^2)`, because the channels are disjoint
components of the same violation vector. But every number here is a MEAN over steps, and a
mean of norms is not the norm of the means. So the audit records the bracket that is true of
the reported quantities (the combined mean lies between the quadrature of the means and their
sum), rather than the quadrature identity, which holds only before averaging.
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

SOURCE = ROOT / "outputs/ind/panel.json"
OUT_DIR = ROOT / "outputs/table_9_10"

PREDICTORS = ("lstm", "ssm", "transformer")
PREDICTOR_LABEL = {"lstm": "Recurrent", "ssm": "State-space", "transformer": "Transformer"}
ROWS = ("raw", "clamp", "smoother", "smoother_dyn", "made_pnp")
ROW_LABEL = {"raw": "raw", "clamp": "clamp", "smoother": "smoother (acc.)",
             "smoother_dyn": "smoother (res.)", "made_pnp": "MaDE"}
# Rate and magnitude, each split three ways. All six columns in one tabular is wider than the
# text block even at a tight column separation -- eight columns of "x.xxxx +- y.yyyy" do not
# fit. Split into two tables, one per quantity, rather than dropping the standard deviations or
# shrinking the type until it is unreadable.
PANELS = {
    "rate": [("state", "inequality_violation_rate_physical_state"),
             ("control", "inequality_violation_rate_physical_control"),
             ("combined", "inequality_violation_rate_physical")],
    "magnitude": [("state", "inequality_violation_magnitude_physical_state"),
                  ("control", "inequality_violation_magnitude_physical_control"),
                  ("combined", "inequality_violation_magnitude_physical")],
}
COLUMNS = PANELS["rate"] + PANELS["magnitude"]
ALIGNMENT = "llccc"
TABCOLSEP_PT = 3
DECIMALS = 4
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


def main() -> int:
    """Render the table from the source artifact and write the outputs.

    Returns:
        Process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="RELATIVE tolerance for the combination checks. The rates are means of "
                         "indicator variables over thousands of STEPS and carry float error "
                         "of order 1e-8 absolute; 1e-9 rejects one cell on rounding alone.")
    a = ap.parse_args()

    payload = json.loads(Path(a.source).read_text())
    cells = payload["cells"]
    acc: dict = {}
    for c in cells:
        key = (c["family"], c["row"])
        for _, m in COLUMNS:
            v = c["scalar_metrics"].get(m)
            if v is not None:
                acc.setdefault(key, {}).setdefault(m, []).append(float(v))
    grid = {k: {m: {"mean": sum(v) / len(v), "sd": _pop_sd(v), "n": len(v)}
                for m, v in per.items()} for k, per in acc.items()}

    # Combination verification, on the numbers rather than asserted.
    rate_union = mag_bracket = mag_quad_exact = 0
    worst_quad_gap = 0.0
    for c in cells:
        m = c["scalar_metrics"]
        rs, rc, rt = (m["inequality_violation_rate_physical_state"],
                      m["inequality_violation_rate_physical_control"],
                      m["inequality_violation_rate_physical"])
        ms, mc, mt = (m["inequality_violation_magnitude_physical_state"],
                      m["inequality_violation_magnitude_physical_control"],
                      m["inequality_violation_magnitude_physical"])
        tol = a.tol * max(abs(rt), abs(rs) + abs(rc), 1e-12)
        if max(rs, rc) - tol <= rt <= rs + rc + tol:
            rate_union += 1
        q = math.hypot(ms, mc)
        mtol = a.tol * max(abs(mt), abs(ms) + abs(mc), 1e-12)
        if q - mtol <= mt <= ms + mc + mtol:
            mag_bracket += 1
        if abs(mt - q) <= mtol:
            mag_quad_exact += 1
        if mt:
            worst_quad_gap = max(worst_quad_gap, abs(mt - q) / mt)

    n = len(cells)
    combination = {
        "rate": {
            "rule": "UNION: a STEP violating either channel is counted once, so the combined "
                    "rate lies between max(state, control) and state + control, short of the "
                    "sum by the overlap.",
            "cells_satisfying": f"{rate_union}/{n}",
        },
        "magnitude": {
            "rule": "quadrature is a PER-STEP identity (the channels are disjoint components "
                    "of one violation vector). Every number reported here is a MEAN over "
                    "steps, and a mean of norms is not the norm of the means, so quadrature "
                    "does NOT hold exactly of the reported values.",
            "what_does_hold": "the combined mean lies between the quadrature of the means and "
                              "their sum",
            "cells_bracketed": f"{mag_bracket}/{n}",
            "cells_where_quadrature_is_exact": f"{mag_quad_exact}/{n} -- and these are the cells "
                                               "where one component is identically zero, so "
                                               "quadrature and sum coincide trivially",
            "worst_relative_gap_from_quadrature_of_means": worst_quad_gap,
        },
    }

    caption_common = (
        "on the recorded panel, MSE-only predictors and the frozen MaDE models. Means over "
        "trajectory STEPS within a cell, then means with population standard deviation over "
        "cells.")
    captions = {
        "rate": "Inequality RATE breakdown " + caption_common + " The combined column is a "
                "UNION: a trajectory step violating both channels is counted once, so it is at most the "
                "sum of the two components.",
        "magnitude": "Inequality MAGNITUDE breakdown " + caption_common + " The combined column "
                     "is NOT the quadrature of the two component means: quadrature is a "
                     "per-window identity and does not survive averaging. It lies between the "
                     "quadrature of the means and their sum.",
    }

    written, widths, md_all = [], {}, []
    for quantity, cols in PANELS.items():
        cap = captions[quantity]
        md = [f"_{cap}_", "",
              "| Predictor | Row | " + " | ".join(la for la, _ in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        tex = [r"{\setlength{\tabcolsep}{" + str(TABCOLSEP_PT) + r"pt}",
               r"\begin{tabular}{" + ALIGNMENT + "}", r"\toprule",
               "Predictor & Row & " + " & ".join(la for la, _ in cols) + r" \\",
               r"\midrule"]
        for family in PREDICTORS:
            for i, row in enumerate(ROWS):
                md_c, tex_c = [], []
                for _, m in cols:
                    cell = grid.get((family, row), {}).get(m)
                    if cell is None:
                        md_c.append("--")
                        tex_c.append("--")
                        continue
                    # One precision for every cell, no e-notation. Values below 1e-4 therefore
                    # print as 0.0000; their exact values are in the audit JSON.
                    num = f"{cell['mean']:.{DECIMALS}f}"
                    sd = f"{cell['sd']:.{DECIMALS}f}"
                    md_c.append(f"{num} ± {sd}")
                    tex_c.append(f"${num}\\,{{\\scriptscriptstyle\\pm}}\\,{sd}$")
                md.append(f"| {PREDICTOR_LABEL[family] if i == 0 else ''} | {ROW_LABEL[row]} | "
                          + " | ".join(md_c) + " |")
                first = (r"\multirow{" + str(len(ROWS)) + r"}{*}{" + PREDICTOR_LABEL[family]
                         + "}") if i == 0 else ""
                tex.append(f"{first} & {ROW_LABEL[row]} & " + " & ".join(tex_c) + r" \\")
            if family != PREDICTORS[-1]:
                tex.append(r"\midrule")
        tex += [r"\bottomrule", r"\end{tabular}", "}"]
        tex_text = "\n".join(tex) + "\n"
        assert_compiles(tex_text, name=f"tab_ineq_breakdown_ind_{quantity}.tex")
        over = overfull_pt(tex_text, textwidth=f"{TEXTWIDTH_PT}pt")
        if over > 0.0:
            raise SystemExit(f"{quantity} table is {over:.2f}pt too wide for {TEXTWIDTH_PT}pt")
        widths[quantity] = over
        written.append((quantity, tex_text, "\n".join(md) + "\n"))
        md_all += md + [""]

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for quantity, tex_text, md_text in written:
        (out_dir / f"tab_ineq_breakdown_ind_{quantity}.tex").write_text(tex_text)
        (out_dir / f"tab_ineq_breakdown_ind_{quantity}.md").write_text(md_text)
    (out_dir / "tab_ineq_breakdown_ind.audit.json").write_text(json.dumps({
        "artifact": _display(out_dir / "tab_ineq_breakdown_ind.audit.json"),
        "what": "The inD inequality breakdown, state and control components with the combined "
                "figure. Two tables, one per quantity.",
        "why_two_tables": "all six columns in one tabular is wider than the text block even at "
                          "a tight column separation. Splitting keeps the standard deviations "
                          "and the type readable.",
        "source": _display(Path(a.source)),
        "aggregation": "mean over windows per cell, then mean with population sd over cells; "
                       "means only",
        "HOW_THE_COMPONENTS_COMBINE": combination,
        "width_pt_target": TEXTWIDTH_PT,
        "overfull_pt": widths,
        "cells": {f"{k[0]}/{k[1]}": v for k, v in grid.items()},
    }, indent=2) + "\n")

    print("\n".join(md_all))
    print(f"\ncombination check: rate union {rate_union}/{n}, magnitude bracketed "
          f"{mag_bracket}/{n}, quadrature exact only {mag_quad_exact}/{n} "
          f"(worst gap {worst_quad_gap:.1%})")
    print(f"widths: {widths} (target {TEXTWIDTH_PT}pt)")
    print(f"wrote 3 files under {_display(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
