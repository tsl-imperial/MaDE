# MaDE: Markovian Dynamics Enforcer.
#
# Copyright (c) 2026 Kevin Yu, Transport Systems and Logistics Laboratory, Imperial College London
# SPDX-License-Identifier: MIT
#
# Part of the code release for:
#   K. Yu, T. Guo, C. Antoniou, P. Angeloudis. "Markovian Dynamics Enforcer: Feasibility
#   Preserving Correction on Learned Dynamics Manifolds." NeurIPS, 2026. arXiv:2609.39888
# If you use this code, please cite the paper (see CITATION.cff and README.md).

"""Render the main inD results table (`tab_e05_ind`) from the filtered evaluation panel
produced by `scripts/ind/evaluate.py`.

The table prints, for every predictor family and metric, the mean over windows within a cell
and then the mean with population standard deviation over cells. Per-cell medians (p50) are
not rendered in the table but are kept in the accompanying JSON audit file, since the prose
sometimes quotes them.
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
OUT_DIR = ROOT / "outputs/tables"

PREDICTORS = ("lstm", "ssm", "transformer")
ROWS = ("raw", "clamp", "smoother", "smoother_dyn", "made_pnp")
PREDICTOR_LABEL = {"lstm": "Recurrent", "ssm": "State-space", "transformer": "Transformer"}
# Public row labels: "smoother-dyn" reads as the dynamic-bicycle model, which the smoother is
# never given, so the two smoother arms are labelled by what they are tuned against instead.
ROW_LABEL = {"raw": "raw", "clamp": "clamp", "smoother": "smoother (acc.)",
             "smoother_dyn": "smoother (res.)", "made_pnp": "MaDE"}

TABCOLSEP_PT = 3

PENDING_ROWS: tuple[str, ...] = ()
PENDING_TEX = r"\resultpending{}"
PENDING_MD = "[pending]"
# Table layout: rows are (predictor family x metric); the five arms are the columns, under a
# `Baselines` spanner over the four baselines and a `MaDE` spanner over ours.
ALIGNMENT = "llccccc"
BASELINE_COLS = ("raw", "clamp", "smoother", "smoother_dyn")
MADE_COL = "made_pnp"
COLUMN_ORDER = BASELINE_COLS + (MADE_COL,)
METRICS = [
    ("ADE (m)", "ade"),
    ("FDE (m)", "fde"),
    ("Dyn.-K", "dynamics_violation"),
    ("Ineq. rate", "inequality_violation_rate_physical"),
]
DIST_METRICS = ("ade", "fde")


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


def _population_std(values: list[float]) -> float:
    """Population standard deviation (divides by n).

    Args:
        values: Non-empty sequence of values.

    Returns:
        The population standard deviation.
    """
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _agg(values: list[float]) -> dict | None:
    """Summarise values as mean, population std and count.

    Args:
        values: Per-cell values.

    Returns:
        Dict with ``mean``, ``std`` and ``n``, or None when ``values`` is empty.
    """
    if not values:
        return None
    return {"mean": sum(values) / len(values), "std": _population_std(values), "n": len(values)}


def _grid(cells: list[dict], rows: tuple[str, ...]) -> tuple[dict, dict]:
    """Aggregate cells per (family, row) for the metrics and the distance medians.

    Args:
        cells: Evaluation cells from the source artifact.
        rows: Table rows to include.

    Returns:
        The metric aggregates and the per-cell p50 aggregates, both keyed by (family, row).
    """
    acc: dict = {(f, r): {k: [] for _, k in METRICS} for f in PREDICTORS for r in rows}
    med: dict = {(f, r): {k: [] for k in DIST_METRICS} for f in PREDICTORS for r in rows}
    for cell in cells:
        key = (cell["family"], cell["row"])
        if key not in acc:
            continue
        for _, metric in METRICS:
            v = cell["scalar_metrics"].get(metric, cell.get(metric))
            if v is not None:
                acc[key][metric].append(float(v))
        for metric in DIST_METRICS:
            blk = cell.get(metric)
            if isinstance(blk, dict) and blk.get("p50") is not None:
                med[key][metric].append(float(blk["p50"]))
    out = {k: {m: _agg(v) for m, v in per.items()} for k, per in acc.items()}
    medians = {k: {m: _agg(v) for m, v in per.items()} for k, per in med.items()}
    return out, medians


def _emphasis(grid: dict, rows: tuple[str, ...]) -> dict:
    """Emphasis runs along each metric row, across the arms, within one predictor family: the
    best cell in a (family, metric) row is bold, the second-best is underlined.

    Args:
        grid: Metric aggregates keyed by (family, row).
        rows: Table rows compared against each other.

    Returns:
        Mapping metric -> (family, row) -> ``best``, ``second`` or None.
    """
    out: dict = {}
    for f in PREDICTORS:
        for _, metric in METRICS:
            vals = [((f, r), grid[(f, r)][metric]["mean"]) for r in rows
                    if grid.get((f, r), {}).get(metric)]
            if not vals:
                continue
            means = sorted({v for _, v in vals})
            best = means[0]
            second = means[1] if len(means) > 1 else None
            for k, v in vals:
                out.setdefault(metric, {})[k] = (
                    "best" if v == best
                    else ("second" if second is not None and v == second else None))
    return out


def _fmt(cell: dict | None, rank: str | None, tex: bool) -> str:
    """Format one table cell as mean plus-minus std, with optional emphasis.

    Args:
        cell: Aggregate record, or None for a missing cell.
        rank: ``best``, ``second`` or None.
        tex: Emit LaTeX when True, markdown otherwise.

    Returns:
        The formatted cell string (``--`` when missing).
    """
    if cell is None:
        return "--"
    if tex:
        body = f"{cell['mean']:.4f}\\,{{\\scriptscriptstyle\\pm}}\\,{cell['std']:.4f}"
        if rank == "best":
            return r"$\mathbf{" + body + r"}$"
        if rank == "second":
            return r"\underline{$" + body + r"$}"
        return f"${body}$"
    body = f"{cell['mean']:.4f} ± {cell['std']:.4f}"
    if rank == "best":
        return f"**{body}**"
    if rank == "second":
        return f"_{body}_"
    return body


TEXTWIDTH_PT_LITERAL = 458.74


def main() -> None:
    """Render the table from the source artifact and write the outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--textwidth-pt", type=float, default=TEXTWIDTH_PT_LITERAL)
    a = ap.parse_args()

    payload = json.loads(Path(a.source).read_text())
    cells = payload["cells"]
    grid, medians = _grid(cells, ROWS)
    final_rows = tuple(r for r in ROWS if r not in PENDING_ROWS)
    emph = _emphasis(grid, final_rows)

    n_by_row = {r: sum(1 for c in cells if c["row"] == r) for r in ROWS}
    made_seeds = sorted({c["made_seed"] for c in cells if c["made_seed"] is not None})
    n_pred_seeds = len({c["predictor_seed"] for c in cells})
    decomposition = (
        f"{len(cells)} cells = {n_by_row['made_pnp']} MaDE ({len(PREDICTORS)} predictor "
        f"families x {n_pred_seeds} predictor seeds x {len(made_seeds)} MaDE seeds) + "
        f"{n_by_row['raw']} raw + {n_by_row['clamp']} clamp + {n_by_row['smoother']} smoother + "
        f"{n_by_row['smoother_dyn']} smoother-dyn")
    caption = (
        f"Filtered protocol, MSE-only predictors. {decomposition}. The MaDE rows average "
        f"{len(made_seeds)} MaDE models, not {n_pred_seeds}; deviations on those rows "
        f"are population standard deviations over {n_by_row['made_pnp']} cells spanning "
        f"{len(made_seeds)} MaDE seeds. Emphasis runs along each metric row, across the "
        f"five arms within a predictor family: best in bold, second underlined.")

    md = [f"_{caption}_", "",
          "| Predictor | Metric | " + " | ".join(ROW_LABEL[c] for c in COLUMN_ORDER) + " |",
          "|---|---|" + "---|" * len(COLUMN_ORDER)]
    nb = len(BASELINE_COLS)
    tex = [r"{\setlength{\tabcolsep}{" + str(TABCOLSEP_PT) + r"pt}",
           r"\begin{tabular}{" + ALIGNMENT + "}", r"\toprule",
           r" & & \multicolumn{" + str(nb) + r"}{c}{Baselines} & \multicolumn{1}{c}{MaDE} \\",
           r"\cmidrule(lr){3-" + str(2 + nb) + r"}\cmidrule(lr){" + str(3 + nb) + "-"
           + str(3 + nb) + "}",
           "Predictor & Metric & " + " & ".join(ROW_LABEL[c] for c in COLUMN_ORDER) + r" \\",
           r"\midrule"]
    for fi, family in enumerate(PREDICTORS):
        for mi, (mlabel, metric) in enumerate(METRICS):
            md_c, tex_c = [], []
            for arm in COLUMN_ORDER:
                if arm in PENDING_ROWS:
                    md_c.append(PENDING_MD)
                    tex_c.append(PENDING_TEX)
                    continue
                rank = emph.get(metric, {}).get((family, arm))
                md_c.append(_fmt(grid[(family, arm)][metric], rank, False))
                tex_c.append(_fmt(grid[(family, arm)][metric], rank, True))
            head = PREDICTOR_LABEL[family] if mi == 0 else ""
            md.append(f"| {head} | {mlabel} | " + " | ".join(md_c) + " |")
            tex.append(f"{head} & {mlabel} & " + " & ".join(tex_c) + r" \\")
        if fi != len(PREDICTORS) - 1:
            tex.append(r"\\[-0.4ex]")
    tex += [r"\bottomrule", r"\end{tabular}", "}"]
    # The table may be scaled to fit, with the natural width and the factor reported: five arm
    # columns beside two label columns is wide, so adjustbox scales only if it must.
    body = tex
    tex = ([r"\begin{adjustbox}{max width=" + str(TEXTWIDTH_PT_LITERAL) + r"pt}"]
           + body + [r"\end{adjustbox}"])

    quoted = {}
    for f in PREDICTORS:
        raw_ade = grid[(f, "raw")]["ade"]["mean"]
        made_ade = grid[(f, "made_pnp")]["ade"]["mean"]
        quoted[f] = {
            "raw_dyn_k": grid[(f, "raw")]["dynamics_violation"]["mean"],
            "made_dyn_k": grid[(f, "made_pnp")]["dynamics_violation"]["mean"],
            "dyn_k_ratio_raw_over_made": (grid[(f, "raw")]["dynamics_violation"]["mean"]
                                          / grid[(f, "made_pnp")]["dynamics_violation"]["mean"]),
            "raw_ade": raw_ade, "made_ade": made_ade,
            "made_over_raw_ade_ratio": made_ade / raw_ade,
        }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_e05_ind.md").write_text("\n".join(md) + "\n")
    tex_text = "\n".join(tex) + "\n"
    assert_compiles(tex_text, name="tab_e05_ind.tex")
    # Compile what is emitted at the target width. assert_compiles builds on a wide page by
    # design, so it cannot see an overfull box; this is the separate check.
    unwrapped = "\n".join(line for line in tex
                          if not line.startswith(r"\begin{adjustbox}")
                          and not line.startswith(r"\end{adjustbox}")) + "\n"
    natural_over = overfull_pt(unwrapped, textwidth=f"{a.textwidth_pt}pt")
    natural_pt_width = a.textwidth_pt + natural_over
    scale_factor = min(1.0, a.textwidth_pt / natural_pt_width)
    over_target = overfull_pt(tex_text, textwidth=f"{a.textwidth_pt}pt")
    if over_target > 0.0:
        raise SystemExit(
            f"tab_e05_ind.tex is {over_target:.2f}pt too wide for a {a.textwidth_pt}pt text "
            f"block.")
    (out_dir / "tab_e05_ind.tex").write_text(tex_text)
    (out_dir / "tab_e05_ind.audit.json").write_text(json.dumps({
        "artifact": _display(out_dir / "tab_e05_ind.audit.json"),
        "what": "The main inD results table, rendered on the MSE-only predictors with the "
                "smoother rows. Means only in the table; medians here.",
        "source": _display(Path(a.source)),
        "predictor_root": payload.get("predictor_root"),
        "columns": [la for la, _ in METRICS],
        "aggregation": "per cell, the mean over windows; then the mean with POPULATION standard "
                       "deviation over cells. No median is rendered.",
        "medians": {f"{k[0]}/{k[1]}": v for k, v in medians.items()},
        "medians_note": "mean over cells of each cell's p50 over windows, for ADE and FDE",
        "n_cells": len(cells),
        "n_by_row": n_by_row,
        "decomposition": decomposition,
        "made_seeds": made_seeds,
        "emphasis": "best per row bold, second underlined, computed over all five rows",
        "width": {
            "target_textwidth_pt": a.textwidth_pt,
            "overfull_at_target_pt": over_target,
            "natural_width_pt": natural_pt_width,
            "natural_overfull_pt": natural_over,
            "scale_factor_applied": scale_factor,
            "mechanism": "adjustbox max width; the table is scaled only if it would "
                         "otherwise overflow",
        },
        "public_labels": ROW_LABEL,
        "quoted_figures_per_family": quoted,
        "cells": {f"{k[0]}/{k[1]}": v for k, v in grid.items()},
    }, indent=2) + "\n")

    print("\n".join(md))
    print()
    print("figures the abstract and results paragraph quote:")
    for f, q in quoted.items():
        print(f"  {PREDICTOR_LABEL[f]:12s} raw Dyn.-K {q['raw_dyn_k']:.4f}  MaDE Dyn.-K "
              f"{q['made_dyn_k']:.6f}  ratio {q['dyn_k_ratio_raw_over_made']:6.1f}x  "
              f"MaDE/raw ADE {q['made_over_raw_ade_ratio']:.3f}")
    print(f"\nwidth: natural {natural_pt_width:.2f}pt ({natural_over:+.2f}pt vs target), "
          f"adjustbox scale {scale_factor:.4f}")
    print(f"wrote 3 files under {_display(out_dir)}")


if __name__ == "__main__":
    main()
