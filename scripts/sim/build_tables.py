"""Aggregate E01 per-seed metrics into LaTeX results tables.

Walks the scoring tree for per-seed ``metrics.json`` files, computes per-cell mean
and population std across seeds, and emits inner ``tabular`` bodies for the
main headline table and the appendix ablation table. Both LaTeX blocks and the
underlying ``mean``/``std``/``n`` numbers are written so every cell is
auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common.tex_check import assert_compiles  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "sim" / "scores"

# Row order in the table.
ROWS: list[tuple[str, str, str]] = [
    # (label, system_dir, condition_dir)
    ("DI", "double_integrator", "fully-specified"),
    ("UNI", "unicycle", "fully-specified"),
    ("KB", "kinematic_bicycle", "fully-specified"),
    ("DB", "dynamic_bicycle", "underspecified"),
]

# Column order in the main-paper headline table.
HEADLINE_COLS: list[tuple[str, str]] = [
    ("Clamp", "clamp"),
    ("MLP", "mlp"),
    ("FAB", "fab"),
    ("MaDE", "made"),
]

# Column order in the appendix ablation table. Full MaDE is retained as a
# reference column so ablation deltas are readable without jumping back to the
# main table.
ABLATION_COLS: list[tuple[str, str]] = [
    (r"w/o $F_a$", "made-no-residual"),
    (r"w/o $\mathcal{C}$", "made-no-corrector"),
    (r"fixed-$\mathcal{I}$", "made-fixed-i"),
    (r"sup.-$\mathcal{I}$", "made-supervised-i"),
    ("MaDE", "made"),
]

# Variants recorded in the seed-level file but printed in neither table. The manuscript quotes
# the prior-only model's dynamic-bicycle numbers directly, and a quoted number has to be
# recomputable from an artifact rather than from a report. These never enter `stats_grid`, so no
# printed cell and no emphasis can move because of them.
SEED_LEVEL_ONLY_VARIANTS: tuple[str, ...] = ("made-prior-only",)

ALL_COLS: list[tuple[str, str]] = []
_seen_variants: set[str] = set()
for _label, _variant in HEADLINE_COLS + ABLATION_COLS:
    if _variant not in _seen_variants:
        ALL_COLS.append((_label, _variant))
        _seen_variants.add(_variant)
del _label, _variant, _seen_variants

METRICS: list[tuple[str, str]] = [
    ("Ineq. rate", "inequality_violation_rate"),
    ("Ineq. mag.", "inequality_violation_magnitude"),
    ("Dyn.-K", "dynamics_violation_known"),
    ("Dyn.-L", "dynamics_violation_learned"),
    ("Dyn.-T", "dynamics_violation_true"),
    ("Fid.", "fidelity"),
]
_METRIC_LABEL: dict[str, str] = {key: label for label, key in METRICS}

# The headline table prints only four metric rows for DI/UNI/KB (dynamics is fully known there,
# so the learned- and true-dynamics rows are redundant); DB keeps all six because its dynamics
# are partly learned. The ablation table and both audit JSONs always keep all six.
FOUR: list[tuple[str, str]] = [METRICS[0], METRICS[1], METRICS[2], METRICS[5]]
HEADLINE_METRICS_BY_ROW: dict[str, list[tuple[str, str]]] = {
    "DI": FOUR,
    "UNI": FOUR,
    "KB": FOUR,
    "DB": METRICS,
}

DISPLAY_DECIMALS = 4

# Dagger rule: a cell carries a dagger when at least one seed's value for that (row, variant,
# metric) exceeds ABS_FLOOR AND exceeds RATIO times the median of the OTHER seeds. This is a
# seed-level flag: one wild seed inside an otherwise sane cell, which a mean and a standard
# deviation together can hide.
DAGGER = r"$^\dagger$"
PENDING_MARK = r"\basispending"
ABS_FLOOR = 1.0
RATIO = 3.0


def _flagged(values: list[float]) -> bool:
    """True when some seed exceeds ABS_FLOOR and RATIO x the median of the OTHER seeds."""
    if len(values) < 2:
        return False
    for i, v in enumerate(values):
        if v <= ABS_FLOOR:
            continue
        others = values[:i] + values[i + 1:]
        if not others:
            continue
        med = median(others)
        if med <= 0 or v > RATIO * med:
            return True
    return False


def _seed_values(seed_level: dict, row_label: str, variant: str, metric_key: str) -> list[float]:
    """Per-seed values for one (row, variant, metric), in sorted seed-dir order."""
    per_variant = (seed_level.get(row_label) or {}).get(variant) or {}
    values: list[float] = []
    for seed_name in sorted(per_variant):
        v = per_variant[seed_name].get(metric_key)
        if isinstance(v, (int, float)):
            values.append(float(v))
    return values


def _dagger_flags(
    seed_level: dict, cols: list[tuple[str, str]]
) -> set[tuple[str, str, str]]:
    """(row_label, variant, metric_key) triples whose per-seed values satisfy `_flagged`.

    A cell rendered as a pending placeholder (no numeric seed values) is never flagged; this
    function only ever sees numeric series, so that exclusion falls out of `_seed_values`
    returning an empty list for such cells.
    """
    flagged: set[tuple[str, str, str]] = set()
    for row_label, _, _ in ROWS:
        for _, variant in cols:
            for _, metric_key in METRICS:
                values = _seed_values(seed_level, row_label, variant, metric_key)
                if values and _flagged(values):
                    flagged.add((row_label, variant, metric_key))
    return flagged


@dataclass
class CellStats:
    mean: float
    std: float
    n: int

    @classmethod
    def from_values(cls, values: list[float]) -> "CellStats | None":
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return None
        m = sum(finite) / len(finite)
        var = sum((v - m) ** 2 for v in finite) / len(finite)
        return cls(mean=m, std=math.sqrt(var), n=len(finite))


def _load_seeds(variant_dir: Path, filename: str) -> dict[str, list[float]]:
    bucket: dict[str, list[float]] = {key: [] for _, key in METRICS}
    if not variant_dir.is_dir():
        return bucket
    for seed_dir in sorted(variant_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        path = seed_dir / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        metrics = payload.get("metrics", {})
        for _, key in METRICS:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                bucket[key].append(float(value))
    return bucket


def _gather_cell(
    system: str, condition: str, variant: str
) -> dict[str, CellStats | None]:
    variant_dir = OUT_ROOT / system / condition / variant
    seeds = _load_seeds(variant_dir, "metrics.json")
    return {key: CellStats.from_values(values) for key, values in seeds.items()}


def _gather_cell_seed_level(system: str, condition: str, variant: str) -> dict[str, dict]:
    """Per-seed metric values, keyed by seed directory name.

    Separate from :func:`_gather_cell` on purpose: that one collapses seeds into mean and
    population std, and several claims rest on an ordering holding in EACH of five seeds,
    which cannot be recovered from a mean.
    """
    import json as _json

    variant_dir = OUT_ROOT / system / condition / variant
    out: dict[str, dict] = {}
    if not variant_dir.is_dir():
        return out
    for seed_dir in sorted(variant_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        path = seed_dir / "metrics.json"
        if not path.exists():
            continue
        payload = _json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        out[seed_dir.name] = {
            key: float(metrics[key])
            for _, key in METRICS
            if isinstance(metrics.get(key), (int, float))
        }
    return out


# A cell is flagged "unstable" when its mean alone exceeds a per-metric
# physical-plausibility threshold — i.e., the model has blown up in absolute
# terms, independent of seed-to-seed scatter. Thresholds are picked so that
# baselines that legitimately ignore dynamics or feasibility (Clamp at
# Dyn.-K~0.2, MLP at Dyn.-K~0.2) stay uncoloured, while catastrophic numerical
# divergence (Dyn.-T~1e5, Ineq.\,mag~500, Fid~400) is highlighted.
_UNSTABLE_MEAN_THRESHOLDS: dict[str, float] = {
    "inequality_violation_rate": 0.6,
    "inequality_violation_magnitude": 5.0,
    "dynamics_violation_known": 10.0,
    "dynamics_violation_learned": 10.0,
    "dynamics_violation_true": 10.0,
    "fidelity": 30.0,
}


def _is_unstable(stats: "CellStats | None", metric_key: str) -> bool:
    if stats is None:
        return False
    if not math.isfinite(stats.mean):
        return True
    threshold = _UNSTABLE_MEAN_THRESHOLDS.get(metric_key)
    if threshold is None:
        return False
    return abs(stats.mean) > threshold


def _row_ranking(
    row_stats: list["CellStats | None"], metric_key: str
) -> tuple[set[int], set[int], set[int]]:
    """Identify (best_idxs, second_idxs, unstable_idxs) for one metric row.

    Best/second-best are computed over cells that are present and not flagged
    as unstable, since unstable means are unreliable for ranking. Lower is
    better for every metric in this table.
    """
    unstable_idxs = {
        i for i, s in enumerate(row_stats) if _is_unstable(s, metric_key)
    }
    candidates = [
        (i, s.mean)
        for i, s in enumerate(row_stats)
        if s is not None and i not in unstable_idxs and math.isfinite(s.mean)
    ]
    if not candidates:
        return set(), set(), unstable_idxs
    means = sorted({m for _, m in candidates})
    best_value = means[0]
    best_idxs = {i for i, m in candidates if m == best_value}
    if len(means) < 2:
        return best_idxs, set(), unstable_idxs
    second_value = means[1]
    second_idxs = {i for i, m in candidates if m == second_value}
    return best_idxs, second_idxs, unstable_idxs


def _format_scalar(value: float) -> str:
    """Format a single mean or std in LaTeX math-mode-ready text."""
    if not math.isfinite(value):
        return r"\mathrm{nan}"
    rounded = round(value, DISPLAY_DECIMALS)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{DISPLAY_DECIMALS}f}"


def _format_cell(
    stats: CellStats | None,
    *,
    rank: str | None = None,
    unstable: bool = False,
    dagger: bool = False,
) -> str:
    """Render one cell with mean and std on the same line."""
    if stats is None:
        return "--"
    body = (
        "$"
        + _format_scalar(stats.mean)
        + r"\,{\scriptscriptstyle\pm}\,"
        + _format_scalar(stats.std)
        + "$"
    )
    if rank == "best":
        # Strip the outer $ delimiters and wrap content in \mathbf{}.
        inner = body[1:-1]
        body = r"$\mathbf{" + inner + r"}$"
    elif rank == "second":
        body = r"\underline{" + body + r"}"
    # The E01 tables carry no red text: the `unstable` flag affects ranking only (see
    # `_row_ranking`, which excludes an unstable mean from best/second) and emits no markup here.
    del unstable  # retained in the signature: callers pass it, and the audit JSON records it
    # A pending placeholder cell is never daggered, and a cell is daggered at most once; the
    # dagger is appended last, after any bold/underline markup.
    if dagger and PENDING_MARK not in body and DAGGER not in body:
        body = body + DAGGER
    return body


def _build_table(
    stats_grid: dict,
    cols: list[tuple[str, str]],
    *,
    table_kind: str,
    seed_level: dict | None = None,
) -> str:
    lines: list[str] = []
    # The column spans are DERIVED from the column list, not written by hand, so the count
    # cannot go wrong when a column is added or removed.
    group_labels = {"headline": ("Baselines", "MaDE"), "ablation": ("Ablations", "Reference")}
    if table_kind not in group_labels:
        raise ValueError(f"Unknown table_kind: {table_kind}")
    left_label, right_label = group_labels[table_kind]
    n_left = len(cols) - 1          # every column but the last is in the left group
    first = 3                        # two label columns precede the data columns
    last_left = first + n_left - 1
    right_col = last_left + 1
    header_groups = (
        f" & & \\multicolumn{{{n_left}}}{{c}}{{{left_label}}} "
        f"& \\multicolumn{{1}}{{c}}{{{right_label}}} \\\\"
    )
    cmidrules = (
        rf"\cmidrule(lr){{{first}-{last_left}}}\cmidrule(lr){{{right_col}-{right_col}}}"
    )
    header_cols = (
        "Cond. & Metric & "
        + " & ".join(label for label, _ in cols)
        + r" \\"
    )
    lines.append(r"\begin{tabular}{ll" + ("c" * len(cols)) + "}")
    lines.append(r"\toprule")
    lines.append(header_groups)
    lines.append(cmidrules)
    lines.append(header_cols)
    lines.append(r"\midrule")
    seed_level = seed_level or {}
    for row_idx, (row_label, system, condition) in enumerate(ROWS):
        row_metrics = HEADLINE_METRICS_BY_ROW[row_label] if table_kind == "headline" else METRICS
        for metric_idx, (metric_label, metric_key) in enumerate(row_metrics):
            row_stats = [
                stats_grid[(row_label, variant)][metric_key] for _, variant in cols
            ]
            best_idxs, second_idxs, unstable_idxs = _row_ranking(row_stats, metric_key)
            row_cells: list[str] = []
            for col_idx, (col_label, variant) in enumerate(cols):
                stats = row_stats[col_idx]
                rank = (
                    "best"
                    if col_idx in best_idxs
                    else ("second" if col_idx in second_idxs else None)
                )
                values = _seed_values(seed_level, row_label, variant, metric_key)
                dagger = bool(values) and _flagged(values)
                row_cells.append(
                    _format_cell(
                        stats, rank=rank, unstable=col_idx in unstable_idxs, dagger=dagger
                    )
                )
            cond_col = row_label if metric_idx == 0 else ""
            row_str = f"        {cond_col} & {metric_label} & " + " & ".join(row_cells) + r" \\"
            lines.append(row_str)
        if row_idx != len(ROWS) - 1:
            lines.append(r"        \\[-0.4ex]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _dyn_learned_recovery(system: str, condition: str, variant: str) -> dict | None:
    """Which Dyn.-L control recovery this cell used, read off its metrics.json.

    A row with its own learned model has no entry: its Dyn.-L is its own emitted control through
    its own model. A row with none carries the borrowed-model record `scripts/sim/evaluate.py`
    writes, saying whether the learned inverse resolved or the known-model fallback applied.
    """
    paths, seeds = set(), 0
    for seed_dir in sorted((OUT_ROOT / system / condition / variant).glob("seed*")):
        f = seed_dir / "metrics.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text()).get("dyn_learned_recovery")
        if rec:
            paths.add(rec["path"])
            seeds += 1
    if not seeds:
        return None
    return {"path": sorted(paths)[0] if len(paths) == 1 else sorted(paths),
            "seeds": seeds,
            "uniform_across_seeds": len(paths) == 1}


def _build_audit(
    stats_grid: dict, cols: list[tuple[str, str]], seed_level: dict | None = None
) -> dict:
    seed_level = seed_level or {}
    dagger_flags = _dagger_flags(seed_level, cols)
    audit: dict = {}
    for row_label, system, condition in ROWS:
        audit[row_label] = {
            "system": system,
            "condition_dir": condition,
            "metrics_file": "metrics.json",
            # Which recovery each cell's Dyn.-L used. Absent for a variant that has its own
            # learned model, because it borrows nothing.
            "dyn_learned_recovery": {
                variant: rec for _, variant in cols
                if (rec := _dyn_learned_recovery(system, condition, variant)) is not None
            },
            "cells": {},
        }
        # Pre-compute rank/unstable flags for each (row, metric) so the audit
        # mirrors the LaTeX highlighting decisions.
        rank_by_metric: dict[str, dict[int, str]] = {}
        unstable_by_metric: dict[str, set[int]] = {}
        for _, metric_key in METRICS:
            row_stats = [
                stats_grid[(row_label, variant)][metric_key] for _, variant in cols
            ]
            best_idxs, second_idxs, unstable_idxs = _row_ranking(row_stats, metric_key)
            rank_by_metric[metric_key] = {
                **{i: "best" for i in best_idxs},
                **{i: "second" for i in second_idxs},
            }
            unstable_by_metric[metric_key] = unstable_idxs
        for col_idx, (col_label, variant) in enumerate(cols):
            cell = {}
            for _, metric_key in METRICS:
                stats = stats_grid[(row_label, variant)][metric_key]
                cell[metric_key] = (
                    None
                    if stats is None
                    else {
                        "mean": stats.mean,
                        "std": stats.std,
                        "n": stats.n,
                        "rank": rank_by_metric[metric_key].get(col_idx),
                        "unstable": col_idx in unstable_by_metric[metric_key],
                        "dagger": (row_label, variant, metric_key) in dagger_flags,
                    }
                )
            audit[row_label]["cells"][variant] = {"label": col_label, "metrics": cell}
    audit["flagging_rules"] = {
        "dagger": (
            f"a cell is flagged when at least one seed's value exceeds {ABS_FLOOR} AND "
            f"exceeds {RATIO}x the median of the other seeds for that (row, variant, metric)"
        ),
        "cells_flagged": [
            f"{row}/{variant}/{_METRIC_LABEL[metric_key]}"
            for (row, variant, metric_key) in sorted(dagger_flags)
        ],
    }
    return audit


def _prior_only_comparison(seed_level: dict) -> dict:
    """DB `made-prior-only` vs `made`, mean/std/n per metric, from the seed-level series.

    This is the artifact a quoted prior-only appendix number is recomputed from.
    """
    db = seed_level.get("DB", {})
    out: dict[str, dict] = {}
    for _, metric_key in METRICS:
        entry: dict[str, dict] = {}
        for variant in ("made-prior-only", "made"):
            values = _seed_values(seed_level, "DB", variant, metric_key)
            stats = CellStats.from_values(values)
            if stats is not None:
                entry[variant] = {"mean": stats.mean, "std_population": stats.std, "n": stats.n}
        if entry:
            out[metric_key] = entry
    del db
    return out


def _checked_write(path, label, text):
    """Compile a fragment before writing it. Refuses rather than warns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    assert_compiles(text, name=label)
    Path(path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "outputs" / "sim" / "scores",
        help=(
            "Directory to read per-seed metrics.json from (the scoring pass output). Point it "
            "at a different tree to regenerate the tables from a different set of scored cells."
        ),
    )
    parser.add_argument(
        "--out-seed-level",
        type=Path,
        default=ROOT / "outputs" / "tables" / "tab_e01_seed_level.json",
        help=(
            "Where to write PER-SEED values alongside the aggregates. Always written, because "
            "claims that rest on an ordering holding in each of five seeds cannot be checked "
            "against a mean."
        ),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "outputs" / "tables" / "tab_e01_results.tex",
        help="Where to write the regenerated tabular block (LaTeX).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs" / "tables" / "tab_e01_results.audit.json",
        help="Where to write per-cell mean/std/n audit JSON.",
    )
    parser.add_argument(
        "--out-ablation-tex",
        type=Path,
        default=ROOT / "outputs" / "tables" / "tab_e01_ablations.tex",
        help="Where to write the appendix ablation tabular block (LaTeX).",
    )
    parser.add_argument(
        "--out-ablation-json",
        type=Path,
        default=ROOT / "outputs" / "tables" / "tab_e01_ablations.audit.json",
        help="Where to write appendix ablation per-cell mean/std/n audit JSON.",
    )
    args = parser.parse_args()

    # Rebind the module-level root so every loader below reads the requested tree. Done here
    # rather than by threading a parameter through five call sites.
    global OUT_ROOT
    OUT_ROOT = args.results_root

    stats_grid: dict = {}
    seed_level: dict = {}
    for row_label, system, condition in ROWS:
        for _, variant in ALL_COLS:
            stats_grid[(row_label, variant)] = _gather_cell(system, condition, variant)
            per_seed = _gather_cell_seed_level(system, condition, variant)
            if per_seed:
                seed_level.setdefault(row_label, {})[variant] = per_seed
        # Seed-level only: deliberately NOT added to `stats_grid`, so no printed cell in
        # either table depends on these variants.
        for variant in SEED_LEVEL_ONLY_VARIANTS:
            per_seed = _gather_cell_seed_level(system, condition, variant)
            if per_seed:
                seed_level.setdefault(row_label, {})[variant] = per_seed

    table_tex = _build_table(
        stats_grid, HEADLINE_COLS, table_kind="headline", seed_level=seed_level
    )
    _checked_write(args.out_tex, "headline .tex", table_tex + "\n")

    audit = _build_audit(stats_grid, HEADLINE_COLS, seed_level=seed_level)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    ablation_tex = _build_table(
        stats_grid, ABLATION_COLS, table_kind="ablation", seed_level=seed_level
    )
    _checked_write(args.out_ablation_tex, "ablation .tex", ablation_tex + "\n")

    ablation_audit = _build_audit(stats_grid, ABLATION_COLS, seed_level=seed_level)
    args.out_ablation_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_ablation_json.write_text(
        json.dumps(ablation_audit, indent=2, sort_keys=True) + "\n"
    )

    prior_only_comparison = _prior_only_comparison(seed_level)
    for metric_key, entry in prior_only_comparison.items():
        po, mo = entry.get("made-prior-only"), entry.get("made")
        if po and mo:
            print(
                f"  prior-only comparison  {_METRIC_LABEL[metric_key]:11s} "
                f"prior-only {po['mean']:.4f} +/- {po['std_population']:.4f}  "
                f"made {mo['mean']:.4f} +/- {mo['std_population']:.4f}"
            )

    args.out_seed_level.parent.mkdir(parents=True, exist_ok=True)
    args.out_seed_level.write_text(json.dumps({
        "artifact": "E01 PER-SEED metric values, one entry per (row, variant, seed)",
        "why": ("A seed-level table sits alongside every aggregate, because claims resting on "
                "an ordering holding in each of five seeds cannot be checked against a mean."),
        "results_root": str(OUT_ROOT),
        "metrics": [key for _, key in METRICS],
        "seed_level_only_variants": list(SEED_LEVEL_ONLY_VARIANTS),
        "seed_level_only_note": ("These appear here but in neither printed table, so a quoted "
                                 "number can be recomputed from this artifact. They never enter "
                                 "the table grid, so no printed cell or emphasis depends on "
                                 "them."),
        "prior_only_comparison": prior_only_comparison,
        "rows": seed_level,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote seed-level JSON: {args.out_seed_level}")

    print(f"Wrote LaTeX tabular: {args.out_tex}")
    print(f"Wrote audit JSON:    {args.out_json}")
    print(f"Wrote ablation LaTeX tabular: {args.out_ablation_tex}")
    print(f"Wrote ablation audit JSON:    {args.out_ablation_json}")


if __name__ == "__main__":
    main()
