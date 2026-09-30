"""Contract tests for E01 results table generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sim.build_tables import (  # noqa: E402
    ABLATION_COLS,
    DAGGER,
    FOUR,
    HEADLINE_COLS,
    HEADLINE_METRICS_BY_ROW,
    METRICS,
    ROWS,
    CellStats,
    _flagged,
    _format_cell,
    _format_scalar,
)


def test_e01_results_table_rows_are_headline_conditions_only():
    assert [label for label, _, _ in ROWS] == ["DI", "UNI", "KB", "DB"]


def test_e01_results_table_uses_native_metrics_rows():
    assert ROWS == [
        ("DI", "double_integrator", "fully-specified"),
        ("UNI", "unicycle", "fully-specified"),
        ("KB", "kinematic_bicycle", "fully-specified"),
        ("DB", "dynamic_bicycle", "underspecified"),
    ]


def test_e01_headline_table_omits_ablation_columns():
    assert HEADLINE_COLS == [
        ("Clamp", "clamp"),
        ("MLP", "mlp"),
        ("FAB", "fab"),
        ("MaDE", "made"),
    ]


def test_e01_ablation_table_keeps_made_reference():
    assert ABLATION_COLS == [
        (r"w/o $F_a$", "made-no-residual"),
        (r"w/o $\mathcal{C}$", "made-no-corrector"),
        (r"fixed-$\mathcal{I}$", "made-fixed-i"),
        (r"sup.-$\mathcal{I}$", "made-supervised-i"),
        ("MaDE", "made"),
    ]


def test_e01_results_table_formats_scalars_to_four_decimal_places():
    assert _format_scalar(1.23456) == "1.2346"
    assert _format_scalar(0.0) == "0.0000"
    assert _format_scalar(1.0e-17) == "0.0000"
    assert _format_scalar(12345.67891) == "12345.6789"


def test_e01_results_table_formats_mean_and_std_to_four_decimal_places():
    assert (
        _format_cell(CellStats(mean=1.23456, std=0.0, n=1))
        == r"$1.2346\,{\scriptscriptstyle\pm}\,0.0000$"
    )
    assert (
        _format_cell(CellStats(mean=1.23454, std=0.98765, n=3))
        == r"$1.2345\,{\scriptscriptstyle\pm}\,0.9877$"
    )


def test_e01_headline_table_keeps_only_four_metric_rows_for_di_uni_kb():
    assert FOUR == [METRICS[0], METRICS[1], METRICS[2], METRICS[5]]
    for row_label in ("DI", "UNI", "KB"):
        assert HEADLINE_METRICS_BY_ROW[row_label] == FOUR


def test_e01_headline_table_keeps_all_six_metric_rows_for_db():
    assert HEADLINE_METRICS_BY_ROW["DB"] == METRICS


def test_e01_dagger_rule_flags_one_wild_seed_but_not_a_uniform_series():
    assert _flagged([0.1, 0.1, 0.1, 0.1, 5.0]) is True
    assert _flagged([0.1] * 5) is False


def test_e01_dagger_rule_renders_at_the_end_of_the_cell():
    stats = CellStats(mean=1.06, std=2.15, n=5)
    cell = _format_cell(stats, dagger=True)
    assert cell.endswith(DAGGER)
    undaggered = _format_cell(stats, dagger=False)
    assert not undaggered.endswith(DAGGER)
