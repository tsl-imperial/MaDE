"""Verify a LaTeX fragment builds before it is written.

`assert_compiles` wraps a fragment in a minimal document and runs the LaTeX engine, raising
rather than warning, so a fragment that does not build is never written. `self_test()` checks
the checker itself against a known-bad fragment (an off-by-one column span) to confirm it can
reject, not just pass.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


# Macros the manuscript defines that a fragment may legitimately use; without stubs the checker
# rejects valid input. Stubs render visibly so check output shows where placeholders sit.
MANUSCRIPT_MACROS = r"""
\newcommand{\resultpending}[1]{[pending: #1]}
\newcommand{\basispending}[1]{[pending basis: #1]}
"""

PREAMBLE = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[table]{xcolor}
\usepackage[margin=1cm,paperwidth=40cm,paperheight=40cm]{geometry}
""" + MANUSCRIPT_MACROS + r"""
\begin{document}
"""
POSTAMBLE = "\n\\end{document}\n"


class TexDoesNotCompile(RuntimeError):
    """Raised when a fragment fails to build. The engine's own error is in the message."""


def _engine() -> str:
    for name in ("pdflatex", "tectonic"):
        if shutil.which(name):
            return name
    raise TexDoesNotCompile(
        "no LaTeX engine found (looked for pdflatex, tectonic); refusing to write a fragment "
        "that cannot be checked. An unverifiable fragment is what this guard exists to prevent."
    )


def assert_compiles(fragment: str, *, name: str = "fragment") -> None:
    """Build *fragment* inside a minimal document. Raise if the engine fails."""
    engine = _engine()
    with tempfile.TemporaryDirectory(prefix="texcheck_") as tmp:
        tex = Path(tmp) / "check.tex"
        tex.write_text(PREAMBLE + fragment + POSTAMBLE)
        proc = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
            detail = "\n  ".join(lines[:5]) or proc.stdout[-1500:]
            raise TexDoesNotCompile(
                f"{name} does not compile; refusing to write it.\n  {detail}"
            )


def self_test() -> bool:
    """Negative control: a fragment with a known off-by-one column span must be rejected."""
    good = (
        r"\begin{tabular}{llccccc}" "\n" r"\toprule" "\n"
        r" & & \multicolumn{4}{c}{Ablations} & \multicolumn{1}{c}{Reference} \\" "\n"
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-7}" "\n"
        r"Cond. & Metric & A & B & C & D & E \\" "\n" r"\midrule" "\n"
        r"DI & Ineq. & 1 & 2 & 3 & 4 & 5 \\" "\n"
        r"\bottomrule" "\n" r"\end{tabular}"
    )
    bad = good.replace(r"\multicolumn{4}{c}{Ablations}", r"\multicolumn{5}{c}{Ablations}")
    bad = bad.replace(r"\cmidrule(lr){3-6}\cmidrule(lr){7-7}",
                      r"\cmidrule(lr){3-7}\cmidrule(lr){8-8}")
    try:
        assert_compiles(good, name="well-formed control")
    except TexDoesNotCompile as exc:
        print(f"SELF-TEST FAILED: the well-formed control did not compile -- {exc}")
        return False
    try:
        assert_compiles(bad, name="the off-by-one that reached the manuscript")
    except TexDoesNotCompile:
        print("self-test PASS: well-formed compiles, the manuscript's off-by-one is rejected")
        return True
    print("SELF-TEST FAILED: the off-by-one COMPILED, so this checker cannot discriminate")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)


# assert_compiles builds on a 40cm page so width never masks a structural error, which makes it
# blind to a table that compiles but is wider than the column it has to sit in. overfull_pt /
# assert_fits_width check width separately.
#
# ICLR's style sets a 5.5in text block; width is a parameter so a caller can target other docs.
ICLR_TEXTWIDTH = "5.5in"

MANUSCRIPT_MACROS = r"""
\newcommand{\resultpending}[1]{[pending: #1]}
\newcommand{\basispending}[1]{[pending basis: #1]}
"""

WIDTH_PREAMBLE = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[table]{xcolor}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=%s,textheight=9in]{geometry}
""" + MANUSCRIPT_MACROS + r"""
\begin{document}
\noindent
"""


def overfull_pt(fragment: str, *, textwidth: str = ICLR_TEXTWIDTH) -> float:
    """How many points the fragment overflows `textwidth` by. 0.0 when it fits.

    Parses the engine's own "Overfull \\hbox (Xpt too wide)" rather than estimating from
    character counts, which is what makes this a measurement instead of a guess.
    """
    import re
    engine = _engine()
    with tempfile.TemporaryDirectory(prefix="texwidth_") as tmp:
        tex = Path(tmp) / "width.tex"
        tex.write_text((WIDTH_PREAMBLE % textwidth) + fragment + POSTAMBLE)
        proc = subprocess.run(
            [engine, "-interaction=nonstopmode", tex.name],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        log = (Path(tmp) / "width.log")
        text = log.read_text(errors="replace") if log.exists() else proc.stdout
        worst = 0.0
        for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", text):
            worst = max(worst, float(m.group(1)))
        return worst


def assert_fits_width(fragment: str, *, name: str = "fragment",
                      textwidth: str = ICLR_TEXTWIDTH) -> float:
    """Raise if the fragment overflows `textwidth`. Returns the measured overflow (0.0)."""
    over = overfull_pt(fragment, textwidth=textwidth)
    if over > 0.0:
        raise TexDoesNotCompile(
            f"{name}: overfull hbox, {over:.2f}pt too wide for a {textwidth} text block. "
            f"Fit it rather than shipping a table the manuscript cannot use."
        )
    return over
