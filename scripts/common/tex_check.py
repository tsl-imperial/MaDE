"""Compile what you emit.

Three rendered tables in a row failed to drop in, and each was caught on the manuscript side,
which is the expensive place to catch it. The third was well-formed but did not build: its
header declared a span one column too wide and LaTeX stopped with "Extra alignment tab has been
changed to \\cr".

A generator that writes a LaTeX fragment can wrap it in a minimal document and run the engine in
about a second. `assert_compiles` does that and **raises rather than warns**, so a fragment that
does not build is never written.

**The principle is the reproduction gate's**: a check that has never been observed to fail is
not a check. `self_test()` below feeds this one the exact off-by-one that reached the manuscript
and asserts it is rejected, so the checker is known to discriminate rather than assumed to.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


# Macros the MANUSCRIPT defines and a fragment may legitimately use. Without stubs here the
# checker rejects a correct fragment. A
# checker that fails valid input is as bad as one that passes invalid input, so the contract is:
# every macro the manuscript provides is stubbed, and anything else is still an error.
#
# The stubs render something visible rather than nothing, so a proof read of the check output
# shows where the placeholders sit.
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
    """Negative control: the exact defect that reached the manuscript must be rejected."""
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


# --- Fitting the TEXT BLOCK, which `assert_compiles` cannot see ----------------------------
#
# `assert_compiles` builds on a 40cm page precisely so that width never masks a structural
# error. That makes it blind to a table that builds perfectly but is wider than the column
# it has to sit in. A fragment can pass the first check and still be unusable, so the two
# checks are separate and both are run.
#
# ICLR's style sets a 5.5in text block. The width is a parameter rather than a constant so a
# caller can measure against whatever the target document actually uses.
ICLR_TEXTWIDTH = "5.5in"

# Macros the MANUSCRIPT defines and a fragment may legitimately use. Without stubs here the
# checker rejects a correct fragment. A
# checker that fails valid input is as bad as one that passes invalid input, so the contract is:
# every macro the manuscript provides is stubbed, and anything else is still an error.
#
# The stubs render something visible rather than nothing, so a proof read of the check output
# shows where the placeholders sit.
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
