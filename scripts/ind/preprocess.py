"""Thin CLI shim for the inD preprocessing pipeline.

Delegates entirely to made.data.ind.preprocess._cli().
Equivalent to: python -m made.data.ind.preprocess <args>
"""

from made.data.ind.preprocess import _cli

if __name__ == "__main__":
    _cli()
