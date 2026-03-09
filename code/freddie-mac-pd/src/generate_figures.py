"""
generate_figures.py
===================
Generate all publication-ready figures for the Freddie Mac mortgage PD model.
Delegates to ``shared/figures.py`` for the standard comparison suite, then
adds dataset-specific EDA overview.

Usage
-----
    python generate_figures.py
    python generate_figures.py --small
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "..", "shared"))
sys.path.insert(0, _SHARED)
sys.path.insert(0, _HERE)

from figures import generate_all_figures

SMALL = "--small" in sys.argv
ROOT = _HERE
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "small") if SMALL else os.path.join(RESULTS_DIR, "tables")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions", "small") if SMALL else os.path.join(RESULTS_DIR, "predictions")
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "small") if SMALL else os.path.join(RESULTS_DIR, "figures")


def main() -> None:
    generate_all_figures(TABLES_DIR, PREDICTIONS_DIR, FIG_DIR, small=SMALL)


if __name__ == "__main__":
    main()
