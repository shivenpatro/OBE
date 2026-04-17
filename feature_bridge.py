"""
feature_bridge.py
=================
Translates the raw xAPI-Educational Mining Dataset columns produced by
`data_loader.py` into the two crisp scalar inputs consumed by the Mamdani
Fuzzy Inference System in `fuzzy_engine.py`.

Why a separate bridge module?
------------------------------
The xAPI dataset tracks *engagement behaviours* (raised hands, resource visits,
discussion posts) rather than explicit numeric scores.  The fuzzy engine expects
two domain inputs:

    assignment_score  ∈ [0, 100]   — academic performance proxy
    attendance        ∈ [0, 100]   — class-participation proxy

This module defines the evidence-based formulae that map the raw xAPI
columns onto those two dimensions, keeping the transformation logic
isolated and testable independently of both the loader and the engine.

─────────────────────────────────────────────────────────────────────
Derivation of assignment_score
─────────────────────────────────────────────────────────────────────
Three xAPI columns each quantify a distinct component of academic effort:

  raised_hands       (0–100)  — active classroom participation; strongest
                                 individual predictor of internal-assessment
                                 performance in Amrieh et al. (2016).  Weight 0.40

  announcements_view (0–100)  — information-seeking / self-directed study;
                                 mirrors mid-term preparation behaviour.  Weight 0.35

  discussion         (0–100)  — collaborative engagement; reflects applied
                                 understanding and peer learning.  Weight 0.25

    assignment_score = 0.40·raised_hands
                     + 0.35·announcements_view
                     + 0.25·discussion

Weights were chosen to reflect the relative predictive importance reported
in the original dataset paper and clipped to [0, 100] as a safety guard.

─────────────────────────────────────────────────────────────────────
Derivation of attendance
─────────────────────────────────────────────────────────────────────
The absence_days column is binary after encoding in data_loader.py:
    0  →  Under-7 absences   (low absenteeism)
    1  →  Above-7 absences   (high absenteeism)

Using only that binary sentinel would produce just two distinct attendance
values across all 480–650 records, which collapses the fuzzy inference onto
two discrete branches.  To recover a continuous distribution, we blend
`absence_days` with `visited_resources` (a continuous 0–100 engagement
signal that correlates with presence and active learning):

    Under-7 group (absence_days == 0):
        base = 70
        attendance = base + 0.30 · visited_resources
        range: [70, 100]  — student is generally present; resource access
                            fine-tunes placement within the High MF region.

    Above-7 group (absence_days == 1):
        base = 10
        attendance = base + 0.40 · visited_resources
        range: [10, 50]  — student misses many classes; resource access
                           can partially compensate but cannot fully
                           offset physical absence.

Result is clipped to [0, 100].  The split ranges mean the two groups
never overlap, preserving the semantic meaning of the binary absence label
while giving each group its own continuous variation for richer fuzzy firing.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column names expected from data_loader.py
# ---------------------------------------------------------------------------
_REQUIRED_COLS = [
    "raised_hands",
    "announcements_view",
    "discussion",
    "visited_resources",
    "absence_days",
]

# ---------------------------------------------------------------------------
# Weights for assignment_score composite
# ---------------------------------------------------------------------------
_W_RAISED_HANDS        = 0.40
_W_ANNOUNCEMENTS_VIEW  = 0.35
_W_DISCUSSION          = 0.25


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class FuzzyInputs(NamedTuple):
    """
    Crisp scalar inputs ready to pass directly into FuzzyAssessmentEngine.assess().

    Attributes
    ----------
    assignment_score : float
        Composite academic engagement score in [0, 100].
    attendance : float
        Derived attendance percentage in [0, 100].
    """
    assignment_score: float
    attendance: float


# ---------------------------------------------------------------------------
# Core derivation functions
# ---------------------------------------------------------------------------

def derive_assignment_score(
    raised_hands: float,
    announcements_view: float,
    discussion: float,
) -> float:
    """
    Compute the weighted composite assignment_score from the three
    xAPI engagement columns.

    Parameters
    ----------
    raised_hands, announcements_view, discussion : float
        Raw xAPI values in [0, 100].

    Returns
    -------
    float
        Crisp assignment_score in [0, 100].
    """
    score = (
        _W_RAISED_HANDS       * raised_hands
        + _W_ANNOUNCEMENTS_VIEW * announcements_view
        + _W_DISCUSSION         * discussion
    )
    return float(np.clip(score, 0.0, 100.0))


def derive_attendance(
    absence_days: int,
    visited_resources: float,
) -> float:
    """
    Derive a continuous attendance percentage from the binary absence_days
    flag and the continuous visited_resources signal.

    Parameters
    ----------
    absence_days : int
        0 = Under-7 absences, 1 = Above-7 absences (encoded by data_loader).
    visited_resources : float
        xAPI resource-access count in [0, 100].

    Returns
    -------
    float
        Derived attendance in [0, 100].
    """
    if int(absence_days) == 0:
        # Low absenteeism: base 70, fine-tuned upward by resource access
        attendance = 70.0 + 0.30 * visited_resources
    else:
        # High absenteeism: base 10, partially lifted by resource access
        attendance = 10.0 + 0.40 * visited_resources

    return float(np.clip(attendance, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Row-level bridge (single student)
# ---------------------------------------------------------------------------

def row_to_fuzzy_inputs(row: pd.Series) -> FuzzyInputs:
    """
    Convert a single DataFrame row (from `data_loader.load_dataset`) into
    the two crisp FIS inputs.

    Parameters
    ----------
    row : pd.Series
        One row of the working set produced by data_loader.load_dataset().

    Returns
    -------
    FuzzyInputs
        Named tuple with `assignment_score` and `attendance`.

    Raises
    ------
    KeyError
        If any of the required xAPI columns are absent from the row.
    """
    missing = [c for c in _REQUIRED_COLS if c not in row.index]
    if missing:
        raise KeyError(
            f"Row is missing required xAPI columns: {missing}. "
            f"Ensure the DataFrame was produced by data_loader.load_dataset()."
        )

    assignment_score = derive_assignment_score(
        raised_hands=float(row["raised_hands"]),
        announcements_view=float(row["announcements_view"]),
        discussion=float(row["discussion"]),
    )
    attendance = derive_attendance(
        absence_days=int(row["absence_days"]),
        visited_resources=float(row["visited_resources"]),
    )
    return FuzzyInputs(assignment_score=assignment_score, attendance=attendance)


# ---------------------------------------------------------------------------
# DataFrame-level bridge (full working set)
# ---------------------------------------------------------------------------

def dataframe_to_fuzzy_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised derivation of fuzzy inputs for the entire working set.

    Adds two new columns to a copy of the DataFrame:
        `assignment_score`  — weighted engagement composite
        `attendance`        — blended absence / resource-access score

    Parameters
    ----------
    df : pd.DataFrame
        Working set from data_loader.load_dataset().

    Returns
    -------
    pd.DataFrame
        Original columns plus `assignment_score` and `attendance`.
        The original xAPI columns are preserved for traceability.
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required xAPI columns: {missing}. "
            f"Ensure it was produced by data_loader.load_dataset()."
        )

    out = df.copy()

    # Vectorised assignment_score
    out["assignment_score"] = np.clip(
        _W_RAISED_HANDS       * out["raised_hands"]
        + _W_ANNOUNCEMENTS_VIEW * out["announcements_view"]
        + _W_DISCUSSION         * out["discussion"],
        0.0,
        100.0,
    )

    # Vectorised attendance (conditional on absence group)
    low_absence  = out["absence_days"] == 0
    high_absence = ~low_absence

    out["attendance"] = 0.0
    out.loc[low_absence,  "attendance"] = np.clip(
        70.0 + 0.30 * out.loc[low_absence,  "visited_resources"], 0.0, 100.0
    )
    out.loc[high_absence, "attendance"] = np.clip(
        10.0 + 0.40 * out.loc[high_absence, "visited_resources"], 0.0, 100.0
    )

    return out


# ---------------------------------------------------------------------------
# Validation / diagnostics
# ---------------------------------------------------------------------------

def validate_bridge_output(df_bridged: pd.DataFrame) -> None:
    """
    Print a statistical summary of the derived fuzzy inputs to verify
    the bridge produces sensible distributions before running inference.

    Parameters
    ----------
    df_bridged : pd.DataFrame
        Output of dataframe_to_fuzzy_inputs().
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print("  Feature Bridge — Derived Fuzzy Input Distributions")
    print(sep)

    for col in ("assignment_score", "attendance"):
        if col not in df_bridged.columns:
            print(f"  [WARN] Column '{col}' not found — bridge may not have run.")
            continue
        s = df_bridged[col]
        print(
            f"\n  {col}:\n"
            f"    min={s.min():.2f}  max={s.max():.2f}  "
            f"mean={s.mean():.2f}  std={s.std():.2f}"
        )
        # Histogram bins of 10 to give a quick shape check
        bins   = np.arange(0, 110, 10)
        labels = [f"{int(b)}–{int(b+10)}" for b in bins[:-1]]
        counts, _ = np.histogram(s, bins=bins)
        for lbl, cnt in zip(labels, counts):
            bar = "█" * (cnt // max(1, max(counts) // 30))
            print(f"    [{lbl:>7}]  {cnt:>4}  {bar}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point — run as smoke test with the actual working set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_loader import load_dataset, DATA_FILE

    try:
        print("[bridge] Loading working set …")
        df_working = load_dataset()

        print("[bridge] Deriving fuzzy inputs …")
        df_bridged = dataframe_to_fuzzy_inputs(df_working)

        validate_bridge_output(df_bridged)

        # Show 5 sample rows with both raw inputs and derived fuzzy values
        sample_cols = [
            "raised_hands", "announcements_view", "discussion",
            "visited_resources", "absence_days",
            "assignment_score", "attendance",
            "performance_class",
        ]
        available = [c for c in sample_cols if c in df_bridged.columns]
        print("  [sample] First 5 rows:\n")
        print(df_bridged[available].head().to_string(index=False))
        print()

        # Persist bridged dataset for downstream pipeline use
        out_path = DATA_FILE.parent / "bridged_set.csv"
        df_bridged.to_csv(out_path, index=False)
        print(f"[bridge] Bridged dataset saved to: {out_path}")

    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print("        Run `python download_dataset.py` first.", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
