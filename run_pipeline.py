"""
run_pipeline.py
===============
End-to-end Phase 1 → Phase 2 demonstration pipeline.

Execution order
---------------
  Step 1  data_loader      — Load, clean, validate, and filter the xAPI CSV
  Step 2  feature_bridge   — Derive fuzzy antecedents (assignment_score, attendance)
  Step 3  FuzzyAssessmentEngine.batch_assess — Score every row via Mamdani FIS
  Step 4  Analytics        — Print class-wise and label-wise statistics
  Step 5  Persistence      — Save scored_set.csv and membership_functions.png

Usage
-----
  python run_pipeline.py
  python run_pipeline.py --no-plot      # skip saving the MF chart
  python run_pipeline.py --verbose      # show fuzzification detail for 3 samples
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

from data_loader   import load_dataset, DATA_DIR
from feature_bridge import dataframe_to_fuzzy_inputs, validate_bridge_output
from fuzzy_engine  import FuzzyAssessmentEngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORED_CSV  = DATA_DIR / "scored_set.csv"
MF_PLOT     = pathlib.Path(__file__).parent / "membership_functions.png"

# ---------------------------------------------------------------------------
# Step 4 helpers — analytics
# ---------------------------------------------------------------------------

def _print_label_distribution(df: pd.DataFrame) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  Attainment Label Distribution  (all students)")
    print(sep)
    order = ["Poor", "Developing", "Satisfactory", "Good", "Excellent"]
    total = len(df)
    for label in order:
        count = (df["attainment_label"] == label).sum()
        pct   = 100 * count / total
        bar   = "█" * int(pct / 2)
        print(f"  {label:<14} {count:>4}  ({pct:5.1f}%)  {bar}")
    print(sep)


def _print_class_breakdown(df: pd.DataFrame) -> None:
    """Cross-tabulate xAPI performance class (H/M/L) vs FIS attainment label."""
    if "performance_class" not in df.columns:
        return

    sep = "─" * 60
    print(f"\n{sep}")
    print("  FIS Attainment vs xAPI Performance Class")
    print(sep)

    label_order = ["Poor", "Developing", "Satisfactory", "Good", "Excellent"]
    class_order = ["H", "M", "L"]

    ct = pd.crosstab(
        df["performance_class"],
        df["attainment_label"],
        margins=True,
    )
    # Reorder columns to logical progression
    present_labels = [l for l in label_order if l in ct.columns]
    if "All" in ct.columns:
        present_labels.append("All")
    ct = ct.reindex(columns=present_labels, fill_value=0)

    present_classes = [c for c in class_order if c in ct.index]
    if "All" in ct.index:
        present_classes.append("All")
    ct = ct.reindex(present_classes, fill_value=0)

    print(f"\n{ct.to_string()}\n")

    # Per-class mean crisp attainment
    print("  Mean crisp attainment by xAPI class:")
    for cls in ["H", "M", "L"]:
        subset = df[df["performance_class"] == cls]["crisp_attainment"]
        if len(subset):
            print(
                f"    {cls} ({len(subset):>3} students):  "
                f"mean={subset.mean():.2f}%  "
                f"std={subset.std():.2f}  "
                f"min={subset.min():.2f}  max={subset.max():.2f}"
            )
    print(sep)


def _print_overall_stats(df: pd.DataFrame) -> None:
    sep = "─" * 60
    col = df["crisp_attainment"]
    print(f"\n{sep}")
    print("  Overall Crisp Attainment Statistics")
    print(sep)
    print(f"  Records    : {len(df)}")
    print(f"  Mean       : {col.mean():.2f} %")
    print(f"  Std dev    : {col.std():.2f}")
    print(f"  Median     : {col.median():.2f} %")
    print(f"  Min / Max  : {col.min():.2f} % / {col.max():.2f} %")
    print(sep)


def _show_sample_verbose(df: pd.DataFrame, engine: FuzzyAssessmentEngine) -> None:
    """Run verbose inference on 3 representative rows from H, M, L classes."""
    print("\n" + "═" * 60)
    print("  Verbose inference — 3 representative students")
    print("═" * 60)
    for cls in ["H", "M", "L"]:
        subset = df[df["performance_class"] == cls] if "performance_class" in df.columns else df
        if subset.empty:
            continue
        # Pick the row closest to the class mean crisp score for representativeness
        mean_score = subset["crisp_attainment"].mean()
        row = subset.iloc[(subset["crisp_attainment"] - mean_score).abs().argsort()[:1]]
        print(f"\n  [xAPI class = {cls}  |  nearest-to-mean student]")
        engine.assess(
            assignment_score=float(row["assignment_score"].iloc[0]),
            attendance=float(row["attendance"].iloc[0]),
            verbose=True,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(save_plot: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Execute the full Phase 1 → Phase 2 pipeline.

    Parameters
    ----------
    save_plot : bool
        If True, renders and saves the membership function chart.
    verbose : bool
        If True, prints fuzzification detail for 3 representative students.

    Returns
    -------
    pd.DataFrame
        The fully scored dataset with ``crisp_attainment`` and
        ``attainment_label`` columns.
    """
    t_start = time.perf_counter()

    # ── Step 1: Load dataset ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Phase 1  →  Loading dataset")
    print("═" * 60)
    df_working = load_dataset()

    # ── Step 2: Derive fuzzy inputs ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Phase 1  →  Deriving fuzzy antecedents (feature bridge)")
    print("═" * 60)
    df_bridged = dataframe_to_fuzzy_inputs(df_working)
    validate_bridge_output(df_bridged)

    # ── Step 3: Batch FIS scoring ────────────────────────────────────────────
    print("═" * 60)
    print("  Phase 2  →  Mamdani FIS batch inference")
    print("═" * 60)
    engine = FuzzyAssessmentEngine()
    df_scored = engine.batch_assess(df_bridged)

    t_infer = time.perf_counter() - t_start
    print(f"  [timing]  Inference complete in {t_infer:.2f} s  "
          f"({len(df_scored)} rows)\n")

    # ── Step 4: Analytics ───────────────────────────────────────────────────
    _print_overall_stats(df_scored)
    _print_label_distribution(df_scored)
    _print_class_breakdown(df_scored)

    if verbose:
        _show_sample_verbose(df_scored, engine)

    # ── Step 5: Persist results ─────────────────────────────────────────────
    df_scored.to_csv(SCORED_CSV, index=False)
    print(f"\n[save]   Scored dataset → {SCORED_CSV}")

    if save_plot:
        # Highlight the student whose score is closest to the global mean
        mean_val = df_scored["crisp_attainment"].mean()
        rep_row  = df_scored.iloc[
            (df_scored["crisp_attainment"] - mean_val).abs().argsort()[:1]
        ]
        highlight = engine.assess(
            assignment_score=float(rep_row["assignment_score"].iloc[0]),
            attendance=float(rep_row["attendance"].iloc[0]),
        )
        engine.plot_membership_functions(
            highlight=highlight,
            save_path=str(MF_PLOT),
        )

    t_total = time.perf_counter() - t_start
    print(f"\n[done]   Total pipeline time: {t_total:.2f} s\n")

    return df_scored


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Phase 1 → Phase 2 OBE assessment pipeline."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the membership function chart.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print fuzzification detail for 3 representative students.",
    )
    args = parser.parse_args()

    try:
        run(save_plot=not args.no_plot, verbose=args.verbose)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print(
            "        Run `python download_dataset.py` to obtain the dataset first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (KeyError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
