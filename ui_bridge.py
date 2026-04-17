"""
ui_bridge.py
============
Maps the three faculty-entered assessment inputs from the dashboard UI
into the two crisp scalar antecedents consumed by the Mamdani FIS.

─────────────────────────────────────────────────────────────────────
Why this module exists  (the 3-input vs 2-antecedent reconciliation)
─────────────────────────────────────────────────────────────────────
The dashboard collects three grading components per student:

    Continuous Assessment (CA)   — periodic tests, quizzes, assignments
    Lab Work                     — practical / hands-on component
    Final Exam                   — end-of-semester summative assessment

The FIS has exactly two antecedents:

    assignment_score  ∈ [0, 100]  — overall academic performance proxy
    attendance        ∈ [0, 100]  — class-participation proxy

Adding a third antecedent (e.g., separating Final Exam) would require
4×4×4 = 64 rules to achieve full Cartesian coverage, which is
disproportionate to the training data size (480–650 rows) and inflates
rule-validation overhead without meaningful precision gain.

Instead, the three academic scores are collapsed into a single weighted
composite via standard OBE grading proportions:

    assignment_score = w_CA  × CA
                     + w_Lab × Lab
                     + w_FE  × Final Exam

The weights below mirror common university credit-hour allocations and
are explicitly declared as module constants so faculty can inspect and
adjust them without touching inference logic.

Attendance is passed through directly — the faculty enters the student's
attendance percentage; no derivation is needed for the UI path (unlike
the xAPI dataset path in feature_bridge.py, which must infer it from
the binary absence_days flag).

─────────────────────────────────────────────────────────────────────
Grading weight rationale
─────────────────────────────────────────────────────────────────────
  Component              Weight   Justification
  ─────────────────────  ──────   ────────────────────────────────────
  Continuous Assessment   0.30    Formative; gauges ongoing engagement
  Lab Work                0.20    Practical competency; skill-based
  Final Exam              0.50    High-stakes summative; covers full CO
  ─────────────────────  ──────
  Total                   1.00
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Grading weights (sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_CA        = 0.30   # Continuous Assessment
WEIGHT_LAB       = 0.20   # Lab Work
WEIGHT_FINAL     = 0.50   # Final Exam


# ---------------------------------------------------------------------------
# Result type (mirrors feature_bridge.FuzzyInputs for a unified contract)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UIFuzzyInputs:
    """
    Crisp inputs derived from faculty-entered UI values, ready to pass
    directly to :meth:`fuzzy_engine.FuzzyAssessmentEngine.assess`.

    Attributes
    ----------
    assignment_score : float
        Weighted composite of CA, Lab, and Final Exam in [0, 100].
    attendance : float
        Faculty-entered attendance percentage in [0, 100].
    continuous_assessment : float
        Raw CA input (stored for display / audit purposes).
    lab_work : float
        Raw Lab input (stored for display / audit purposes).
    final_exam : float
        Raw Final Exam input (stored for display / audit purposes).
    """
    assignment_score: float
    attendance: float
    continuous_assessment: float
    lab_work: float
    final_exam: float

    def __str__(self) -> str:
        sep = "─" * 52
        return "\n".join([
            sep,
            "  UI Bridge — Derived FIS Inputs",
            sep,
            f"  CA          : {self.continuous_assessment:.1f} / 100  (weight {WEIGHT_CA:.0%})",
            f"  Lab Work    : {self.lab_work:.1f} / 100  (weight {WEIGHT_LAB:.0%})",
            f"  Final Exam  : {self.final_exam:.1f} / 100  (weight {WEIGHT_FINAL:.0%})",
            "  " + "─" * 36,
            f"  assignment_score → {self.assignment_score:.2f}",
            f"  attendance       → {self.attendance:.1f} %",
            sep,
        ])


# ---------------------------------------------------------------------------
# Primary mapping function
# ---------------------------------------------------------------------------

def map_ui_inputs(
    continuous_assessment: float,
    lab_work: float,
    final_exam: float,
    attendance: float,
) -> UIFuzzyInputs:
    """
    Convert the four faculty-entered UI values into FIS-ready crisp inputs.

    Parameters
    ----------
    continuous_assessment : float
        CA score in [0, 100].
    lab_work : float
        Lab Work score in [0, 100].
    final_exam : float
        Final Exam score in [0, 100].
    attendance : float
        Attendance percentage in [0, 100].

    Returns
    -------
    UIFuzzyInputs
        Frozen dataclass containing the derived ``assignment_score``,
        the pass-through ``attendance``, and the original raw inputs.

    Raises
    ------
    ValueError
        If any input falls outside [0, 100].
    """
    inputs = {
        "continuous_assessment": continuous_assessment,
        "lab_work":              lab_work,
        "final_exam":            final_exam,
        "attendance":            attendance,
    }
    for name, value in inputs.items():
        if not (0.0 <= value <= 100.0):
            raise ValueError(
                f"'{name}' must be in [0, 100], got {value}. "
                f"All scores and percentages are expressed out of 100."
            )

    assignment_score = float(np.clip(
        WEIGHT_CA    * continuous_assessment
        + WEIGHT_LAB   * lab_work
        + WEIGHT_FINAL * final_exam,
        0.0, 100.0,
    ))

    return UIFuzzyInputs(
        assignment_score=assignment_score,
        attendance=float(attendance),
        continuous_assessment=float(continuous_assessment),
        lab_work=float(lab_work),
        final_exam=float(final_exam),
    )


# ---------------------------------------------------------------------------
# Convenience: directly assess from UI inputs without constructing the bridge
# separately — useful for the API layer in Phase 3 / 4.
# ---------------------------------------------------------------------------

def assess_from_ui(
    continuous_assessment: float,
    lab_work: float,
    final_exam: float,
    attendance: float,
    verbose: bool = False,
) -> "AssessmentResult":  # noqa: F821  (imported lazily to avoid circular dependency)
    """
    One-shot helper: map UI inputs → FIS → :class:`~fuzzy_engine.AssessmentResult`.

    This is the single function the API server (Phase 3) and the frontend
    bridge (Phase 4) will call.  It encapsulates the full UI path:

        UI values → ui_bridge.map_ui_inputs()
                  → fuzzy_engine.assess()
                  → AssessmentResult

    Parameters
    ----------
    continuous_assessment, lab_work, final_exam, attendance : float
        Faculty-entered values in [0, 100].
    verbose : bool
        If True, prints fuzzification membership details.

    Returns
    -------
    AssessmentResult
        Crisp attainment score, linguistic label, and fired rules.
    """
    from fuzzy_engine import assess  # lazy import prevents circular deps

    ui_inputs = map_ui_inputs(
        continuous_assessment=continuous_assessment,
        lab_work=lab_work,
        final_exam=final_exam,
        attendance=attendance,
    )

    return assess(
        assignment_score=ui_inputs.assignment_score,
        attendance=ui_inputs.attendance,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  UI Bridge — Smoke Test")
    print("  " + "═" * 50)

    test_cases = [
        ("High-performing student",   90, 85, 88, 95),
        ("Average student",           55, 60, 58, 72),
        ("Weak academic, present",    30, 40, 35, 90),
        ("Strong academic, absent",   85, 80, 88, 40),
        ("Borderline student",        50, 50, 50, 65),
    ]

    for description, ca, lab, final, att in test_cases:
        from fuzzy_engine import assess
        ui = map_ui_inputs(ca, lab, final, att)
        result = assess(ui.assignment_score, ui.attendance)

        print(f"\n  [{description}]")
        print(ui)
        print(
            f"  FIS Output  → crisp={result.crisp_attainment:.2f}%  "
            f"label={result.label}"
        )
