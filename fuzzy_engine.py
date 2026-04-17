"""
fuzzy_engine.py
===============
Mamdani Fuzzy Inference System for OBE Student Learning Assessment.

Architecture
------------
  Inputs  : assignment_score  ∈ [0, 100]   (marks out of 100)
            attendance        ∈ [0, 100]   (attendance percentage)

  Output  : attainment        ∈ [0, 100]   (outcome attainment percentage)

Membership Functions
--------------------
All MFs are Triangular (three-point: [left, peak, right]).
Overlapping boundaries create smooth transitions so that a score of, e.g.,
55 fires both 'Average' and 'Good' simultaneously rather than snapping to
one category — the defining advantage of fuzzy over crisp classification.

Linguistic Terms
----------------
  assignment_score : Poor | Average | Good | Excellent
  attendance       : Low  | Moderate | High
  attainment       : Poor | Developing | Satisfactory | Good | Excellent

Rule Base  (12 weighted rules — full Cartesian coverage of 4×3 antecedents)
----------
  IF assignment is Poor     AND attendance is Low      → attainment is Poor
  IF assignment is Poor     AND attendance is Moderate → attainment is Developing
  IF assignment is Poor     AND attendance is High     → attainment is Satisfactory
  IF assignment is Average  AND attendance is Low      → attainment is Developing
  IF assignment is Average  AND attendance is Moderate → attainment is Satisfactory
  IF assignment is Average  AND attendance is High     → attainment is Good
  IF assignment is Good     AND attendance is Low      → attainment is Satisfactory
  IF assignment is Good     AND attendance is Moderate → attainment is Good
  IF assignment is Good     AND attendance is High     → attainment is Excellent
  IF assignment is Excellent AND attendance is Low     → attainment is Good
  IF assignment is Excellent AND attendance is Moderate→ attainment is Excellent
  IF assignment is Excellent AND attendance is High    → attainment is Excellent

Defuzzification
---------------
  Centroid method (Centre-of-Gravity):
      z* = ∫ z · μ_agg(z) dz  /  ∫ μ_agg(z) dz
  Returns the precise crisp attainment percentage.

Usage
-----
  from fuzzy_engine import FuzzyAssessmentEngine

  engine = FuzzyAssessmentEngine()
  result = engine.assess(assignment_score=78, attendance=85)
  print(result)
  # AssessmentResult(crisp=74.3, label='Good', ...)

  # or via the module-level convenience wrapper:
  from fuzzy_engine import assess
  result = assess(assignment_score=45, attendance=60)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Universe of discourse
# Resolution of 101 points gives 1-unit granularity across [0,100] — fine
# enough for centroid precision without unnecessary memory cost.
# ---------------------------------------------------------------------------
_UNIVERSE: Final[np.ndarray] = np.linspace(0, 100, 101)

# ---------------------------------------------------------------------------
# Linguistic label → crisp threshold mapping (used only for final labelling,
# NOT for inference — the FIS decides everything numerically).
# ---------------------------------------------------------------------------
_ATTAINMENT_LABELS: Final[list[tuple[float, str]]] = [
    (20.0, "Poor"),
    (40.0, "Developing"),
    (60.0, "Satisfactory"),
    (80.0, "Good"),
    (101.0, "Excellent"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssessmentResult:
    """Immutable result returned by :meth:`FuzzyAssessmentEngine.assess`."""

    assignment_score: float
    attendance: float
    crisp_attainment: float
    label: str
    fired_rules: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "─" * 52,
            f"  Fuzzy OBE Assessment Result",
            "─" * 52,
            f"  Input  │ Assignment Score : {self.assignment_score:.1f} / 100",
            f"  Input  │ Attendance       : {self.attendance:.1f} %",
            "         │",
            f"  Output │ Attainment Score : {self.crisp_attainment:.2f} %",
            f"  Output │ Classification   : {self.label}",
            "─" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class FuzzyAssessmentEngine:
    """
    Singleton-safe Mamdani FIS for OBE student learning assessment.

    The control system and simulation object are built once at construction
    time.  Repeated calls to :meth:`assess` only update the simulation's
    input values — no repeated graph compilation overhead.
    """

    def __init__(self) -> None:
        self._antecedents, self._consequent = self._build_variables()
        self._system    = self._build_control_system()
        self._sim       = ctrl.ControlSystemSimulation(self._system)

    # ------------------------------------------------------------------
    # Step 1 — Linguistic variables and membership functions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_variables() -> tuple[dict, ctrl.Consequent]:
        """
        Define all Antecedent and Consequent objects with triangular MFs.

        Triangular MF three-point notation: [left_foot, peak, right_foot]
          • left_foot  → membership grade rises from 0 to 1
          • peak       → membership grade = 1 (full membership)
          • right_foot → membership grade falls from 1 to 0

        Overlapping shoulders at boundary points ensure continuity:
          e.g., assignment 'Average' peaks at 50 but shares support with
          'Poor' (which falls to 0 at 50) and 'Good' (which rises from 50).
        """

        # ── Assignment Score ────────────────────────────────────────────
        assignment = ctrl.Antecedent(_UNIVERSE, "assignment_score")

        #                      [left,  peak, right]
        assignment["Poor"]      = fuzz.trimf(_UNIVERSE, [  0,   0,  40])
        assignment["Average"]   = fuzz.trimf(_UNIVERSE, [ 20,  45,  65])
        assignment["Good"]      = fuzz.trimf(_UNIVERSE, [ 50,  70,  85])
        assignment["Excellent"] = fuzz.trimf(_UNIVERSE, [ 70, 100, 100])

        # ── Attendance ──────────────────────────────────────────────────
        attendance = ctrl.Antecedent(_UNIVERSE, "attendance")

        attendance["Low"]      = fuzz.trimf(_UNIVERSE, [  0,   0,  50])
        attendance["Moderate"] = fuzz.trimf(_UNIVERSE, [ 30,  60,  80])
        attendance["High"]     = fuzz.trimf(_UNIVERSE, [ 65, 100, 100])

        # ── Attainment (output) ─────────────────────────────────────────
        attainment = ctrl.Consequent(_UNIVERSE, "attainment", defuzzify_method="centroid")

        attainment["Poor"]         = fuzz.trimf(_UNIVERSE, [  0,   0,  25])
        attainment["Developing"]   = fuzz.trimf(_UNIVERSE, [  0,  25,  50])
        attainment["Satisfactory"] = fuzz.trimf(_UNIVERSE, [ 25,  50,  75])
        attainment["Good"]         = fuzz.trimf(_UNIVERSE, [ 50,  75,  90])
        attainment["Excellent"]    = fuzz.trimf(_UNIVERSE, [ 75, 100, 100])

        antecedents = {"assignment_score": assignment, "attendance": attendance}
        return antecedents, attainment

    # ------------------------------------------------------------------
    # Step 2 — Rule base
    # ------------------------------------------------------------------

    def _build_control_system(self) -> ctrl.ControlSystem:
        """
        Construct the 12-rule Mamdani rule base.

        Operator semantics (scikit-fuzzy):
          •  &  → fuzzy AND  (min-of-membership, i.e. Mamdani conjunction)
          •  |  → fuzzy OR   (max-of-membership)

        Each rule's strength is computed as:
            strength = min(μ_assignment(x), μ_attendance(y))
        and is used to clip (implication) the consequent MF before aggregation.
        """
        a = self._antecedents["assignment_score"]
        t = self._antecedents["attendance"]
        o = self._consequent

        rules = [
            # ── Poor assignment ──────────────────────────────────────
            ctrl.Rule(a["Poor"]      & t["Low"],      o["Poor"],         label="R01"),
            ctrl.Rule(a["Poor"]      & t["Moderate"], o["Developing"],   label="R02"),
            ctrl.Rule(a["Poor"]      & t["High"],     o["Satisfactory"], label="R03"),

            # ── Average assignment ───────────────────────────────────
            ctrl.Rule(a["Average"]   & t["Low"],      o["Developing"],   label="R04"),
            ctrl.Rule(a["Average"]   & t["Moderate"], o["Satisfactory"], label="R05"),
            ctrl.Rule(a["Average"]   & t["High"],     o["Good"],         label="R06"),

            # ── Good assignment ──────────────────────────────────────
            ctrl.Rule(a["Good"]      & t["Low"],      o["Satisfactory"], label="R07"),
            ctrl.Rule(a["Good"]      & t["Moderate"], o["Good"],         label="R08"),
            ctrl.Rule(a["Good"]      & t["High"],     o["Excellent"],    label="R09"),

            # ── Excellent assignment ─────────────────────────────────
            ctrl.Rule(a["Excellent"] & t["Low"],      o["Good"],         label="R10"),
            ctrl.Rule(a["Excellent"] & t["Moderate"], o["Excellent"],    label="R11"),
            ctrl.Rule(a["Excellent"] & t["High"],     o["Excellent"],    label="R12"),
        ]

        return ctrl.ControlSystem(rules)

    # ------------------------------------------------------------------
    # Step 3 — Inference + Defuzzification
    # ------------------------------------------------------------------

    def assess(
        self,
        assignment_score: float,
        attendance: float,
        verbose: bool = False,
    ) -> AssessmentResult:
        """
        Run the full Mamdani inference pipeline for one student record.

        Parameters
        ----------
        assignment_score : float
            Raw assignment marks in [0, 100].
        attendance : float
            Attendance percentage in [0, 100].
        verbose : bool, optional
            If True, prints per-term membership activations before returning.

        Returns
        -------
        AssessmentResult
            Immutable result containing the centroid crisp score, linguistic
            label, and the list of rules that fired (strength > 0).

        Raises
        ------
        ValueError
            If either input is outside [0, 100].
        """
        if not (0.0 <= assignment_score <= 100.0):
            raise ValueError(
                f"assignment_score must be in [0, 100], got {assignment_score}"
            )
        if not (0.0 <= attendance <= 100.0):
            raise ValueError(
                f"attendance must be in [0, 100], got {attendance}"
            )

        # ── Fuzzification: compute membership grades for each input term ──
        a_var = self._antecedents["assignment_score"]
        t_var = self._antecedents["attendance"]

        assignment_memberships = {
            term: float(fuzz.interp_membership(_UNIVERSE, a_var[term].mf, assignment_score))
            for term in a_var.terms
        }
        attendance_memberships = {
            term: float(fuzz.interp_membership(_UNIVERSE, t_var[term].mf, attendance))
            for term in t_var.terms
        }

        if verbose:
            _print_memberships("Assignment Score", assignment_score, assignment_memberships)
            _print_memberships("Attendance",       attendance,       attendance_memberships)

        # ── Rules that fire (min-conjunction strength > 0) ────────────────
        fired = _compute_fired_rules(assignment_memberships, attendance_memberships)

        # ── Fuzzy inference via scikit-fuzzy ─────────────────────────────
        self._sim.input["assignment_score"] = assignment_score
        self._sim.input["attendance"]       = attendance
        self._sim.compute()

        crisp: float = round(float(self._sim.output["attainment"]), 2)
        label: str   = _crisp_to_label(crisp)

        return AssessmentResult(
            assignment_score=assignment_score,
            attendance=attendance,
            crisp_attainment=crisp,
            label=label,
            fired_rules=fired,
        )

    # ------------------------------------------------------------------
    # Batch inference (full dataset)
    # ------------------------------------------------------------------

    def batch_assess(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Run FIS inference over every row of a bridged DataFrame and return
        the original frame augmented with two new columns:

            ``crisp_attainment``  — centroid defuzzified score  [0, 100]
            ``attainment_label``  — linguistic classification string

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``assignment_score`` and ``attendance`` columns as
            produced by :func:`feature_bridge.dataframe_to_fuzzy_inputs`.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the two result columns appended.

        Raises
        ------
        KeyError
            If either required column is absent from *df*.

        Notes
        -----
        The loop is intentional: ``skfuzzy.control.ControlSystemSimulation``
        is not thread-safe and does not expose a vectorised compute path.
        At 480–650 rows the overhead is negligible (< 2 s on a laptop CPU).
        """
        import pandas as pd

        missing = [c for c in ("assignment_score", "attendance") if c not in df.columns]
        if missing:
            raise KeyError(
                f"batch_assess requires columns {missing}. "
                f"Run feature_bridge.dataframe_to_fuzzy_inputs() first."
            )

        crisp_scores: list[float] = []
        labels: list[str] = []

        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            result = self.assess(
                assignment_score=float(row["assignment_score"]),
                attendance=float(row["attendance"]),
            )
            crisp_scores.append(result.crisp_attainment)
            labels.append(result.label)

            if i % 100 == 0 or i == total:
                print(f"  [batch]  {i}/{total} rows processed …", end="\r")

        print()

        out = df.copy()
        out["crisp_attainment"] = crisp_scores
        out["attainment_label"] = labels
        return out

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def plot_membership_functions(
        self,
        highlight: AssessmentResult | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Render all three membership function diagrams side-by-side.

        Parameters
        ----------
        highlight : AssessmentResult, optional
            When provided, vertical dashed lines mark the input values and
            a dot marks the defuzzified output on the attainment axis.
        save_path : str, optional
            If given, saves the figure to this path instead of displaying it.
        """
        matplotlib.use("Agg") if save_path else None

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(
            "OBE Fuzzy Membership Functions  —  Mamdani FIS",
            fontsize=13, fontweight="bold", y=1.02,
        )

        # ── Assignment Score ─────────────────────────────────────────────
        ax = axes[0]
        a_var = self._antecedents["assignment_score"]
        colours = {"Poor": "#e74c3c", "Average": "#e67e22",
                   "Good": "#2ecc71", "Excellent": "#3498db"}
        for term, colour in colours.items():
            ax.plot(_UNIVERSE, a_var[term].mf, label=term, color=colour, lw=2)
            ax.fill_between(_UNIVERSE, 0, a_var[term].mf, alpha=0.08, color=colour)
        if highlight:
            ax.axvline(highlight.assignment_score, color="black",
                       linestyle="--", lw=1.4, label=f"Input={highlight.assignment_score}")
        ax.set_title("Assignment Score", fontsize=11)
        ax.set_xlabel("Score (0–100)")
        ax.set_ylabel("Membership grade  μ")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # ── Attendance ────────────────────────────────────────────────────
        ax = axes[1]
        t_var = self._antecedents["attendance"]
        colours_t = {"Low": "#e74c3c", "Moderate": "#e67e22", "High": "#2ecc71"}
        for term, colour in colours_t.items():
            ax.plot(_UNIVERSE, t_var[term].mf, label=term, color=colour, lw=2)
            ax.fill_between(_UNIVERSE, 0, t_var[term].mf, alpha=0.08, color=colour)
        if highlight:
            ax.axvline(highlight.attendance, color="black",
                       linestyle="--", lw=1.4, label=f"Input={highlight.attendance}")
        ax.set_title("Attendance", fontsize=11)
        ax.set_xlabel("Attendance %")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # ── Attainment (output) ───────────────────────────────────────────
        ax = axes[2]
        o_var = self._consequent
        colours_o = {
            "Poor": "#e74c3c", "Developing": "#e67e22", "Satisfactory": "#f1c40f",
            "Good": "#2ecc71", "Excellent": "#3498db",
        }
        for term, colour in colours_o.items():
            ax.plot(_UNIVERSE, o_var[term].mf, label=term, color=colour, lw=2)
            ax.fill_between(_UNIVERSE, 0, o_var[term].mf, alpha=0.08, color=colour)
        if highlight:
            ax.axvline(highlight.crisp_attainment, color="black",
                       linestyle="--", lw=1.4,
                       label=f"Centroid={highlight.crisp_attainment:.1f}%")
            ax.scatter([highlight.crisp_attainment], [0.5], color="black",
                       zorder=5, s=60, clip_on=False)
        ax.set_title("Attainment Output", fontsize=11)
        ax.set_xlabel("Attainment %")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[plot]  Membership function chart saved → {save_path}")
        else:
            plt.show()

        plt.close(fig)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _crisp_to_label(value: float) -> str:
    """Map a defuzzified attainment score to a linguistic classification."""
    for threshold, label in _ATTAINMENT_LABELS:
        if value < threshold:
            return label
    return "Excellent"


def _print_memberships(
    variable_name: str,
    crisp_value: float,
    memberships: dict[str, float],
) -> None:
    """Pretty-print per-term membership activation for a single input."""
    print(f"\n  [{variable_name}]  crisp input = {crisp_value:.1f}")
    for term, grade in memberships.items():
        bar = "█" * int(grade * 20)
        print(f"    {term:<14} μ = {grade:.4f}  {bar}")


def _compute_fired_rules(
    assignment_memberships: dict[str, float],
    attendance_memberships: dict[str, float],
) -> list[str]:
    """
    Return human-readable labels for rules whose conjunction strength > 0.
    Used purely for diagnostics; the scikit-fuzzy engine handles actual
    rule activation internally.
    """
    _RULE_TABLE: list[tuple[str, str, str]] = [
        ("Poor",      "Low",      "R01"),
        ("Poor",      "Moderate", "R02"),
        ("Poor",      "High",     "R03"),
        ("Average",   "Low",      "R04"),
        ("Average",   "Moderate", "R05"),
        ("Average",   "High",     "R06"),
        ("Good",      "Low",      "R07"),
        ("Good",      "Moderate", "R08"),
        ("Good",      "High",     "R09"),
        ("Excellent", "Low",      "R10"),
        ("Excellent", "Moderate", "R11"),
        ("Excellent", "High",     "R12"),
    ]

    fired = []
    for a_term, t_term, label in _RULE_TABLE:
        strength = min(
            assignment_memberships.get(a_term, 0.0),
            attendance_memberships.get(t_term, 0.0),
        )
        if strength > 0.0:
            fired.append(
                f"{label}: assignment={a_term}, attendance={t_term}  "
                f"(strength={strength:.4f})"
            )
    return fired


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

# A single shared engine instance avoids re-compiling the control system
# graph on every standalone call while remaining safe for sequential use.
_DEFAULT_ENGINE: FuzzyAssessmentEngine | None = None


def assess(
    assignment_score: float,
    attendance: float,
    verbose: bool = False,
) -> AssessmentResult:
    """
    Module-level wrapper around :class:`FuzzyAssessmentEngine`.

    Initialises the shared engine on first call (lazy singleton), then
    delegates to :meth:`FuzzyAssessmentEngine.assess`.

    Parameters
    ----------
    assignment_score : float  — marks out of 100
    attendance       : float  — attendance percentage
    verbose          : bool   — print fuzzification details if True

    Returns
    -------
    AssessmentResult
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = FuzzyAssessmentEngine()
    return _DEFAULT_ENGINE.assess(assignment_score, attendance, verbose=verbose)


# ---------------------------------------------------------------------------
# Demo  (python fuzzy_engine.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = FuzzyAssessmentEngine()

    test_cases: list[tuple[float, float, str]] = [
        (15,  30, "Very weak student"),
        (45,  55, "Below-average student"),
        (62,  72, "Mid-range student"),
        (78,  85, "Good student"),
        (95,  95, "Excellent student"),
        (50,  50, "Boundary case — all inputs at midpoint"),
    ]

    print("\n" + "═" * 52)
    print("  Mamdani FIS — OBE Assessment Demo")
    print("═" * 52)

    for assignment, attendance, description in test_cases:
        result = engine.assess(assignment, attendance, verbose=False)
        print(f"\n  [{description}]")
        print(
            textwrap.indent(str(result), "  ")
        )
        if result.fired_rules:
            print("  Fired rules:")
            for rule in result.fired_rules:
                print(f"    ► {rule}")

    # Save membership function chart with the mid-range student highlighted
    mid_result = engine.assess(62, 72)
    engine.plot_membership_functions(
        highlight=mid_result,
        save_path="membership_functions.png",
    )
