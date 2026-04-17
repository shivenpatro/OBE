"""
api_server.py
=============
FastAPI server that bridges the Next.js dashboard (Phase 4) to the
Python FIS + LLM pipeline (Phases 2 & 3).

Endpoints
---------
  POST /api/assess    — Full pipeline: UI inputs → FIS → LLM → JSON response
  GET  /api/health    — Liveness check + LM Studio connectivity status
  GET  /api/sample    — Return a sample payload for frontend development/testing

Architecture
------------
  Next.js (port 3000)
       │  fetch('/api/assess', { method: 'POST', body: JSON })
       ▼
  FastAPI (port 8000)    ← this file
       │
       ├─ ui_bridge.map_ui_inputs()         Phase 1/2 glue
       ├─ FuzzyAssessmentEngine.assess()    Phase 2 FIS
       └─ agentic_feedback.generate_feedback()  Phase 3 LLM
       │
       └─ JSON response → Next.js → Dashboard render

CORS
----
Configured to allow requests from http://localhost:3000 (Next.js dev server)
and http://localhost:3001 (alternate Next.js port).  In production, replace
with your actual deployment origin.

Usage
-----
  # Development
  uvicorn api_server:app --reload --port 8000

  # Production (single worker — FIS engine is stateful, multi-worker unsafe)
  uvicorn api_server:app --port 8000 --workers 1
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agentic_feedback import FeedbackResult, LMStudioClient, generate_feedback
from fuzzy_engine import FuzzyAssessmentEngine
from ui_bridge import UIFuzzyInputs, map_ui_inputs

# ---------------------------------------------------------------------------
# Application lifecycle — build the FIS engine once at startup
# ---------------------------------------------------------------------------

_engine: FuzzyAssessmentEngine | None = None
_lm_client: LMStudioClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build expensive singletons once on server startup, not per-request.
    FuzzyAssessmentEngine compiles the skfuzzy control graph at __init__;
    keeping a single instance avoids ~50 ms overhead per request.
    """
    global _engine, _lm_client
    print("[startup] Building Mamdani FIS engine …")
    _engine    = FuzzyAssessmentEngine()
    _lm_client = LMStudioClient()
    print("[startup] FIS engine ready.")
    print(f"[startup] LM Studio available: {_lm_client.is_available()}")
    yield
    # Teardown (nothing needed for these objects)
    print("[shutdown] Server stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OBE Fuzzy Assessment API",
    description=(
        "REST API for the Fuzzy-Logic-Based Student Learning Assessment System. "
        "Accepts faculty-entered scores, runs the Mamdani FIS, and returns the "
        "defuzzified attainment score plus AI-generated feedback from a local LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Allow the Next.js dev server to call this API without a proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AssessRequest(BaseModel):
    """
    Payload sent by the Next.js dashboard when the faculty clicks 'Analyse'.
    All scores are expressed as percentages out of 100.
    """
    continuous_assessment: float = Field(
        ...,
        ge=0, le=100,
        description="Continuous Assessment score (quizzes + internal tests), 0–100",
        examples=[72.0],
    )
    lab_work: float = Field(
        ...,
        ge=0, le=100,
        description="Lab Work score (practical component), 0–100",
        examples=[68.0],
    )
    final_exam: float = Field(
        ...,
        ge=0, le=100,
        description="Final Examination score, 0–100",
        examples=[74.0],
    )
    attendance: float = Field(
        ...,
        ge=0, le=100,
        description="Attendance percentage, 0–100",
        examples=[85.0],
    )

    @field_validator("continuous_assessment", "lab_work", "final_exam", "attendance")
    @classmethod
    def round_to_two_dp(cls, v: float) -> float:
        return round(v, 2)


class FISResult(BaseModel):
    assignment_score: float
    attendance: float
    crisp_attainment: float
    label: str
    fired_rules: list[str]


class AssessResponse(BaseModel):
    """
    Full pipeline response returned to the Next.js dashboard.
    Always has the same shape — ``llm_available`` signals whether the
    LLM produced the feedback or the rule-based fallback was used.
    """
    # Raw inputs echoed back for display
    continuous_assessment: float
    lab_work: float
    final_exam: float
    attendance: float

    # FIS outputs
    fis: FISResult

    # LLM / fallback feedback
    breakdown: str
    study_schedule: Any          # list[dict] from LLM or list[dict] fallback
    weak_areas: list[str]
    llm_available: bool
    latency_ms: float
    model_used: str

    # Pipeline timing
    pipeline_ms: float


class HealthResponse(BaseModel):
    status: str
    fis_engine: bool
    lm_studio: bool
    lm_studio_url: str


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns the status of the FIS engine and LM Studio connectivity so
    the frontend can show an appropriate indicator (e.g., an offline badge
    when LM Studio is not running).
    """
    lm_available = _lm_client.is_available() if _lm_client else False
    return HealthResponse(
        status="ok",
        fis_engine=_engine is not None,
        lm_studio=lm_available,
        lm_studio_url="http://localhost:1234",
    )


@app.post("/api/assess", response_model=AssessResponse, tags=["Assessment"])
async def assess(payload: AssessRequest) -> AssessResponse:
    """
    Full OBE assessment pipeline in a single request.

    Execution order
    ---------------
    1. Map UI inputs → FIS antecedents (ui_bridge)
    2. Run Mamdani FIS inference (fuzzy_engine)
    3. Extract weak areas deterministically (agentic_feedback)
    4. Generate LLM feedback via LM Studio (agentic_feedback)
       — falls back to rule-based output if LM Studio is offline

    The response always has the same schema regardless of LLM availability.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="FIS engine not initialised.")

    t_start = time.perf_counter()

    # ── Step 1+2: UI bridge + FIS ──────────────────────────────────────────
    ui_inputs: UIFuzzyInputs = map_ui_inputs(
        continuous_assessment=payload.continuous_assessment,
        lab_work=payload.lab_work,
        final_exam=payload.final_exam,
        attendance=payload.attendance,
    )

    fis_result = _engine.assess(
        assignment_score=ui_inputs.assignment_score,
        attendance=ui_inputs.attendance,
    )

    # ── Step 3+4: Weak-area extraction + LLM ──────────────────────────────
    feedback: FeedbackResult = generate_feedback(
        result=fis_result,
        ui_inputs=ui_inputs,
        client=_lm_client,
    )

    pipeline_ms = (time.perf_counter() - t_start) * 1000

    return AssessResponse(
        continuous_assessment=payload.continuous_assessment,
        lab_work=payload.lab_work,
        final_exam=payload.final_exam,
        attendance=payload.attendance,
        fis=FISResult(
            assignment_score=fis_result.assignment_score,
            attendance=fis_result.attendance,
            crisp_attainment=fis_result.crisp_attainment,
            label=fis_result.label,
            fired_rules=fis_result.fired_rules,
        ),
        breakdown=feedback.breakdown,
        study_schedule=feedback.study_schedule,
        weak_areas=feedback.weak_areas,
        llm_available=feedback.llm_available,
        latency_ms=feedback.latency_ms,
        model_used=feedback.model_used,
        pipeline_ms=round(pipeline_ms, 1),
    )


@app.get("/api/sample", tags=["Development"])
async def sample_payload() -> dict:
    """
    Return a sample request payload and a simulated response for frontend
    development when the Python server is not available.
    Used by the Next.js dashboard in mock mode.
    """
    return {
        "request": {
            "continuous_assessment": 72.0,
            "lab_work": 68.0,
            "final_exam": 74.0,
            "attendance": 85.0,
        },
        "note": "POST this payload to /api/assess for a real inference result.",
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n  OBE Assessment API Server")
    print("  ─────────────────────────")
    print("  API docs : http://localhost:8000/docs")
    print("  Health   : http://localhost:8000/api/health")
    print("  Assess   : POST http://localhost:8000/api/assess\n")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,    # Single worker — FIS engine is not multiprocess-safe
    )
