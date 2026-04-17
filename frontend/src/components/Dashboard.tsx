"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─── API Types (mirrors api_server.py AssessResponse) ────────────────────────

interface StudyWeek {
  week: number;
  focus: string;
  tasks: string[];
}

interface FISDetails {
  assignment_score: number;
  attendance: number;
  crisp_attainment: number;
  label: string;
  fired_rules: string[];
}

interface AssessResponse {
  continuous_assessment: number;
  lab_work: number;
  final_exam: number;
  attendance: number;
  fis: FISDetails;
  breakdown: string;
  study_schedule: StudyWeek[] | string;
  weak_areas: string[];
  llm_available: boolean;
  latency_ms: number;
  model_used: string;
  pipeline_ms: number;
}

// ─── Label → accent colour ────────────────────────────────────────────────────

const LABEL_STYLE: Record<string, string> = {
  Poor:         "bg-red-500 text-white",
  Developing:   "bg-orange-400 text-black",
  Satisfactory: "bg-yellow-400 text-black",
  Good:         "bg-blue-500 text-white",
  Excellent:    "bg-green-500 text-white",
};

// ─── Gauge component ──────────────────────────────────────────────────────────

function Gauge({ score, isDark }: { score: number; isDark: boolean }) {
  const ARC = 125;
  const offset = ARC - (ARC * score) / 100;
  return (
    <div className="relative w-60 h-32 mx-auto select-none">
      <svg className="w-full h-full" viewBox="0 0 100 52">
        <defs>
          <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#ef4444" />
            <stop offset="50%"  stopColor="#eab308" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>
        {/* Track */}
        <path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke={isDark ? "#2a2a2a" : "#e2e8f0"}
          strokeWidth="9"
          strokeLinecap="butt"
        />
        {/* Fill */}
        <motion.path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke="url(#gGrad)"
          strokeWidth="9"
          strokeLinecap="butt"
          initial={{ strokeDasharray: ARC, strokeDashoffset: ARC }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.6, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-end pb-1 pointer-events-none">
        <motion.div
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.7, type: "spring", stiffness: 200 }}
          className="flex items-baseline gap-0.5"
        >
          <span className="text-5xl font-black tracking-tighter tabular-nums">
            {score.toFixed(1)}
          </span>
          <span className={`text-xl font-black ${isDark ? "text-gray-400" : "text-gray-400"}`}>
            %
          </span>
        </motion.div>
      </div>
    </div>
  );
}

// ─── Inline tag / badge ───────────────────────────────────────────────────────

function Tag({
  children,
  colour = "bg-yellow-400 text-black",
  isDark,
}: {
  children: React.ReactNode;
  colour?: string;
  isDark: boolean;
}) {
  return (
    <span
      className={`inline-block px-2 py-0.5 text-[11px] font-black uppercase tracking-widest border ${
        isDark ? "border-white" : "border-black"
      } ${colour}`}
    >
      {children}
    </span>
  );
}

// ─── Input field ──────────────────────────────────────────────────────────────

function ScoreInput({
  label,
  hint,
  name,
  value,
  onChange,
  isDark,
}: {
  label: string;
  hint: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  isDark: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label
          htmlFor={name}
          className="text-xs font-black uppercase tracking-widest"
        >
          {label}
        </label>
        <span className={`text-xs ${isDark ? "text-gray-400" : "text-gray-500"}`}>
          {hint}
        </span>
      </div>
      <input
        id={name}
        name={name}
        type="number"
        value={value}
        onChange={onChange}
        required
        min="0"
        max="100"
        step="0.1"
        placeholder="0 – 100"
        className={`w-full p-3 text-sm font-mono transition-colors ${
          isDark
            ? "bg-[#111] border-2 border-white text-white placeholder:text-gray-500 focus:border-yellow-400 outline-none"
            : "bg-white border-2 border-black text-black placeholder:text-gray-400 focus:border-yellow-500 outline-none"
        }`}
      />
    </div>
  );
}

// ─── Stat pill ────────────────────────────────────────────────────────────────

function StatPill({
  label,
  value,
  isDark,
}: {
  label: string;
  value: string;
  isDark: boolean;
}) {
  return (
    <div className={`p-2 border ${isDark ? "border-gray-600" : "border-gray-300"}`}>
      <p className={`text-xs uppercase tracking-widest font-black ${isDark ? "text-gray-400" : "text-gray-500"}`}>
        {label}
      </p>
      <p className="text-sm font-black mt-1 tabular-nums">{value}</p>
    </div>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export default function NeoBrutalistDashboard() {
  const [isDark, setIsDark] = useState(false);
  const [scores, setScores] = useState({
    assessment: "",
    lab: "",
    exam: "",
    attendance: "",
  });
  const [result, setResult]   = useState<AssessResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showRules, setShowRules] = useState(false);

  // ── Theme ──────────────────────────────────────────────────────────────────
  const toggleDark = () => {
    setIsDark((d) => {
      const next = !d;
      document.documentElement.classList.toggle("dark", next);
      return next;
    });
  };

  // ── Input ──────────────────────────────────────────────────────────────────
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setScores((prev) => ({ ...prev, [name]: value }));
  };

  // ── Submit → real fetch to FastAPI (proxied via next.config.ts rewrites) ───
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    setShowRules(false);

    try {
      const res = await fetch("/api/assess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          continuous_assessment: parseFloat(scores.assessment),
          lab_work:              parseFloat(scores.lab),
          final_exam:            parseFloat(scores.exam),
          attendance:            parseFloat(scores.attendance),
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail ?? `HTTP ${res.status}`);
      }

      const data: AssessResponse = await res.json();
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      if (msg.includes("fetch") || msg.includes("Failed") || msg.includes("NetworkError")) {
        setError(
          "Cannot reach the API server. Make sure it is running:\n" +
          "  uvicorn api_server:app --reload --port 8000"
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Class helpers ──────────────────────────────────────────────────────────
  const card = isDark
    ? "bg-[#1a1a1a] border-2 border-white shadow-[4px_4px_0_0_rgba(255,255,255,0.75)]"
    : "bg-white border-2 border-black shadow-[4px_4px_0_0_black]";

  const btnBase =
    "w-full py-4 font-black text-sm uppercase tracking-widest transition-all duration-100 disabled:opacity-40 disabled:cursor-not-allowed disabled:translate-x-0 disabled:translate-y-0 disabled:shadow-none";
  const btnColour = isDark
    ? "bg-yellow-400 text-black border-2 border-white shadow-[4px_4px_0_0_rgba(255,255,255,0.75)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0_0_rgba(255,255,255,0.75)] active:translate-x-[4px] active:translate-y-[4px] active:shadow-none"
    : "bg-black text-white border-2 border-black shadow-[4px_4px_0_0_black] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0_0_black] active:translate-x-[4px] active:translate-y-[4px] active:shadow-none";

  // Normalise study_schedule to array regardless of model output shape
  const studySchedule: StudyWeek[] = Array.isArray(result?.study_schedule)
    ? (result.study_schedule as StudyWeek[])
    : [];

  const labelCls = LABEL_STYLE[result?.fis.label ?? ""] ?? "bg-gray-400 text-white";

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className={`min-h-screen transition-colors duration-200 ${
        isDark ? "bg-[#111111] text-white" : "bg-[#FFFCF0] text-black"
      }`}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header
        className={`sticky top-0 z-50 border-b-2 ${
          isDark ? "bg-[#111111] border-white" : "bg-[#FFFCF0] border-black"
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-10 h-10 bg-yellow-400 border-2 border-black flex items-center justify-center font-black text-black text-sm shrink-0">
              OBE
            </div>
            <div className="flex flex-col justify-center">
              <p className="font-black text-base uppercase tracking-tight leading-tight">
                Fuzzy Assessment System
              </p>
              <p className={`text-xs uppercase tracking-tight leading-tight ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                Mamdani FIS · Local LLM · Zero-Egress
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleDark}
              className={`px-3 py-1.5 text-xs font-black uppercase tracking-widest border-2 transition-all ${
                isDark
                  ? "border-white text-white hover:bg-white hover:text-black"
                  : "border-black text-black hover:bg-black hover:text-white"
              }`}
            >
              {isDark ? "☀ Light" : "☾ Dark"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <div className="mb-10">
          <h1 className="text-4xl md:text-5xl font-black uppercase leading-[1.05] tracking-tighter mb-3">
            Student<br />
            <span className="bg-yellow-400 text-black px-2 inline-block">
              Assessment Engine
            </span>
          </h1>
          <p
            className={`max-w-2xl text-sm leading-relaxed mt-4 ${
              isDark ? "text-gray-300" : "text-gray-600"
            }`}
          >
            Enter crisp score values. The Mamdani FIS maps them through triangular
            membership functions, aggregates 12 fuzzy rules, and
            centroid-defuzzifies to a precise OBE attainment metric. A local
            zero-egress LLM then generates personalised feedback — no student
            data leaves this machine.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* ── Input Panel ──────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            className={`lg:col-span-4 p-6 ${card}`}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-black text-sm uppercase tracking-widest">
                Crisp Inputs
              </h2>
              <Tag isDark={isDark}>FIS Engine</Tag>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <ScoreInput
                label="Continuous Assessment"
                hint="Weight 30%"
                name="assessment"
                value={scores.assessment}
                onChange={handleChange}
                isDark={isDark}
              />
              <ScoreInput
                label="Lab Work"
                hint="Weight 20%"
                name="lab"
                value={scores.lab}
                onChange={handleChange}
                isDark={isDark}
              />
              <ScoreInput
                label="Final Exam"
                hint="Weight 50%"
                name="exam"
                value={scores.exam}
                onChange={handleChange}
                isDark={isDark}
              />
              <ScoreInput
                label="Attendance %"
                hint="Class participation"
                name="attendance"
                value={scores.attendance}
                onChange={handleChange}
                isDark={isDark}
              />

              <button
                type="submit"
                disabled={loading}
                className={`${btnBase} ${btnColour}`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    Running Inference…
                  </span>
                ) : (
                  "▶ Run FIS + LLM"
                )}
              </button>
            </form>

            {/* Weight reminder */}
            <div
              className={`mt-6 p-3 border text-xs leading-relaxed font-mono ${
                isDark ? "border-gray-600 text-gray-400" : "border-gray-300 text-gray-500"
              }`}
            >
              <p className="font-black uppercase mb-1">Score formula</p>
              <p>assignment = 0.30·CA + 0.20·Lab + 0.50·Final</p>
              <p>attendance = direct pass-through</p>
            </div>
          </motion.div>

          {/* ── Results Panel ────────────────────────────────────────────── */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            <AnimatePresence mode="wait">

              {/* Empty state */}
              {!result && !loading && !error && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`flex flex-col items-center justify-center min-h-[440px] border-2 border-dashed p-12 text-center ${
                    isDark ? "border-gray-700 text-gray-400" : "border-gray-300 text-gray-500"
                  }`}
                >
                  <p className="text-7xl font-black opacity-10 select-none mb-4">∿</p>
                  <p className="font-black uppercase text-xs tracking-widest mb-2">
                    Awaiting Input
                  </p>
                  <p className="text-sm max-w-xs leading-relaxed">
                    Fill in the four score fields and click{" "}
                    <strong>Run FIS + LLM</strong> to start the inference
                    pipeline.
                  </p>
                </motion.div>
              )}

              {/* Loading skeleton */}
              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {[180, 100, 280].map((h, i) => (
                    <div
                      key={i}
                      className={`animate-pulse border-2 ${
                        isDark
                          ? "bg-gray-900 border-gray-800"
                          : "bg-gray-100 border-gray-200"
                      }`}
                      style={{ height: h }}
                    />
                  ))}
                  <p
                    className={`text-xs font-black uppercase tracking-widest text-center ${
                      isDark ? "text-gray-400" : "text-gray-500"
                    }`}
                  >
                    Mamdani inference → LLM generation in progress…
                  </p>
                </motion.div>
              )}

              {/* Error state */}
              {error && !loading && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`p-6 border-2 border-red-500 shadow-[4px_4px_0_0_rgb(239,68,68)] ${
                    isDark ? "bg-[#1a1a1a]" : "bg-white"
                  }`}
                >
                  <p className="font-black uppercase text-red-500 text-xs tracking-widest mb-2">
                    ✗ Pipeline Error
                  </p>
                  <p className="text-xs font-mono whitespace-pre-wrap">{error}</p>
                </motion.div>
              )}

              {/* Results */}
              {result && !loading && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-6"
                >
                  {/* ── Gauge card ─────────────────────────────────────── */}
                  <div className={`p-6 ${card}`}>
                    <div className="flex items-center justify-between mb-5 flex-wrap gap-2">
                      <h3 className="font-black uppercase text-sm tracking-widest">
                        Attainment Score
                      </h3>
                      <div className="flex items-center gap-2 flex-wrap">
                        <Tag colour="bg-emerald-400 text-black" isDark={isDark}>
                          Centroid Defuzzified
                        </Tag>
                        {!result.llm_available && (
                          <Tag colour="bg-yellow-400 text-black" isDark={isDark}>
                            ⚠ LLM Offline — Fallback Active
                          </Tag>
                        )}
                      </div>
                    </div>

                    <div className="flex flex-col md:flex-row items-center gap-8">
                      <Gauge score={result.fis.crisp_attainment} isDark={isDark} />

                      <div className="flex-1 space-y-4">
                        {/* Classification label */}
                        <div>
                          <p
                            className={`text-xs font-black uppercase tracking-widest mb-1 ${
                              isDark ? "text-gray-400" : "text-gray-500"
                            }`}
                          >
                            OBE Classification
                          </p>
                          <span
                            className={`inline-block px-4 py-1.5 font-black text-sm uppercase tracking-wider border-2 ${
                              isDark ? "border-white" : "border-black"
                            } ${labelCls}`}
                          >
                            {result.fis.label}
                          </span>
                        </div>

                        {/* Stat grid */}
                        <div className="grid grid-cols-2 gap-2">
                          <StatPill
                            label="Assignment Score"
                            value={`${result.fis.assignment_score.toFixed(1)} / 100`}
                            isDark={isDark}
                          />
                          <StatPill
                            label="Attendance"
                            value={`${result.fis.attendance.toFixed(1)} %`}
                            isDark={isDark}
                          />
                          <StatPill
                            label="LLM Latency"
                            value={`${(result.latency_ms / 1000).toFixed(2)} s`}
                            isDark={isDark}
                          />
                          <StatPill
                            label="Pipeline Total"
                            value={`${(result.pipeline_ms / 1000).toFixed(2)} s`}
                            isDark={isDark}
                          />
                        </div>

                        <p
                          className={`text-xs font-mono ${
                            isDark ? "text-gray-400" : "text-gray-500"
                          }`}
                        >
                          Model: {result.model_used}
                        </p>
                      </div>
                    </div>

                    {/* Fired fuzzy rules (collapsible) */}
                    {result.fis.fired_rules.length > 0 && (
                      <div className="mt-5 pt-4 border-t border-dashed border-gray-500">
                        <button
                          onClick={() => setShowRules((r) => !r)}
                          className={`text-xs font-black uppercase tracking-widest underline underline-offset-2 ${
                            isDark ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-black"
                          }`}
                        >
                          {showRules ? "▼" : "▶"}{" "}
                          {result.fis.fired_rules.length} Fuzzy Rules Fired
                        </button>
                        <AnimatePresence>
                          {showRules && (
                            <motion.ul
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className={`overflow-hidden mt-2 font-mono text-xs space-y-1.5 p-3 border ${
                                isDark
                                  ? "border-gray-600 text-gray-300"
                                  : "border-gray-300 text-gray-600"
                              }`}
                            >
                              {result.fis.fired_rules.map((r, i) => (
                                <li key={i}>► {r}</li>
                              ))}
                            </motion.ul>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                  </div>

                  {/* ── Weak areas ─────────────────────────────────────── */}
                  {result.weak_areas.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                      className={`p-6 ${card}`}
                    >
                      <div className="flex items-center gap-3 mb-4 flex-wrap">
                        <h3 className="font-black uppercase text-sm tracking-widest">
                          Identified Weak Areas
                        </h3>
                        <Tag colour="bg-red-500 text-white" isDark={isDark}>
                          {result.weak_areas.length} flagged
                        </Tag>
                      </div>
                      <ul className="space-y-2">
                        {result.weak_areas.map((area, i) => (
                          <li
                            key={i}
                            className={`flex items-start gap-2 text-xs p-2 border-l-4 border-red-500 ${
                              isDark ? "bg-red-950/20" : "bg-red-50"
                            }`}
                          >
                            <span className="text-red-500 font-black shrink-0 mt-0.5">
                              ✗
                            </span>
                            <span className="font-mono leading-relaxed">{area}</span>
                          </li>
                        ))}
                      </ul>
                    </motion.div>
                  )}

                  {/* ── LLM Breakdown ──────────────────────────────────── */}
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className={`p-6 ${card}`}
                  >
                    <div className="flex items-center gap-3 mb-4 flex-wrap">
                      <h3 className="font-black uppercase text-sm tracking-widest">
                        AI Deficiency Analysis
                      </h3>
                      <Tag
                        colour={
                          result.llm_available
                            ? "bg-green-400 text-black"
                            : "bg-yellow-400 text-black"
                        }
                        isDark={isDark}
                      >
                        {result.llm_available ? "✓ LLM Generated" : "⚠ Rule-Based Fallback"}
                      </Tag>
                    </div>
                    <p
                      className={`text-xs font-mono leading-relaxed whitespace-pre-wrap ${
                        isDark ? "text-gray-300" : "text-gray-700"
                      }`}
                    >
                      {result.breakdown}
                    </p>
                  </motion.div>

                  {/* ── Study Schedule ─────────────────────────────────── */}
                  {studySchedule.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className={`p-6 ${card}`}
                    >
                      <div className="flex items-center gap-3 mb-5 flex-wrap">
                        <h3 className="font-black uppercase text-sm tracking-widest">
                          Remediation Schedule
                        </h3>
                        <Tag colour="bg-blue-400 text-black" isDark={isDark}>
                          {studySchedule.length} Weeks
                        </Tag>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {studySchedule.map((week) => (
                          <motion.div
                            key={week.week}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 + week.week * 0.07 }}
                            className={`p-4 border-2 ${
                              isDark
                                ? "border-gray-700 bg-[#111]"
                                : "border-black bg-[#FFFCF0]"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="bg-yellow-400 text-black font-black text-xs px-2 py-0.5 border border-black shrink-0">
                                WK {week.week}
                              </span>
                              <span className="font-black text-xs uppercase tracking-wide leading-tight truncate">
                                {week.focus}
                              </span>
                            </div>
                            <ul className="space-y-1">
                              {week.tasks.map((task, ti) => (
                                <li
                                  key={ti}
                                  className={`text-xs font-mono flex gap-1.5 ${
                                    isDark ? "text-gray-300" : "text-gray-600"
                                  }`}
                                >
                                  <span className="text-yellow-500 shrink-0">→</span>
                                  <span>{task}</span>
                                </li>
                              ))}
                            </ul>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <footer
        className={`mt-16 border-t-2 ${
          isDark ? "border-gray-800" : "border-black"
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between flex-wrap gap-2">
          <p
            className={`text-xs font-mono uppercase tracking-widest ${
              isDark ? "text-gray-500" : "text-gray-500"
            }`}
          >
            OBE Assessment System · Mamdani FIS · Llama 3.2 3B · Zero-Egress
          </p>
          <p
            className={`text-xs font-mono uppercase tracking-widest ${
              isDark ? "text-gray-500" : "text-gray-500"
            }`}
          >
            API: localhost:8000 · LLM: localhost:1234
          </p>
        </div>
      </footer>
    </div>
  );
}
