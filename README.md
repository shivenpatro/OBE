# OBE Fuzzy Learning Assessment System

A comprehensive Outcome-Based Education (OBE) assessment platform that combines Mamdani Fuzzy Inference Systems with local privacy-preserving Large Language Models to provide data-driven, personalized student feedback.

## Overview

This system implements a four-phase architecture for student learning assessment:

1. **Phase 1: Environment & Dataset Setup** — Ingest and process the xAPI Educational Mining Dataset
2. **Phase 2: Core FIS MVP** — Mamdani Fuzzy Inference System for crisp academic scoring
3. **Phase 3: Local Agentic LLM** — Privacy-preserving natural language feedback via LM Studio
4. **Phase 4: Frontend Dashboard** — Neo-Brutalist faculty interface with real-time assessment visualization

### Key Features

- **Mamdani Fuzzy Inference System** — Triangular membership functions with 12 fuzzy rules for semantic reasoning over academic performance
- **Centroid Defuzzification** — Precise crisp output scores mapped to OBE classifications (Poor, Developing, Satisfactory, Good, Excellent)
- **Local LLM Integration** — Zero-egress architecture ensures all student data remains on the faculty member's machine
- **Multi-dimensional Input** — Continuous Assessment (30%), Lab Work (20%), Final Exam (50%), and Attendance
- **Weak Area Detection** — Deterministic rule-based identification of student knowledge gaps
- **Customized Study Schedules** — AI-generated 4-week remediation plans with weekly focus areas and actionable tasks
- **Neo-Brutalist Design** — Bold typography, thick borders, offset shadows, and responsive light/dark theming
- **Full-Stack Integration** — FastAPI backend with Next.js frontend, CORS-enabled for local development

---

## Architecture

### Backend Stack

- **Python 3.11+** with FastAPI and Uvicorn
- **scikit-fuzzy** — Fuzzy logic computations
- **pandas & numpy** — Data processing and numerical operations
- **LM Studio** (local, OpenAI-compatible API) — Privacy-preserving LLM inference
- **Pydantic** — Request/response validation

### Frontend Stack

- **Next.js 15+** (App Router)
- **React 19+** — Component-based UI
- **Tailwind CSS** — Utility-first styling with Neo-Brutalist tokens
- **Framer Motion** — Smooth animations and transitions
- **TypeScript** — Type-safe React components

### Data Flow

```
Faculty Input (4 scores)
    |
    v
[UI Bridge] — Aggregates CA/Lab/Final into assignment_score
    |
    v
[Fuzzy Engine] — Mamdani FIS (12 rules, centroid defuzzification)
    |
    v
[Weak Area Extraction] — Deterministic rule-based analysis
    |
    v
[LM Studio] — Local LLM generates breakdown & study schedule
    |
    v
[Dashboard] — Neo-Brutalist visualization
```

---

## Installation & Setup

### Prerequisites

- **Python 3.11** (3.14+ not supported due to pandas/scikit-fuzzy wheel availability)
- **Node.js 18+** and npm
- **LM Studio** running locally on `localhost:1234` (optional, system gracefully falls back if offline)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/CodingSagnik/outcome-based-education.git
cd outcome-based-education
```

### Step 2: Backend Setup

#### Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- pandas==2.2.3
- numpy==1.26.4
- scikit-fuzzy==0.4.2
- matplotlib==3.9.0
- scipy==1.14.0
- networkx==3.2
- requests==2.32.3
- fastapi==0.115.6
- uvicorn[standard]==0.32.1
- pydantic==2.10.3

#### Download Dataset

The system uses the xAPI Educational Mining Dataset (StudentPerformance.csv). Download it using one of the provided methods:

```bash
python download_dataset.py
```

This script attempts:
1. Kaggle CLI (if authenticated)
2. Direct HTTP from UCI Machine Learning Repository
3. Manual download instructions as fallback

Once downloaded, the dataset is automatically loaded and filtered to 480-650 optimal records in `working_set.csv`.

#### Verify Phase 1-2 Pipeline

```bash
python run_pipeline.py
```

This orchestrates:
- Data loading and feature bridging
- Batch FIS inference across all records
- Membership function visualization (`membership_functions.png`)
- Statistical analysis and performance distribution

### Step 3: LM Studio Setup (Optional but Recommended)

For local privacy-preserving LLM feedback:

1. **Download LM Studio** from https://lmstudio.ai/
2. **Load a model** (recommended: Llama 3.2 3B for balance of speed and quality)
3. **Start the local API server** (typically listens on `localhost:1234`)

Verify connectivity:
```bash
curl http://localhost:1234/v1/models
```

Expected output: JSON list of loaded models.

**Note:** If LM Studio is offline, the system gracefully falls back to rule-based feedback generation.

### Step 4: Start Backend API Server

```bash
uvicorn api_server:app --reload --port 8000
```

The server will:
- Initialize the Fuzzy Assessment Engine
- Connect to LM Studio (or note if unavailable)
- Expose `/api/assess` (POST), `/api/health` (GET), and `/api/sample` (GET)
- Display interactive docs at `http://localhost:8000/docs`

### Step 5: Frontend Setup

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will launch at `http://localhost:3000` and automatically proxy API calls to `localhost:8000`.

---

## Usage

### Web Interface

1. Navigate to `http://localhost:3000`
2. Enter four scores:
   - Continuous Assessment (0-100)
   - Lab Work (0-100)
   - Final Exam (0-100)
   - Attendance Percentage (0-100)
3. Click "Run FIS + LLM"
4. View results:
   - Crisp attainment score with dynamic gauge
   - OBE classification label
   - Identified weak areas
   - AI deficiency analysis
   - Customized 4-week study schedule

### API Endpoint

#### Health Check

```bash
curl http://localhost:8000/api/health
```

#### Assessment Request

```bash
curl -X POST http://localhost:8000/api/assess \
  -H "Content-Type: application/json" \
  -d '{
    "continuous_assessment": 75,
    "lab_work": 80,
    "final_exam": 70,
    "attendance": 85
  }' | python3 -m json.tool
```

#### Sample Response

```json
{
  "continuous_assessment": 75,
  "lab_work": 80,
  "final_exam": 70,
  "attendance": 85,
  "fis": {
    "assignment_score": 74.0,
    "attendance": 85.0,
    "crisp_attainment": 79.5,
    "label": "Good",
    "fired_rules": [
      "IF assignment=High AND attendance=Good THEN attainment=Good",
      "IF assignment=High AND attendance=Excellent THEN attainment=Excellent"
    ]
  },
  "weak_areas": [],
  "breakdown": "The student demonstrates strong overall performance with high assignment scores and excellent attendance...",
  "study_schedule": [
    {
      "week": 1,
      "focus": "Advanced topics review",
      "tasks": ["Review lecture notes on topic X", "Complete practice problems"]
    }
  ],
  "llm_available": true,
  "latency_ms": 2450,
  "model_used": "Llama 3.2 3B",
  "pipeline_ms": 2680
}
```

---

## Technical Details

### Phase 1: Data Processing

The system processes the xAPI Educational Mining Dataset with the following transformations:

**Input Features:**
- `raised_hands` — Count of times student raised hand in class
- `announcements_view` — Number of announcements viewed
- `discussion` — Participation in discussion forums
- `absence_days` — Days absent from class
- `visited_resources` — Count of learning resources accessed

**Derived Features:**
- `assignment_score = 0.35 * normalized(raised_hands) + 0.30 * normalized(announcements_view) + 0.35 * normalized(discussion)`
- `attendance = 100 - (absence_days / total_days * 100) * 0.6 + (visited_resources / max_resources * 100) * 0.4`

**Dataset Size Justification:**
The working set is capped at 480-650 records to:
- Ensure computational tractability for FIS rule firing analysis
- Maintain stratified representation across performance classes
- Enable rapid iteration during dashboard development
- Avoid unnecessary memory overhead without sacrificing statistical validity

### Phase 2: Fuzzy Inference System

**Linguistic Variables:**
- `assignment_score` (0-100): Poor, Average, Excellent
- `attendance` (0-100): Poor, Good, Excellent
- `attainment` (0-100): Poor, Developing, Satisfactory, Good, Excellent

**Membership Functions:**
Triangular membership functions (trimf) for smooth linguistic transitions:
```
Poor:        tri(0, 0, 30)
Average:     tri(20, 50, 80)
Excellent:   tri(70, 100, 100)
```

**Fuzzy Rule Base (12 rules):**
Examples:
- IF assignment=Poor AND attendance=Poor THEN attainment=Poor
- IF assignment=Average AND attendance=Good THEN attainment=Satisfactory
- IF assignment=Excellent AND attendance=Excellent THEN attainment=Excellent

**Defuzzification:**
Centroid method returns crisp output: `sum(mu * x) / sum(mu)` where mu is membership degree and x is the membership function value.

### Phase 3: Privacy-Preserving LLM

**Zero-Egress Guarantee:**
All processing occurs locally:
- Student data never leaves the faculty member's machine
- LM Studio runs on local GPU (RTX 3050 recommended for sub-2s latency)
- No external API calls, no cloud dependencies
- Llama 3.2 3B (8GB model) or equivalent runs entirely offline

**Weak Area Extraction:**
Deterministic rule-based identification:
```python
if assignment_score < 40:
    flag "Fundamental concepts mastery"
if attendance < 70:
    flag "Class engagement and participation"
```

**LLM Prompt:**
The system passes weak areas and FIS scores to the LLM, requesting:
1. Detailed breakdown of deficiencies with specific topics
2. Structured 4-week remediation schedule with weekly focus areas and actionable tasks

**Fallback Mechanism:**
If LM Studio is unavailable, the system automatically generates rule-based feedback using templates, ensuring graceful degradation.

### Phase 4: Frontend Architecture

**Neo-Brutalist Design System:**
- **Colors:** Monochrome with yellow accents (#FBBF24)
- **Typography:** Monospace fonts (Geist Mono) with high contrast
- **Styling:** Thick borders (2px), hard offset shadows (4px/4px)
- **Layout:** Grid-based with clear information hierarchy
- **Theming:** Light mode (#FFFCF0 bg) and dark mode (#111111 bg) with automatic contrast adjustment

**Component Structure:**
- `Gauge` — SVG semi-circular gauge with animated fill
- `ScoreInput` — Numeric input with weight hints
- `StatPill` — Info card with label and value
- `Tag` — Colored badge for status indicators
- Main `Dashboard` — Orchestrates form, results, and error states

**State Management:**
Form inputs are managed with React hooks; API responses update result state with Framer Motion animations for smooth transitions.

---

## File Structure

```
outcome-based-education/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data_loader.py                     # Dataset ingestion & validation
├── download_dataset.py                # Automated dataset download
├── feature_bridge.py                  # xAPI → FIS input transformation
├── ui_bridge.py                       # UI inputs → FIS input mapping
├── fuzzy_engine.py                    # Mamdani FIS implementation
├── agentic_feedback.py                # LM Studio integration & feedback
├── api_server.py                      # FastAPI backend
├── run_pipeline.py                    # Phase 1-2 orchestration & analysis
├── membership_functions.png           # Generated FIS visualization
├── data/
│   └── working_set.csv               # Processed dataset (480-650 rows)
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── next.config.ts                # API proxy configuration
│   ├── tsconfig.json
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx            # Root layout & metadata
│   │   │   ├── page.tsx              # Main page
│   │   │   └── globals.css           # Neo-Brutalist design tokens
│   │   └── components/
│   │       └── Dashboard.tsx          # Main assessment UI
│   └── node_modules/                 # Frontend dependencies
└── .gitignore
```

---

## Troubleshooting

### "venv/bin/activate: No such file or directory"
Create the virtual environment first:
```bash
python3.11 -m venv venv
```

### "subprocess-exited-with-error" during pip install
Ensure Python 3.11 is in use:
```bash
python3.11 --version
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### LM Studio connection errors
1. Verify LM Studio is running: `curl http://localhost:1234/v1/models`
2. Check that the model is loaded in LM Studio UI
3. Inspect backend logs for detailed error messages
4. The system will use rule-based fallback if LM Studio is offline

### Frontend permission denied on `npm run dev`
Clean and reinstall dependencies:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Turbopack build errors
Always use `npm run dev` (not `npx next dev`) to ensure the project's webpack-based Next.js is used, not a globally cached version.

---

## Performance Characteristics

- **FIS Inference (single):** ~5-10ms
- **LLM Latency (Llama 3.2 3B on RTX 3050):** 1.5-3 seconds
- **Total Pipeline (assessment + feedback):** 2-4 seconds
- **Batch Processing (650 records):** ~15-20 seconds
- **Memory Usage:** ~2GB (Python) + 8GB (LLM model) = ~10GB total

---

## Design Rationale

### Why Fuzzy Logic?

Traditional binary classification ("pass/fail") loses nuance in academic assessment. Fuzzy logic models the semantic continuum of linguistic terms ("Average," "Good," "Excellent") and naturally handles overlapping boundaries. This aligns with OBE principles, which emphasize meaningful learning outcomes rather than hard cutoffs.

### Why Mamdani?

Mamdani's rule-based approach provides interpretability — educators can inspect the fired fuzzy rules and understand exactly how the system arrived at a conclusion. This transparency is critical for academic credibility and student fairness.

### Why Local LLM?

Privacy. Student assessment data is sensitive. Sending it to cloud APIs (OpenAI, Anthropic) violates data sovereignty and introduces compliance risks. A local LLM on the faculty member's machine ensures zero data egress while maintaining state-of-the-art feedback quality.

### Why Neo-Brutalism?

The heavy, bold aesthetic conveys precision and seriousness — appropriate for academic assessment. It also ensures accessibility through high contrast and readable typography, with no reliance on subtle color gradients or light borders.

---

## Future Enhancements

- Multi-student batch assessment UI with CSV export
- Real-time performance analytics dashboard for cohort trends
- Customizable fuzzy membership functions via UI
- Support for multiple LLM backends (Ollama, llama.cpp)
- Persistent result history with trend analysis
- Role-based access control for departments and administrators
- Integration with institutional LMS systems (Canvas, Moodle)
- Advanced weak area prediction using historical performance patterns

---

## Citation

If you use this system in academic research, please cite:

```
@software{obe_fuzzy_assessment_2024,
  author = {Sagnik},
  title = {OBE Fuzzy Learning Assessment System},
  year = {2024},
  url = {https://github.com/CodingSagnik/outcome-based-education}
}
```

---

## License

This project is provided as-is for educational and research purposes. See LICENSE file for details.

---

## Support & Contribution

For issues, feature requests, or contributions, please open an issue or pull request on GitHub.

**Maintainer:** Sagnik (Computer Science & Engineering Student)

---

## Acknowledgments

- xAPI Educational Mining Dataset (UCI Machine Learning Repository)
- scikit-fuzzy community for Mamdani FIS implementations
- LM Studio for local LLM serving infrastructure
- Outcome-Based Education framework and best practices
