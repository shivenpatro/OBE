# OBE Fuzzy Learning Assessment System

A high-performance Outcome-Based Education (OBE) platform that bridges **Mamdani Fuzzy Inference Systems** (FIS) with **Agentic AI** to provide data-driven, privacy-preserving student attainment analysis.

---

## 🚀 Key Features

- **Hybrid Intelligence Architecture**: Combines deterministic fuzzy logic for scoring with non-deterministic LLMs for natural language feedback.
- **Zero-Egress Privacy**: Designed to work with local LLMs via **LM Studio**, ensuring sensitive student data never leaves your machine.
- **Graceful Degradation**: The system is fully functional without an LLM—it algorithmically extracts weak areas and provides rule-based remediation plans when offline.
- **Neo-Brutalist Interface**: A bold, high-contrast faculty dashboard designed for clarity and rapid data entry.
- **Mamdani FIS Engine**: Implements a 12-rule inference system using triangular membership functions and centroid defuzzification.

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Asynchronous Python API framework.
- **scikit-fuzzy**: Advanced fuzzy logic toolkit.
- **pandas/numpy**: High-performance data processing.
- **Uvicorn**: ASGI server for deployment.

### Frontend
- **Next.js 15 (App Router)**: Modern React framework.
- **Tailwind CSS**: Utility-first styling with custom Neo-Brutalist tokens.
- **Framer Motion**: Smooth UI transitions and micro-interactions.
- **Lucide React**: Scalable vector icons.

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/shivenpatro/OBE.git
cd OBE
```

### 2. Backend Environment
- **Prerequisite**: Python 3.10 or 3.11 (Recommended).
- **Setup**:
```bash
# Create a virtual environment
python -m venv venv
# Activate it (Windows)
.\venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Environment
- **Prerequisite**: Node.js 18+ and npm.
- **Setup**:
```bash
cd frontend
npm install
```

---

## 🧠 Using with Local LLM (Optional but Recommended)

To enable **AI-Powered Agentic Feedback**, you must run a local LLM server using **LM Studio**.

1. **Download LM Studio**: [lmstudio.ai](https://lmstudio.ai/)
2. **Load a Model**: We recommend `Llama 3 (8B)` or `Mistral (7B)`.
3. **Start Local Server**:
   - Go to the **Local Server** tab in LM Studio.
   - Select your model and click **Start Server**.
   - Ensure the port is set to `1234`.
4. **Verification**: The system will automatically detect the server and replace generic fallback text with personalized, multi-paragraph student feedback.

---

## 🏃 Running the Project

### Start the Backend
From the project root:
```bash
python api_server.py
```
*The API will be available at `http://localhost:8000`.*

### Start the Frontend
In a separate terminal (inside the `frontend` folder):
```bash
npm run dev
```
*The dashboard will be available at `http://localhost:3000`.*

---

## 📁 Project Structure

- `api_server.py`: FastAPI entry point and pipeline orchestration.
- `fuzzy_engine.py`: Core Mamdani FIS implementation.
- `agentic_feedback.py`: Logic for LLM integration and fallback mechanisms.
- `ui_bridge.py`: Maps UI percentages to fuzzy engine antecedents.
- `frontend/`: Next.js source code and Neo-Brutalist components.
- `requirements.txt`: Python dependency manifest.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed by Shiven Patro**
[GitHub Profile](https://github.com/shivenpatro)
