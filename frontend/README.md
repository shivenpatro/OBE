# OBE Frontend Dashboard

The user-facing component of the OBE Fuzzy Learning Assessment System. Built with **Next.js 15** and **Tailwind CSS**, featuring a bold **Neo-Brutalist** design language.

## ✨ Features
- **Real-time Assessment**: Instant feedback from the FastAPI backend.
- **Dynamic Visualization**: Attainment scores visualized via a custom gauge and progress indicators.
- **AI-Powered Insights**: Detailed student feedback and 6-week study plans (when connected to LM Studio).
- **Responsive Design**: Fully optimized for both desktop and mobile viewing.
- **Dark Mode Support**: Seamlessly toggles between light and dark themes.

## 🛠️ Tech Stack
- **Framework**: [Next.js 15](https://nextjs.org/) (App Router)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Animations**: [Framer Motion](https://www.framer.com/motion/)
- **Type Safety**: TypeScript

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the dashboard.

### 3. Connection to Backend
Ensure the FastAPI server is running at `http://localhost:8000`. The frontend communicates with the backend via the `/api/assess` endpoint.

## 📁 Structure
- `src/app`: App Router pages and global styles.
- `src/components`: Reusable UI components (Dashboard, Gauge, etc.).
- `public`: Static assets (SVG icons, etc.).

---
Part of the [OBE Fuzzy Learning Assessment System](https://github.com/shivenpatro/OBE)
