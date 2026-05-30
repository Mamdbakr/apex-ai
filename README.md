# APEX AI v14 — AI Personal Trainer Platform

An AI-powered fitness platform with **real ML predictions, real RAG-powered chatbot,
real computer vision form analysis, and a stable cookie-based auth system**.

> **What changed in v14**
> - **Auth rewritten from JWT → cookie-based sessions** (no more "token expired" crashes)
> - **Groq + Llama 3.1** added as the default LLM (fastest free option)
> - **Theme system** — masculine (athletic) / feminine (rose-violet) palettes that follow user gender
> - **Training pipelines converted to Jupyter notebooks** — see `notebooks/`
> - Cleaner config, working end-to-end auth tests

---

## Table of contents
1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [Environment variables](#environment-variables)
4. [Running the backend](#running-the-backend)
5. [Running the frontend](#running-the-frontend)
6. [Training the AI models](#training-the-ai-models)
7. [Testing](#testing)
8. [Project structure](#project-structure)
9. [Auth model](#auth-model)
10. [API surface](#api-surface)
11. [Themes](#themes)
12. [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite + Tailwind + Framer Motion)                  │
│  · Cookie-only auth (HttpOnly, signed)                                │
│  · Theme switches by user gender (masculine / feminine)               │
│  · Pages: Dashboard, AI Coach, Predictions, CV Trainer, Progress      │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │ withCredentials: true
                                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI + SQLAlchemy async + SQLite/Postgres)               │
│                                                                       │
│  /auth/*          cookie sessions (itsdangerous + bcrypt)             │
│  /chat            RAG chatbot — Groq Llama 3.1 + FAISS + LangGraph    │
│  /vision/*        MediaPipe pose extraction + PyTorch form classifier │
│  /predict/*       scikit-learn calorie / weight / fitness models      │
│  /recommend       TF-IDF content-based exercise recommender           │
│  /insights/*      AI dashboard payload (forecasts, anomalies, cohort) │
│  /user-data/*     profile, workouts, weights, nutrition CRUD          │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
   ai_models/ml_models/         ai_models/dl_models/          knowledge_base/
   • calorie_regression.pkl     • exercise_classifier.pth     • FAISS index
   • weight_regression.pkl      • cv_keypoint_scaler.pkl      • exercise guides
   • fitness_classifier.pkl     • exercise_classifier_config  • nutrition data
   • recommender.pkl                                          • ISSN papers
```

Every artifact under `ai_models/` and `knowledge_base/faiss_index/` is produced
by code in this repo — none of it is mocked. The notebooks under `notebooks/`
are the authoritative training pipelines.

---

## Quick start

```bash
# 1. Clone and enter the project
cd apex-ai

# 2. Backend setup
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY (free at https://console.groq.com/keys)

# 4. (Optional) train fresh models. The repo ships with pre-trained models
# so you can skip this for a first run.
jupyter notebook notebooks/

# 5. Start the backend
uvicorn backend.main:app --reload --port 8000

# 6. In a separate terminal, start the frontend
cd frontend-react
npm install
npm run dev
```

Visit **http://localhost:5173**, click **Create one free**, fill in your stats,
and you're in.

---

## Environment variables

See `.env.example` for the full list with comments. The bare minimum to run
locally is just `SESSION_SECRET` (auto-defaulted to a dev value) — everything
else is optional. To make the chatbot speak to a real LLM, set `GROQ_API_KEY`.

**For production**, you must:
- Change `SESSION_SECRET` to a random 48-byte string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`)
- Set `COOKIE_SECURE=true`
- Pin `CORS_ORIGINS` to your frontend domain(s)
- Switch `DATABASE_URL` to Postgres

---

## Running the backend

```bash
# from the project root
uvicorn backend.main:app --reload --port 8000
```

You should see startup logs like:
```
🚀  APEX AI v14.0.0 starting · env=DEV
🔐  Auth: cookie-based sessions (no JWT)
🦙  Groq · model=llama-3.1-70b-versatile
✅  Database ready
✅  ExerciseNet loaded · 4 classes · cpu
✅  ML service ready · 3/3 models loaded · explainability=on
🌐  Listening on http://0.0.0.0:8000  ·  docs at /docs
```

Interactive docs: **http://localhost:8000/docs** (Swagger UI).

If `ExerciseNet not loaded`: open `notebooks/04_pose_keypoint_classifier.ipynb`
and run all cells. Same for any of the other "missing" warnings.

---

## Running the frontend

```bash
cd frontend-react
npm install
npm run dev          # http://localhost:5173
```

Production build:
```bash
npm run build        # output goes to dist/
npm run preview      # serves dist/ for verification
```

The frontend reads `VITE_API_URL` if set, otherwise falls back to
`http://localhost:8000`. To point at a remote backend:
```bash
VITE_API_URL=https://api.yourdomain.com npm run dev
```

---

## Training the AI models

The repo ships with pre-trained models, so you can skip this section unless you
want to retrain on new data. **Everything trains in Jupyter notebooks now**:

| Notebook | Output | What it does |
|---|---|---|
| `notebooks/01_calorie_prediction_training.ipynb` | `ai_models/ml_models/calorie_regression.pkl` | Ridge regression on engineered features → daily calorie target |
| `notebooks/02_fitness_classification_training.ipynb` | `ai_models/ml_models/fitness_classifier.pkl` | Gradient-boosted 3-class classifier (Beginner/Intermediate/Advanced) |
| `notebooks/03_workout_recommender_training.ipynb` | `ai_models/ml_models/recommender.pkl` | TF-IDF + popularity-weighted exercise ranking per fitness level |
| `notebooks/04_pose_keypoint_classifier.ipynb` | `ai_models/dl_models/exercise_classifier.pth` | PyTorch MLP that maps MediaPipe keypoints → exercise label |

Run them with:
```bash
jupyter notebook notebooks/
```
Each notebook is self-contained — load → engineer → split → train → evaluate
→ save → smoke-test through the live service. Cells include matplotlib charts,
metrics tables, and a final round-trip test that reloads the model the same
way the running backend will.

The legacy `training/*.py` CLI scripts are kept for backwards compatibility
but the notebooks are the canonical pipelines.

---

## Testing

End-to-end auth lifecycle test:
```bash
pytest tests/test_smoke.py -v
```

Manual smoke test:
```bash
python scripts/auth_smoke_test.py
```

Self-test of all major subsystems (DB, ML, CV, RAG):
```bash
python scripts/self_test.py
```

---

## Project structure

```
apex-ai/
├── backend/
│   ├── core/               config + structured logging
│   ├── database/           SQLAlchemy models + async engine
│   ├── middleware/         cookie-based auth guard, password hashing
│   ├── routes/             FastAPI routers (auth, chat, vision, predict…)
│   ├── services/           ML, chatbot, insight, cohort, anomaly services
│   ├── chatbot/            v9 fallback LLM + vector store + memory
│   ├── rag_coach/          LangGraph-based primary chatbot engine
│   ├── cv/                 MediaPipe pose extractor + rep counter + classifier
│   ├── data_pipeline/      ingest events → DB + feature vectors
│   └── main.py             FastAPI entrypoint
├── frontend-react/
│   ├── src/
│   │   ├── pages/          Landing, Login, Register, Dashboard, Chatbot…
│   │   ├── components/     Layout (sidebar + theme toggle)
│   │   ├── lib/api.js      cookie-based HTTP client
│   │   ├── store/useStore  zustand store + theme system
│   │   ├── App.jsx         routes + AuthBootstrap + ThemeEffect
│   │   └── index.css       design tokens + masculine + feminine themes
│   ├── tailwind.config.js
│   └── vite.config.js
├── notebooks/              ★ AI training pipelines (Jupyter)
│   ├── 01_calorie_prediction_training.ipynb
│   ├── 02_fitness_classification_training.ipynb
│   ├── 03_workout_recommender_training.ipynb
│   └── 04_pose_keypoint_classifier.ipynb
├── ai_models/
│   ├── ml_models/          scikit-learn .pkl files + per-model reports/
│   ├── dl_models/          PyTorch .pth + scaler + config
│   └── vector_store/       Chroma fallback for v9 chatbot
├── datasets/               training data (CSV) + generator
├── knowledge_base/
│   ├── raw/                exercise & nutrition source docs
│   └── faiss_index/        FAISS index for RAG
├── training/               legacy CLI training scripts (kept for compatibility)
├── scripts/                seed/maintenance scripts (auth_smoke_test, self_test…)
├── tests/                  pytest suite
├── requirements.txt
├── .env.example
└── README.md               (this file)
```

---

## Auth model

Cookie-based sessions, **no JWT anywhere**.

1. **Signup / signin** → backend creates a `Session` row in the DB and returns
   a `Set-Cookie: apex_session=<signed sid>` header.
2. The cookie is `HttpOnly`, signed with `SESSION_SECRET` via `itsdangerous`,
   and lasts 30 days.
3. Every subsequent API call goes out with `withCredentials: true`. The browser
   attaches the cookie automatically.
4. The backend's auth dependency reads the cookie, verifies the signature,
   looks up the row, returns the user.
5. **Logout** revokes the row server-side AND clears the cookie. Stolen cookies
   stop working instantly.

Why this is more stable than JWT:
- No expiry race conditions in the middle of a session
- Server-side revocation is instant
- No client storage of secrets — JS cannot read `HttpOnly` cookies
- Works without any `Authorization` header plumbing

---

## API surface

Auto-generated docs at `/docs`. Highlights:

```
POST   /auth/signup           create account (sets cookie)
POST   /auth/signin           login          (sets cookie)
POST   /auth/logout           revoke this session, clear cookie
POST   /auth/logout-all       revoke every session for the user
GET    /auth/me               current user + profile (cookie-protected)
GET    /auth/sessions         list active sessions for the current user
DELETE /auth/sessions/{id}    revoke a specific session

POST   /chat                  RAG chatbot — fitness/nutrition/coaching
POST   /chat/stream           SSE streaming version
GET    /chat/history/{uid}    last 50 messages
DELETE /chat/history/{uid}    clear chat history

POST   /vision/analyze        single-image pose + form analysis
POST   /vision/reset          reset rep counter for a session
POST   /vision/session/finish close session → write workout log
GET    /vision/history        recent CV analyses
WS     /vision/stream         real-time pose stream

POST   /predict/all           calorie + weight-change + fitness in one shot
POST   /predict/calories      calorie prediction
POST   /predict/weight-change 30-day projected weight delta
POST   /predict/fitness-level beginner / intermediate / advanced
POST   /predict/explain       feature attributions for any prediction

GET    /recommend             top-K personalised exercises
GET    /recommend/history     previous recommendations served

GET    /insights/dashboard    AI dashboard payload (forecast + anomalies + cohort)
GET    /insights/forecast     N-day weight + TDEE forecast
GET    /insights/anomalies    detected pattern breaks
POST   /insights/refresh      recompute the user's feature vector

GET    /user-data/profile     read the user's profile
POST   /user-data/profile     update it
POST   /user-data/workout     log a completed workout
GET    /user-data/workouts    list workouts
GET    /user-data/dashboard   compact dashboard data

GET    /health                deep health check (auth, ML, CV, chatbot status)
```

---

## Themes

The frontend has two themes that swap purely via CSS variables — every component
reads from tokens like `var(--accent)`, `var(--grad-accent)`, `var(--bg-primary)`,
so no JSX changes when the theme switches.

| | Masculine (`m`) | Feminine (`f`) |
|---|---|---|
| Accent | Lime green `#00ff88` | Rose `#ff6fae` |
| Secondary | Cyan `#00d4ff` | Lavender `#c084fc` |
| Tertiary | Violet `#7b5cff` | Peach `#ffb47a` |
| Mood | dark athletic / neon | warm elegant / soft glow |

Theme follows the user's gender by default. They can override it via the
sidebar toggle (third click resets to gender default). The choice is persisted
in localStorage. The `<html data-theme="…">` attribute is what CSS keys off.

---

## Troubleshooting

**"Not authenticated" on every request after signin**
- Make sure the frontend `axios` instance has `withCredentials: true` (it does
  by default in v14).
- Make sure `CORS_ORIGINS` includes your frontend URL exactly. **You cannot
  use `*` with credentialed cookies** — the browser will refuse to send them.
- In production over HTTPS, set `COOKIE_SECURE=true` and `COOKIE_SAMESITE=none`.

**"ExerciseNet not loaded" warning at startup**
- Run `notebooks/04_pose_keypoint_classifier.ipynb` to train it. The vision
  endpoints fall back gracefully (no crash, just no form classification).

**Chatbot returns generic answers**
- No LLM key is set. Add `GROQ_API_KEY=gsk_...` to `.env` (free at
  https://console.groq.com/keys) and restart.

**WebSocket auth fails on `/vision/stream`**
- Same-origin (running frontend behind a Vite proxy → backend): cookies are
  sent automatically and the WS handshake works.
- Cross-origin: browsers can't send HttpOnly cookies on WS upgrades, so live
  streaming requires same-origin deployment. Use `/vision/analyze` for
  single-frame analysis instead.

**Stale models after retraining**
- The `MLService` and `ExerciseClassifier` are cached singletons. Restart the
  uvicorn process to pick up new `.pkl` / `.pth` files.

---

## License

This project is for educational use. The bundled exercise descriptions and
fitness PDFs in `knowledge_base/raw/` are aggregated from public sources and
ISSN publications — see the individual files for attribution.
"# apex-ai" 
"# apex-ai" 
"# apex-ai" 
