# CHANGELOG

## v14.1.0 — Computer Vision: YOLOv8 AI Gym pipeline

CV backbone rebuilt to mirror the Ultralytics "AI Gym" architecture, while
preserving every public API the rest of the application depends on.

### What changed (CV backend only — nothing else was touched)

**New modules**
- `backend/cv/yolo_pose.py` — YOLOv8-pose wrapper. Auto-downloads
  `yolov8n-pose.pt` on first use (≈ 6.5 MB), GPU/CPU auto-detect,
  warm-up pass to amortise JIT compile latency.
- `backend/cv/exercise_profiles.py` — single source of truth: per-exercise
  configuration (kpt triplets, up/down angle thresholds, mode, form-check
  rules). 15 exercises shipped, adding a new one = one entry.
- `backend/cv/ai_gym.py` — production AI Gym counter:
  bilateral angle averaging, visibility-gated frames, EMA-smoothed form
  scoring, frame annotator (server-side overlay of skeleton + reps + cues).

**Replaced (legacy public API preserved)**
- `backend/cv/pose_extractor.py` — now a facade that picks YOLOv8-pose
  primary, MediaPipe fallback. Same `extract(frame_bgr) -> (kp51, landmarks)`
  contract.
- `backend/cv/rep_counter.py` — wraps the new `AIGymCounter` but keeps the
  legacy `update(session_id, exercise_id, angles_10)` and `RepState` shape
  so existing tests + routes work unchanged.
- `backend/cv/pipeline.py` — uses YOLO + AI Gym. `FrameResult` keeps every
  legacy field and adds: `landmarks`, `stage`, `primary_angle`,
  `left_angle`, `right_angle`, `visible`, `backend`, plus frontend-aliased
  keys (`person_detected`, `pose_detected`, `rep_count`, `feedback_cues`).

**Routes (CV only)**
- `POST /vision/analyze` — now also accepts an `exercise_hint` form field
  (the frontend was already sending it). When present, the auto-classifier
  is bypassed and the AI Gym counter runs against that profile directly —
  much more accurate.
- `GET /vision/exercises` — new — returns the catalogue of supported
  exercises so the frontend stays in lock-step with the backend.
- `WS /vision/stream` — same auth, now accepts `?exercise_hint=…` on the
  handshake and `hint:<exercise_id>` text messages to retune mid-stream.

**Config**
- New env vars: `POSE_BACKEND` (yolo|mediapipe), `YOLO_MODEL_PATH`,
  `YOLO_CONF_THRESHOLD`, `YOLO_IMG_SIZE`. All have sensible defaults.
- `CV_DEVICE` now accepts `auto` (the default).

**Dependencies**
- Added: `ultralytics>=8.2.0`. MediaPipe stays in `requirements.txt` as the
  fallback path.

**Tests**
- New: `tests/test_cv_upgrade.py` — 8 integration tests covering the
  YOLO → AI Gym → FrameResult pipeline (squat reps, hold-mode timer,
  form-rule scoring, no-person, reset, frontend key contract, profile
  registry coverage, legacy `angles_10` back-compat).
- Existing `tests/test_smoke.py` — all 8 tests still pass; the legacy
  `test_rep_counter_squat_cycle` deliberately exercises the back-compat
  path through the new state machine.

**Verified working**
- 16/16 tests pass under `pytest`.
- `scripts/run_webcam.py` reads only `result.exercise_name`, `confidence`,
  `reps`, `form_score`, `fps`, `form_cues`, `keypoints` — every one of
  those is on the new `FrameResult`, so the script is unchanged.
- The frontend `Vision.jsx` reads `data.pose_detected`, `data.rep_count`,
  `data.form_score`, `data.feedback_cues`, `data.landmarks` — all present
  on the new payload.

### Things deliberately NOT changed
- Frontend (no edits to `frontend-react/`)
- Auth, database schema, config keys other than CV
- Chatbot, ML predict/recommend/insights routes
- The `exercise_classifier` MLP and its config / .pth — kept as the
  auto-detection fallback when no `exercise_hint` is provided
- `tests/test_smoke.py` — kept untouched as the back-compat regression suite

---

## v14.0.0 — Cookie auth, themes, Groq, notebook-based training

### Authentication: JWT → cookie sessions
- **Removed `python-jose` and the entire JWT layer.** The `RefreshToken` table,
  access/refresh token rotation, `/auth/refresh`, and `Authorization: Bearer`
  headers are all gone.
- **Added `itsdangerous`-signed session cookies** backed by a `Session` row in
  the database. Cookie is `HttpOnly`, 30-day max age, signed with
  `SESSION_SECRET`. Server-side revocation is instant (just delete the row).
- New: `set_session_cookie`, `clear_session_cookie`,
  `get_user_for_websocket` helpers in `backend/middleware/auth_guard.py`.
- The frontend axios client now uses `withCredentials: true` on every request
  and never touches localStorage for tokens.

**Why:** the old JWT setup had cascading 401 / refresh-race / "session
revoked mid-typing" failure modes. Cookie sessions are simpler and more stable
for a dashboard-style app.

### Themes (masculine / feminine)
- New CSS theme system in `frontend-react/src/index.css`. Two complete palettes
  toggled via `data-theme` on the `<html>` element.
- Masculine: dark athletic (lime/cyan/violet). Feminine: rose/lavender/peach.
- Theme follows the user's gender by default; sidebar has a manual toggle.
- Every component reads from CSS variables, so no JSX changes needed for
  themes to apply across the entire app.

### Chatbot: Groq added
- New `GroqLLM` class (`backend/chatbot/llm_provider.py`) using Llama 3.1 70B
  via `console.groq.com`.
- Groq is now the default `LLM_PROVIDER` and the first item in the fallback
  chain. OpenAI / Anthropic / Ollama still work as fallbacks.

### Training pipelines: scripts → Jupyter notebooks
- `notebooks/01_calorie_prediction_training.ipynb` — Ridge regression
- `notebooks/02_fitness_classification_training.ipynb` — Gradient Boosting
- `notebooks/03_workout_recommender_training.ipynb` — TF-IDF recommender
- `notebooks/04_pose_keypoint_classifier.ipynb` — PyTorch MLP
- Each notebook is self-contained: load → engineer → split → train → evaluate
  → save → smoke-test through the live service.
- The legacy `training/*.py` scripts still work; the notebooks are the
  canonical pipelines going forward.

### Configuration
- Removed: `JWT_SECRET`, `JWT_ALGO`, `JWT_ACCESS_MINUTES`, `JWT_REFRESH_DAYS`
- Added: `SESSION_SECRET`, `COOKIE_SECURE`, `COOKIE_SAMESITE`,
  `GROQ_API_KEY`, `GROQ_MODEL`
- `CORS_ORIGINS` default tightened to `localhost:5173` only — you cannot use
  `*` with credentialed cookies.
- New comprehensive `.env.example` with comments on every variable.

### Tests
- Rewrote `scripts/auth_smoke_test.py` and `scripts/self_test.py` for cookie
  auth.
- Added a real cookie-auth lifecycle test in `tests/test_smoke.py` that
  exercises signup → /me → logout → revoked-cookie rejection.

### Documentation
- New `README.md` covering architecture, quick start, env vars, training,
  testing, project structure, auth model, API surface, themes, and
  troubleshooting.

### Removed
- `python-jose[cryptography]` (no JWT)
- `chromadb` (only used by a fallback path that's no longer wired in)
- `redis`, `prometheus-client`, `langchain-google-genai`
  (not used anywhere in the active code path)
- The `RefreshToken` SQLAlchemy model and table
- Old changelog files (`V12_CHANGELOG.md`, `V13_CHANGELOG.md`, `FIXES.md`,
  `STARTUP.md`, `RESET-AND-RUN.ps1`)

### Verified working
- `npm run build` in `frontend-react/` → 2,797 modules, 0 errors
- Backend cookie-auth lifecycle test → all 7 assertions pass
- All four training notebooks → inline-tested for runnability
- Backend Python compiles (46 files, 0 syntax errors)
