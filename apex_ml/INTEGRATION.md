# apex_ml integration into APEX AI v14

This document describes the integration that has already been applied to
this project. It is the complete record of every change made.

## Summary of changes

| File | Status | Lines changed |
|------|--------|---------------|
| `apex_ml/`                                      | NEW package         | (drop-in)   |
| `apex_ml/utils/coco_adapter.py`                 | NEW (project-fit)   | 130         |
| `apex_ml/integrations/apex_ai_bridge.py`        | NEW (project-fit)   | 195         |
| `backend/routes/ml_coaching.py`                 | NEW                 | 200         |
| `backend/main.py`                               | MODIFIED (additive) | +9 lines    |
| `backend/cv/pipeline.py`                        | MODIFIED (additive) | +20 lines   |
| `backend/routes/vision.py`                      | MODIFIED (additive) | +6 lines    |

Total touched lines in existing files: **35**, all additive. Zero
removed, zero replaced. Every addition is inside a `try/except` block
or behind a default-`None` dataclass field so that any failure in
apex_ml is silently ignored and the existing pipeline runs unchanged.

## What was preserved

- Every existing route, including `/auth`, `/chat`, `/vision/*`,
  `/predict/*`, `/recommend`, `/data`, `/insights`, `/user-data`.
- The cookie-session auth flow and `get_current_user` dependency.
- The YOLOv8-pose + MediaPipe fallback pose extractor.
- The existing `ExerciseClassifier` + `RepCounter` (AI Gym counter).
- The existing rep counting via `update_keypoints`.
- The `CVAnalysis`, `WorkoutLog`, `RecommendationLog`, `UserProfile`
  schema — apex_ml only **reads** from `UserProfile` and `WorkoutLog`,
  and writes adaptive workouts to the existing `RecommendationLog`
  with `rec_type="ml_workout"`.
- The WebSocket protocol `/vision/stream` — new payload fields are
  added; no field is changed or removed.
- All chatbot, RAG, and chat-streaming logic.
- `requirements.txt`, `.env`, all configs.

## What was added

### 1. Temporal pose analysis layer
A per-session ring-buffer of YOLOv8-pose keypoints, smoothed with a
One-Euro filter, drives a finite-state machine that classifies frames
into start/concentric/eccentric/lockout/reset and counts reps from
state transitions. The COCO-17 keypoints emitted by your existing
pose extractor are adapted to MediaPipe-33 internally — your CV code
sees no change.

This **coexists** with the existing `RepCounter`. Both run in parallel;
the existing `reps` and `phase` fields in `FrameResult` come from the
existing counter, and a new sibling field `temporal` carries the
apex_ml output. The frontend can use either or both.

### 2. Real-time form correction
Exercise-specific biomechanical rules (squat / pushup / plank / lunge /
bicep_curl) emit human-readable corrections during each frame:
"Increase squat depth", "Knees are collapsing inward", etc. Each rule
is debounced so the UI never gets flooded. On rep completion the
engine computes a four-component quality score (form / depth /
stability / tempo) and attaches it to the next frame's `temporal`
payload as `last_rep_quality`.

### 3. Adaptive recommendation
Three new authenticated routes mounted under `/ml/*`:

```
GET  /ml/health        liveness probe + supported exercises
GET  /ml/workout       generate today's adaptive workout
GET  /ml/coaching      AI coach: deload / progression / substitution
GET  /ml/recovery      readiness / fatigue / overtraining risk
POST /ml/session/{id}/end   close a temporal-pose session
```

These reuse the existing `get_current_user` dependency, read from the
existing `UserProfile` and `WorkoutLog` tables, and persist adaptive
workouts to `RecommendationLog` with `rec_type="ml_workout"` so they
show up in the existing `/recommend/history` endpoint automatically.

## New JSON fields the frontend may opportunistically render

Inside every `FrameResult` from `/vision/analyze` and `/vision/stream`:

```jsonc
{
  // … all existing fields unchanged …

  "temporal": {
    "version": 1,
    "exercise": "squat",
    "phase": "concentric",         // start|concentric|eccentric|lockout|reset
    "rep_count": 7,                // apex_ml reps (separate from "reps")
    "partial_rep_count": 2,        // shallow / incomplete reps detected
    "feedback": [
      { "rule": "squat_shallow", "message": "Increase squat depth.",
        "severity": 2, "body_part": "hips" }
    ],
    "indicators": {
      "spine_aligned": true,
      "knees_tracking": false,
      "elbows_aligned": true,
      "hips_stable": true
    },
    "path": [[0.50, 0.62], …],     // last 20 COM points for trail rendering
    "last_rep_quality": {          // present only on the frame after a rep
      "form": 85, "depth": 78, "stability": 90, "tempo": 82, "overall": 84
    }
  }
}
```

Old clients ignore the `temporal` field entirely — fully backward
compatible. The frontend can adopt parts of it progressively.

## How to verify

1. **Install the apex_ml requirements** (most should already be present):

   ```bash
   pip install numpy 'pydantic>=2'
   # Already in requirements.txt: fastapi, torch
   ```

2. **Start the server normally**:

   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

   You should see all existing startup logs plus no `apex_ml routes
   not mounted` warning.

3. **Liveness probe**:

   ```bash
   curl http://localhost:8000/ml/health
   # → {"status":"ok","layer":"apex_ml",
   #    "supported_exercises":["squat","pushup","plank","lunge","bicep_curl"]}
   ```

4. **Workout generation** (auth required — log in first via your
   normal flow, then):

   ```bash
   curl --cookie cookies.txt http://localhost:8000/ml/workout
   ```

   Returns the adaptive workout, also persisted to `recommendation_logs`
   with `rec_type="ml_workout"`.

5. **Live CV with temporal output**:

   Use the existing `/vision/stream` WebSocket exactly as before, but
   pass `?exercise_hint=squat` (or pushup / plank / lunge / bicep_curl).
   The frame results will include the new `temporal` block.

## Files at a glance

```
apex_ai/
├── apex_ml/                    ← NEW: the entire ML/AI layer
│   ├── README.md
│   ├── INTEGRATION.md          ← this file
│   ├── temporal_pose/          sequence buffer + state machine
│   ├── sequence_models/        LSTM / TCN / Transformer encoders
│   ├── motion_analysis/        velocity, jerk, symmetry, …
│   ├── form_correction/        rules, engine, overlay
│   ├── recommendation/         profile, fatigue, ranking, …
│   ├── integrations/
│   │   └── apex_ai_bridge.py   ← project-specific bridge (COCO-17 ↔ 33)
│   ├── utils/
│   │   └── coco_adapter.py     ← project-specific adapter
│   ├── api/                    (unused — we use ml_coaching.py instead)
│   └── tests/
│
├── backend/
│   ├── main.py                 ← +9 lines (mount ml_coaching router)
│   ├── cv/
│   │   └── pipeline.py         ← +20 lines (append temporal block)
│   └── routes/
│       ├── ml_coaching.py      ← NEW: /ml/workout, /ml/coaching, /ml/recovery
│       └── vision.py           ← +6 lines (reset temporal state too)
│
└── (everything else: unchanged)
```

## Rollback procedure

If anything goes wrong, revert by removing the additions:

1. Delete `apex_ml/` and `backend/routes/ml_coaching.py`.
2. In `backend/main.py`, delete the 9 lines under
   `# ── apex_ml additive ML layer …`.
3. In `backend/cv/pipeline.py`, delete the `temporal: Optional[dict] =
   None` dataclass field and the `temporal_block = None / try / except`
   block plus `temporal=temporal_block,` in the constructor call.
4. In `backend/routes/vision.py`, delete the 6 lines inside `/reset`
   under `# Additive: also clear apex_ml temporal state`.

After rollback, the server runs exactly as it did before integration.
