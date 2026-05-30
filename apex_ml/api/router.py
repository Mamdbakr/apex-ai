"""
Optional FastAPI router exposing apex_ml endpoints.

WIRING (additive to existing app — pick the line you want):

    # In your existing FastAPI app file, ADD (do not remove anything):
    from apex_ml.api.router import router as apex_ml_router
    app.include_router(apex_ml_router, prefix="/ml")

That is the entire integration. All routes live under /ml so they
cannot collide with existing routes. Authentication is left to the host
app — protect this router with whatever Depends() your project already
uses.

The router intentionally does NOT touch any database. State is held in
small in-memory session objects keyed by `session_id`. Persistence is
the host app's responsibility — see `serialize_profile` for a helper.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    _FASTAPI_OK = True
except ImportError:                                  # pragma: no cover
    _FASTAPI_OK = False

from ..form_correction import FormCorrectionEngine, build_overlay
from ..recommendation import (
    AICoach, ExerciseRanker, PerformancePredictor,
    UserGoals, UserProfile, WorkoutGenerator,
    estimate_recovery,
)
from ..temporal_pose import EXERCISE_DEFAULTS, ExerciseStateMachine, SequenceBuffer


# ---------------------------------------------------------------- guard
if not _FASTAPI_OK:
    # The package still imports fine even without FastAPI; the router
    # simply won't be available. This lets non-web users of apex_ml
    # avoid a hard dependency.
    router = None
else:
    router = APIRouter(tags=["apex_ml"])

    # ============================================================ models
    class StartSessionRequest(BaseModel):
        exercise: str = Field(..., description="One of: squat|pushup|plank|lunge|bicep_curl")
        window_size: int = 60
        fps_hint: float = 30.0

    class StartSessionResponse(BaseModel):
        session_id: str
        exercise: str

    class FrameRequest(BaseModel):
        session_id: str
        timestamp: float = Field(..., description="Frame time in seconds")
        landmarks: List[List[float]] = Field(
            ..., description="33 x 3 array of MediaPipe pose landmarks (x,y,z)"
        )

    class FrameResponse(BaseModel):
        phase: str
        rep_count: int
        partial_rep_count: int
        feedback: List[Dict]
        indicators: Dict
        path: List[List[float]]
        last_rep_quality: Optional[Dict] = None

    class WorkoutRequest(BaseModel):
        user_id: str
        goal: str = "general_fitness"
        weekly_sessions: int = 3
        available_minutes: int = 45
        equipment: List[str] = Field(default_factory=lambda: ["bodyweight"])
        history: List[Dict] = Field(
            default_factory=list,
            description="Serialized WorkoutSession dicts (see UserProfile.to_dict)",
        )

    class CoachingRequest(BaseModel):
        user_id: str
        history: List[Dict] = Field(default_factory=list)
        goals: Optional[Dict] = None

    # ============================================================ state
    # In-memory active sessions. Host app should persist whatever it needs
    # to disk/DB; this is intentionally ephemeral.
    _SESSIONS: Dict[str, dict] = {}

    def _get_session(sid: str) -> dict:
        if sid not in _SESSIONS:
            raise HTTPException(status_code=404, detail="session not found")
        return _SESSIONS[sid]

    def _hydrate_profile(payload: dict) -> UserProfile:
        """Build a UserProfile from the request payload (history + goals)."""
        d = {
            "user_id": payload["user_id"],
            "goals": {
                "primary": payload.get("goal", "general_fitness"),
                "weekly_sessions": payload.get("weekly_sessions", 3),
                "available_minutes": payload.get("available_minutes", 45),
                "equipment": payload.get("equipment", ["bodyweight"]),
            },
            "sessions": payload.get("history", []),
        }
        return UserProfile.from_dict(d)

    # ============================================================ routes
    @router.get("/health")
    def health() -> dict:
        """Liveness probe — confirms apex_ml is mounted and importable."""
        return {"status": "ok", "exercises": list(EXERCISE_DEFAULTS.keys())}

    @router.post("/session/start", response_model=StartSessionResponse)
    def start_session(req: StartSessionRequest) -> StartSessionResponse:
        """Create a streaming pose-analysis session for one exercise."""
        if req.exercise not in EXERCISE_DEFAULTS:
            raise HTTPException(400, f"Unknown exercise: {req.exercise}")
        sid = str(uuid.uuid4())
        buf = SequenceBuffer(window_size=req.window_size,
                              sample_rate_hz=req.fps_hint)
        fsm = ExerciseStateMachine(req.exercise)
        # Map state-machine primary joint to FormCorrection's depth scorer
        engine = FormCorrectionEngine(
            req.exercise,
            primary_joint=EXERCISE_DEFAULTS[req.exercise].primary_joint,
        )
        _SESSIONS[sid] = {
            "exercise": req.exercise,
            "buf": buf,
            "fsm": fsm,
            "engine": engine,
            "last_quality": None,
            "rep_durations": [],
            "created_at": time.time(),
        }

        # Hook rep completion to compute quality once per rep
        def _on_rep(rep, _sid=sid):
            sess = _SESSIONS.get(_sid)
            if sess is None:
                return
            quality = sess["engine"].rep_quality(rep, sess["buf"])
            sess["last_quality"] = quality
            sess["rep_durations"].append(rep.end_t - rep.start_t)

        fsm.on_rep_completed = _on_rep
        return StartSessionResponse(session_id=sid, exercise=req.exercise)

    @router.post("/session/frame", response_model=FrameResponse)
    def push_frame(req: FrameRequest) -> FrameResponse:
        """Submit one frame of landmarks; receive phase/rep/feedback."""
        sess = _get_session(req.session_id)
        landmarks = np.array(req.landmarks, dtype=np.float64)
        if landmarks.shape != (33, 3):
            raise HTTPException(400,
                f"landmarks must be 33x3; got {landmarks.shape}")
        sess["buf"].push(landmarks, t=req.timestamp)
        phase = sess["fsm"].update(sess["buf"])
        fb = sess["engine"].step(sess["buf"])

        overlay = build_overlay(
            exercise=sess["exercise"],
            phase=phase,
            rep_count=sess["fsm"].rep_count,
            feedback=fb,
            buf=sess["buf"],
            quality=sess["last_quality"],
        )

        return FrameResponse(
            phase=overlay["phase"],
            rep_count=overlay["rep_count"],
            partial_rep_count=sess["fsm"].partial_count,
            feedback=overlay["feedback"],
            indicators=overlay["indicators"],
            path=overlay["path"],
            last_rep_quality=overlay.get("quality"),
        )

    @router.post("/session/{session_id}/end")
    def end_session(session_id: str) -> dict:
        """Tear down a streaming session and return its summary."""
        sess = _get_session(session_id)
        summary = sess["engine"].session_summary(
            sess["buf"], sess["rep_durations"]
        )
        summary.update({
            "rep_count": sess["fsm"].rep_count,
            "partial_rep_count": sess["fsm"].partial_count,
            "duration_seconds": time.time() - sess["created_at"],
        })
        del _SESSIONS[session_id]
        return summary

    @router.post("/workout/generate")
    def generate_workout(req: WorkoutRequest) -> dict:
        """Generate a personalized adaptive workout for a user."""
        profile = _hydrate_profile(req.model_dump())
        gen = WorkoutGenerator()
        return gen.generate(profile).to_dict()

    @router.post("/coaching/suggest")
    def coaching_suggest(req: CoachingRequest) -> dict:
        """Return long-form coaching suggestions for a user."""
        payload = {"user_id": req.user_id, "history": req.history}
        if req.goals:
            payload.update(req.goals)
        profile = _hydrate_profile(payload)
        coach = AICoach()
        return {
            "suggestions": [s.to_dict() for s in coach.suggest(profile)],
            "recovery": estimate_recovery(profile).__dict__,
        }
