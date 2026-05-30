"""
tests/test_smoke.py
─────────────────────
Smoke tests — no network, no trained models required.
Run: pytest -q tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


# ─── imports work ────────────────────────────────────────────────────────────

def test_imports():
    """Every critical module should import without side effects."""
    from backend.core.config import settings
    from backend.chatbot.prompts import SYSTEM_PROMPT_COACH, build_user_context_block
    from backend.chatbot.memory import MemoryManager
    from backend.cv.exercise_classifier import compute_joint_angles, build_feature_vector
    from backend.cv.rep_counter import RepCounter
    from backend.services.ml_service import build_features, FEATURE_COLUMNS
    from backend.data_pipeline import WorkoutEvent

    assert settings().COOKIE_SAMESITE in ("lax", "strict", "none")
    assert len(FEATURE_COLUMNS) == 9


# ─── auth: cookie sessions wire end-to-end ──────────────────────────────────

def test_cookie_auth_lifecycle(tmp_path, monkeypatch):
    """Sign up → /me → logout → revoked-cookie rejected. No JWT involved."""
    import asyncio
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Use an isolated DB for this test
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite+aiosqlite:///{tmp_path / 'auth_test.db'}")
    # Force-reset cached engine
    import importlib
    from backend.database import db as db_mod
    importlib.reload(db_mod)
    from backend.routes import auth as auth_routes
    importlib.reload(auth_routes)

    app = FastAPI()
    app.include_router(auth_routes.router)
    asyncio.get_event_loop().run_until_complete(db_mod.init_db())

    with TestClient(app) as c:
        r = c.post("/auth/signup", json={
            "full_name": "T", "email": "t@x.com", "password": "p1234567",
        })
        assert r.status_code == 200
        cookie = r.cookies.get("apex_session")
        assert cookie

        assert c.get("/auth/me").status_code == 200

        c.post("/auth/logout")
        c.cookies.clear()
        # Old cookie should now be revoked server-side
        assert c.get("/auth/me", cookies={"apex_session": cookie}).status_code == 401


# ─── prompt builder never crashes on partial data ────────────────────────────

def test_user_context_with_empty_data():
    from backend.chatbot.prompts import build_user_context_block
    out = build_user_context_block(None)
    assert "USER_CONTEXT" in out
    out2 = build_user_context_block({"name": "Alex"})
    assert "Alex" in out2


def test_user_context_full():
    from backend.chatbot.prompts import build_user_context_block
    out = build_user_context_block({
        "name": "Alex", "age": 28, "weight_kg": 80, "height_cm": 180,
        "gender": "m", "activity_level": 3, "goal": "lose", "target_weight": 75,
    })
    assert "BMI" in out
    assert "TDEE" in out


# ─── rep counter state machine ───────────────────────────────────────────────

def test_rep_counter_squat_cycle():
    from backend.cv.rep_counter import RepCounter
    rc = RepCounter()
    # Angles array: 10 values; indices 2,3 are knee angles (squat rule).
    def angles(knee): return np.array([180, 180, knee, knee, 170, 170, 170, 170, 170, 170], dtype=np.float32)

    # Start standing → squat down → up → that's one rep.
    rc.update("t1", "squat", angles(180))   # phase up
    rc.update("t1", "squat", angles(80))    # phase down
    rc.update("t1", "squat", angles(80))
    rc.update("t1", "squat", angles(170))   # back to up → rep!
    state = rc.update("t1", "squat", angles(180))
    assert state["state"]["reps"] == 1


# ─── feature vector shape ────────────────────────────────────────────────────

def test_feature_vector_shape():
    from backend.cv.exercise_classifier import build_feature_vector
    kp = np.random.rand(51).astype(np.float32)
    feat = build_feature_vector(kp)
    assert feat.shape == (61,)


# ─── ML feature builder ──────────────────────────────────────────────────────

def test_ml_features():
    from backend.services.ml_service import build_features
    X = build_features(age=28, weight_kg=80, height_cm=180,
                        activity_level=3, gender=1)
    assert "bmi" in X.columns
    assert "calories_tdee" in X.columns
    assert round(X.loc[0, "bmi"], 1) == 24.7


# ─── data pipeline validation ────────────────────────────────────────────────

def test_workout_event_validates():
    from backend.data_pipeline import WorkoutEvent
    evt = WorkoutEvent(user_id=1, exercise="squat", sets=3, reps=10,
                        weight_kg=100, form_score=0.9)
    assert evt.form_score == 0.9

    with pytest.raises(Exception):
        WorkoutEvent(user_id=1, exercise="squat", form_score=2.0)  # >1 invalid
