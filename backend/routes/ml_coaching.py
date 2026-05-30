"""
backend/routes/ml_coaching.py
──────────────────────────────
New (additive) routes that surface the apex_ml adaptive workout
generator and AI coach over the existing database.

This file is fully self-contained:
  - Uses the project's existing auth dependency (cookie-session).
  - Reads from existing tables (UserProfile, WorkoutLog) without
    modifying them.
  - Persists results to the existing RecommendationLog table so the
    history endpoints continue to work uniformly.
  - Routes mount under /ml/* to guarantee no URL collision with the
    existing /recommend/* surface.

Endpoints
  GET  /ml/health           liveness + supported exercises (no auth)
  GET  /ml/workout          generate today's adaptive workout (auth)
  GET  /ml/coaching         long-form AI coach suggestions   (auth)
  GET  /ml/recovery         readiness / fatigue summary       (auth)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import (
    RecommendationLog, UserProfile, WorkoutLog, get_db,
)
from backend.middleware.auth_guard import get_current_user

# apex_ml imports — kept inside the route module so a broken apex_ml
# never crashes app startup. We resolve them lazily on first call.
from apex_ml.recommendation import (
    AICoach, UserGoals, WorkoutGenerator,
    UserProfile as MLProfile, WorkoutSession as MLSession,
    WorkoutSet as MLSet, estimate_recovery,
)
from apex_ml.temporal_pose import EXERCISE_DEFAULTS


router = APIRouter(prefix="/ml", tags=["ML Coaching"])


# ── Goal mapping ─────────────────────────────────────────────────────────────
# UserProfile.goal is a free-form string in the project ("lose", "gain",
# "maintain", "build_muscle", etc.). Map to apex_ml's enum.
_GOAL_MAP = {
    "lose":          "fat_loss",
    "lose_weight":   "fat_loss",
    "fat_loss":      "fat_loss",
    "cut":           "fat_loss",
    "maintain":      "general_fitness",
    "general":       "general_fitness",
    "general_fitness": "general_fitness",
    "gain":          "hypertrophy",
    "gain_weight":   "hypertrophy",
    "build_muscle":  "hypertrophy",
    "hypertrophy":   "hypertrophy",
    "strength":      "strength",
    "strong":        "strength",
    "endurance":     "endurance",
    "cardio":        "endurance",
}


def _equipment_for(activity_level: int) -> list:
    """Heuristic equipment guess until the schema exposes it directly.

    activity_level is 1..5 in the project. Beginners get bodyweight only;
    intermediate users add dumbbells; advanced add barbell + pullup bar.
    Safe defaults — the user can always override via the UI later.
    """
    if activity_level <= 2:
        return ["bodyweight"]
    if activity_level <= 3:
        return ["bodyweight", "dumbbell"]
    return ["bodyweight", "dumbbell", "barbell", "cable", "pullup_bar"]


async def _load_ml_profile(db: AsyncSession, user_id: int) -> MLProfile:
    """Build an apex_ml UserProfile from the project's tables."""
    prof_row = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )).scalar_one_or_none()
    if prof_row is None:
        raise HTTPException(404, "Complete your profile before requesting ML coaching")

    goal = _GOAL_MAP.get((prof_row.goal or "").lower(), "general_fitness")
    # We don't currently expose weekly_sessions in UserProfile, so use a
    # reasonable activity-level-derived default.
    weekly_sessions = max(2, min(6, int(prof_row.activity_level or 3) + 1))
    available_minutes = 45

    ml_profile = MLProfile(
        user_id=str(user_id),
        goals=UserGoals(
            primary=goal,
            weekly_sessions=weekly_sessions,
            available_minutes=available_minutes,
            equipment=_equipment_for(int(prof_row.activity_level or 3)),
        ),
    )

    # Hydrate with the last 90 days of workouts. WorkoutLog stores one
    # "logged" workout per row; apex_ml expects sessions composed of sets.
    # We treat each row as a single-set session (good enough for the
    # readiness model — strength estimate uses Epley over the row).
    rows = (await db.execute(
        select(WorkoutLog)
        .where(WorkoutLog.user_id == user_id)
        .order_by(WorkoutLog.logged_at.desc())
        .limit(200)
    )).scalars().all()

    for r in rows:
        # Project form_score may be in [0, 1] or [0, 100] — normalize.
        fs = float(r.form_score or 0.0)
        quality = fs * 100.0 if fs <= 1.0 else fs
        # Make timestamps timezone-aware (apex_ml uses tz-aware datetimes).
        ts = r.logged_at if r.logged_at else datetime.utcnow()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ml_profile.add_session(MLSession(
            timestamp=ts,
            duration_minutes=float(r.duration_min or 0),
            perceived_difficulty=float(r.rpe) if r.rpe is not None else None,
            notes=r.notes or "",
            sets=[MLSet(
                exercise=r.exercise or "unknown",
                reps=int(r.reps_counted or r.reps or 0),
                weight_kg=float(r.weight_kg or 0.0),
                rpe=float(r.rpe) if r.rpe is not None else None,
                quality_score=quality if quality > 0 else None,
                completed=True,
            )],
        ))
    return ml_profile


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def ml_health():
    """Liveness probe for the apex_ml layer. Does not require auth so
    monitoring tools can scrape it from anywhere."""
    return {
        "status": "ok",
        "layer": "apex_ml",
        "supported_exercises": list(EXERCISE_DEFAULTS.keys()),
    }


@router.get("/workout")
async def ml_workout(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Generate an adaptive workout for today, driven by the user's history."""
    ml_profile = await _load_ml_profile(db, user["user_id"])
    workout = WorkoutGenerator().generate(ml_profile)
    out = workout.to_dict()

    # Persist into the existing recommendation_logs table so the project's
    # history endpoint can show this alongside /recommend rows.
    try:
        db.add(RecommendationLog(
            user_id=user["user_id"],
            rec_type="ml_workout",
            items=out["blocks"],
            rationale={
                "goal":      out["goal"],
                "readiness": out["readiness"],
                "recovery_note": out["recovery_note"],
                "coaching":  out["coaching"],
                "based_on":  ["apex_ml.WorkoutGenerator",
                              "user_profiles", "workout_logs"],
            },
            fitness_level_id=None,
            served_at=datetime.utcnow(),
        ))
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.debug(f"ml_workout persist skipped: {e}")

    return out


@router.get("/coaching")
async def ml_coaching(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Return long-form coaching suggestions (deload/substitution/progression)."""
    ml_profile = await _load_ml_profile(db, user["user_id"])
    coach = AICoach()
    return {
        "suggestions": [s.to_dict() for s in coach.suggest(ml_profile)],
        "recovery": estimate_recovery(ml_profile).__dict__,
    }


@router.get("/recovery")
async def ml_recovery(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Recovery / readiness summary on its own — cheap call for badges/UI."""
    ml_profile = await _load_ml_profile(db, user["user_id"])
    return estimate_recovery(ml_profile).__dict__


@router.post("/session/{session_id}/end")
async def end_temporal_session(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """End an apex_ml temporal pose session and return its summary.

    Use this for the *temporal-pose* session aggregator only. The
    existing /vision/session/finish (which writes a WorkoutLog) is
    untouched — call BOTH if you want both effects.
    """
    try:
        from apex_ml.integrations.apex_ai_bridge import get_temporal_sessions
    except Exception as e:
        raise HTTPException(500, f"apex_ml unavailable: {e}")
    summary = get_temporal_sessions().end(session_id)
    if summary is None:
        raise HTTPException(404, "no temporal session for that id")
    return summary
