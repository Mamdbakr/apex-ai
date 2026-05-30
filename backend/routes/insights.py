"""
backend/routes/insights.py
────────────────────────────
AI dashboard endpoints. Orchestrates the existing ML service, forecast,
anomaly, cohort, and insight services into one rich payload that the
frontend dashboard consumes.

Endpoints:
  GET  /insights/dashboard        → full AI dashboard payload
  GET  /insights/forecast         → just the weight curve
  GET  /insights/anomalies        → just the anomaly list
  GET  /insights/cohort           → just the peer comparison
  POST /insights/refresh          → forces re-computation (no cache to bust right now,
                                    but reserved for when we add Redis caching)

All endpoints honour cookie sessions — when a token is present the user_id comes from it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import (
    NutritionLog, PredictionLog, UserProfile, WeightLog, WorkoutLog, get_db,
)
from backend.middleware.auth_guard import get_current_user
from backend.services.anomaly_service import get_anomaly_service
from backend.services.cohort_service import get_cohort_service
from backend.services.explanation_service import get_explanation_service
from backend.services.forecast_service import get_forecast_service
from backend.services.insight_service import get_insight_service
from backend.services.ml_service import get_ml_service


router = APIRouter(prefix="/insights", tags=["AI Dashboard"])


# ─── helpers ──────────────────────────────────────────────────────────────────

def _resolve_user_id(current_user: dict) -> int:
    """Auth is required; user_id always comes from the session — never the body."""
    return int(current_user["user_id"])


async def _load_profile_dict(db: AsyncSession, user_id: int) -> Optional[dict]:
    p = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )).scalar_one_or_none()
    if not p:
        return None
    return {
        "name": p.name, "age": p.age, "weight_kg": p.weight_kg,
        "height_cm": p.height_cm, "activity_level": p.activity_level,
        "gender": p.gender, "goal": p.goal, "target_weight": p.target_weight,
    }


async def _load_workouts(db: AsyncSession, user_id: int) -> list[dict]:
    rows = (await db.execute(
        select(WorkoutLog).where(WorkoutLog.user_id == user_id)
        .order_by(WorkoutLog.logged_at.asc())
    )).scalars().all()
    return [
        {
            "id": r.id, "exercise": r.exercise, "muscle_group": r.muscle_group,
            "sets": r.sets, "reps": r.reps, "weight_kg": r.weight_kg,
            "duration_min": r.duration_min, "form_score": r.form_score,
            "logged_at": r.logged_at,
        }
        for r in rows
    ]


async def _load_weights(db: AsyncSession, user_id: int) -> list[dict]:
    rows = (await db.execute(
        select(WeightLog).where(WeightLog.user_id == user_id)
        .order_by(WeightLog.logged_at.asc())
    )).scalars().all()
    return [{"weight_kg": r.weight_kg, "logged_at": r.logged_at} for r in rows]


def _kpis(profile: dict, workouts: list[dict], weights: list[dict]) -> dict:
    """Real DB-derived KPIs identical in spirit to /user-data/dashboard-full."""
    now = datetime.utcnow()
    w30 = now - timedelta(days=30)
    w7  = now - timedelta(days=7)

    bmi = tdee = calories_goal = 0
    if profile:
        h_m = profile["height_cm"] / 100
        bmi = round(profile["weight_kg"] / (h_m * h_m), 1) if h_m else 0
        bmr = (10 * profile["weight_kg"] + 6.25 * profile["height_cm"]
               - 5 * profile["age"] + (5 if profile.get("gender") == 1 else -161))
        mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}.get(profile.get("activity_level", 2), 1.375)
        tdee = round(bmr * mult)
        goal = (profile.get("goal") or "").lower()
        if goal in ("lose", "cut"): calories_goal = max(int(tdee - 500), 1200)
        elif goal in ("gain", "bulk", "build"): calories_goal = int(tdee + 300)
        else: calories_goal = int(tdee)

    workouts_30d = [w for w in workouts if w["logged_at"] >= w30]
    workouts_7d  = [w for w in workouts if w["logged_at"] >= w7]
    form_scores  = [w["form_score"] for w in workouts if w.get("form_score") and w["form_score"] > 0]
    avg_form     = round(mean(form_scores), 3) if form_scores else 0.0

    workout_dates = sorted({w["logged_at"].date() for w in workouts}, reverse=True)
    streak = 0; cursor = now.date()
    for d in workout_dates:
        if d == cursor or d == cursor - timedelta(days=1):
            streak += 1; cursor = d - timedelta(days=1)
        else:
            break

    weight_trend_30d = 0.0
    recent_weights = [w for w in weights if w["logged_at"] >= w30]
    if len(recent_weights) >= 2:
        delta = recent_weights[-1]["weight_kg"] - recent_weights[0]["weight_kg"]
        days = max((recent_weights[-1]["logged_at"] - recent_weights[0]["logged_at"]).days, 1)
        weight_trend_30d = round(delta * 30 / days, 2)

    return {
        "bmi": bmi, "tdee": tdee, "calories_goal": calories_goal,
        "streak_days": streak,
        "workouts_total": len(workouts),
        "workouts_30d": len(workouts_30d),
        "workouts_7d": len(workouts_7d),
        "avg_form_score": avg_form,
        "consistency": round(min(len(workouts_30d) / 12.0, 1.0), 2),
        "weight_trend_30d": weight_trend_30d,
    }


def _ml_predictions(profile: dict) -> dict:
    """Real ML model calls — no fabricated values."""
    if not profile:
        return {}
    ml = get_ml_service()
    cal = ml.predict_calories(
        age=int(profile["age"]), weight_kg=float(profile["weight_kg"]),
        height_cm=float(profile["height_cm"]),
        activity_level=int(profile.get("activity_level", 2)),
        gender=int(profile.get("gender", 1)),
    )
    chg = ml.predict_weight_change(
        age=int(profile["age"]), weight_kg=float(profile["weight_kg"]),
        height_cm=float(profile["height_cm"]),
        activity_level=int(profile.get("activity_level", 2)),
        gender=int(profile.get("gender", 1)),
    )
    fit = ml.classify_fitness(
        age=int(profile["age"]), weight_kg=float(profile["weight_kg"]),
        height_cm=float(profile["height_cm"]),
        activity_level=int(profile.get("activity_level", 2)),
        gender=int(profile.get("gender", 1)),
    )
    return {
        "calories_target": cal["calories"],
        "calories_source": cal.get("source"),
        "calories_explanation": cal.get("explanation"),
        "weight_change_30d_kg": chg["weight_change_kg_30d"],
        "weight_source": chg.get("source"),
        "weight_explanation": chg.get("explanation"),
        "fitness_level_id": fit["level_id"],
        "fitness_level": fit["level_name"],
        "fitness_probabilities": fit["probabilities"],
        "fitness_confidence": fit.get("confidence"),
        "fitness_explanation": fit.get("explanation"),
        "fitness_source": "model" if fit.get("probabilities") else "heuristic",
    }


def _recommendations(level_id: int, profile: Optional[dict] = None, top_k: int = 5) -> list[dict]:
    return get_ml_service().recommend_exercises(level_id, top_k=top_k, profile=profile)


def _fatigue_risk(workouts: list[dict], avg_form: float) -> dict:
    """Build a fatigue-risk explanation from real recent workout data."""
    now = datetime.utcnow()
    last_3d = [w for w in workouts if w["logged_at"] >= now - timedelta(days=3)]
    prev_3d = [w for w in workouts
               if now - timedelta(days=6) <= w["logged_at"] < now - timedelta(days=3)]

    def _vol(ws):
        return sum((w.get("sets", 0) or 0) * (w.get("reps", 0) or 0) for w in ws)

    v_now, v_prev = _vol(last_3d), _vol(prev_3d)
    intensity_pct = ((v_now - v_prev) / v_prev * 100.0) if v_prev > 0 else 0.0

    workout_dates = sorted({w["logged_at"].date() for w in last_3d})
    rest_days = max(0, 3 - len(workout_dates))

    return get_explanation_service().explain_fatigue_risk(
        intensity_change_pct=intensity_pct,
        avg_form_score=avg_form or 0.0,
        workouts_last_3d=len(last_3d),
        rest_days=rest_days,
    )


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_ai_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Single endpoint that returns everything the AI dashboard renders:
      • profile + computed KPIs
      • all 3 ML predictions (calorie, weight change, fitness level)
      • multi-horizon weight curve from blended models + observed history
      • timeline-to-goal estimate
      • calorie schedule for the next 14 days
      • personalised exercise recommendations matched to predicted fitness level
      • anomaly alerts from real DB statistics
      • peer comparison (cohort) with privacy-safe aggregates
      • LLM-generated insights grounded in all of the above
    """
    user_id = _resolve_user_id(current_user)

    profile = await _load_profile_dict(db, user_id)
    workouts = await _load_workouts(db, user_id)
    weights = await _load_weights(db, user_id)

    if not profile:
        return {
            "available": False,
            "reason": "no_profile",
            "message": "Complete your profile to unlock the AI dashboard.",
            "computed_at": datetime.utcnow().isoformat(),
        }

    # Compute everything (real ML + real stats)
    kpi = _kpis(profile, workouts, weights)
    ml = _ml_predictions(profile)
    forecast = get_forecast_service().weight_curve(profile, weights, days=90)
    timeline = get_forecast_service().timeline_to_goal(profile, weights)
    calorie_curve = get_forecast_service().calorie_curve(profile, days=14)
    recs = _recommendations(ml.get("fitness_level_id", 1), profile=profile, top_k=5)
    anomalies = get_anomaly_service().detect(profile, workouts, weights)
    cohort = await get_cohort_service().compare(db, profile, user_id)
    fatigue_risk = _fatigue_risk(workouts, kpi.get("avg_form_score", 0.0))

    # LLM-generated personalised insights (uses all of the above)
    brief = {
        "profile": profile, "kpi": kpi, "ml_predictions": ml,
        "forecast": forecast, "timeline_to_goal": timeline,
        "anomalies": anomalies, "cohort": cohort,
    }
    insights = await get_insight_service().generate(brief, max_insights=5)

    # Persist the prediction snapshot for history
    try:
        db.add(PredictionLog(
            user_id=user_id,
            prediction_type="ai_dashboard",
            input_data={"profile": profile},
            output_data={
                "kpi": kpi, "ml": ml, "forecast_summary": {
                    "trend_kg_per_day": forecast.get("trend_kg_per_day"),
                    "method": forecast.get("method"),
                    "stability": forecast.get("stability"),
                },
                "anomaly_codes": [a["code"] for a in anomalies],
            },
            confidence=forecast.get("stability"),
        ))
        await db.commit()
    except Exception:
        pass

    return {
        "available": True,
        "computed_at": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "profile": profile,
        "kpi": kpi,
        "ml_predictions": ml,
        "forecast": forecast,
        "timeline_to_goal": timeline,
        "calorie_curve": calorie_curve,
        "recommendations": recs,
        "anomalies": anomalies,
        "cohort": cohort,
        "fatigue_risk": fatigue_risk,
        "insights": insights,
    }


@router.get("/forecast")
async def get_forecast(
    days: int = 90,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = _resolve_user_id(current_user)
    profile = await _load_profile_dict(db, user_id)
    if not profile:
        raise HTTPException(404, "No profile found")
    weights = await _load_weights(db, user_id)
    return get_forecast_service().weight_curve(profile, weights, days=days)


@router.get("/anomalies")
async def get_anomalies(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = _resolve_user_id(current_user)
    profile = await _load_profile_dict(db, user_id) or {}
    workouts = await _load_workouts(db, user_id)
    weights = await _load_weights(db, user_id)
    return {"anomalies": get_anomaly_service().detect(profile, workouts, weights)}


@router.get("/cohort")
async def get_cohort(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = _resolve_user_id(current_user)
    profile = await _load_profile_dict(db, user_id)
    if not profile:
        raise HTTPException(404, "No profile found")
    return await get_cohort_service().compare(db, profile, user_id)


@router.post("/refresh")
async def refresh_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Reserved for cache busting — currently a no-op that returns 200 ok."""
    return {"status": "ok", "user_id": _resolve_user_id(current_user)}
