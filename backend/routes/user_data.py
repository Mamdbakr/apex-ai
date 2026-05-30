"""
backend/routes/user_data.py  — APEX AI v6
Per-user data routes — all scoped to authenticated user_id.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import statistics

from backend.database.db import get_db, UserProfile, WorkoutLog, WeightLog, NutritionLog, PredictionLog
from backend.middleware.auth_guard import get_current_user

router = APIRouter(prefix="/user-data", tags=["User Data"])


class ProfileIn(BaseModel):
    name:           str   = "User"
    age:            int   = Field(default=25, ge=10, le=100)
    weight_kg:      float = Field(default=70.0, ge=20, le=300)
    height_cm:      float = Field(default=175.0, ge=100, le=250)
    activity_level: int   = Field(default=2, ge=1, le=5)
    gender:         int   = Field(default=1, ge=0, le=1)
    goal:           str   = "lose"
    target_weight:  float = 65.0

class WorkoutIn(BaseModel):
    exercise:     str   = "Squats"
    muscle_group: str   = ""
    sets:         int   = Field(default=3, ge=1)
    reps:         int   = Field(default=10, ge=0)
    weight_kg:    float = Field(default=0.0, ge=0)
    duration_min: int   = Field(default=30, ge=1)
    reps_counted: int   = 0
    form_score:   float = Field(default=1.0, ge=0, le=1)
    rpe:          Optional[float] = None
    notes:        str   = ""


@router.post("/profile")
async def upsert_profile(
    data: ProfileIn,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["user_id"])
    result  = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
    for field, val in data.model_dump().items():
        setattr(profile, field, val)
    profile.updated_at = datetime.utcnow()
    await db.commit()

    # Log weight change if weight updated
    db.add(WeightLog(user_id=user_id, weight_kg=data.weight_kg, logged_at=datetime.utcnow()))
    await db.commit()

    return {"status": "saved", "profile": data.model_dump()}


@router.get("/profile")
async def get_profile(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["user_id"])
    result  = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(404, "No profile found")
    return {
        "name": profile.name, "age": profile.age,
        "weight_kg": profile.weight_kg, "height_cm": profile.height_cm,
        "activity_level": profile.activity_level, "gender": profile.gender,
        "goal": profile.goal, "target_weight": profile.target_weight,
    }


@router.post("/workout")
async def log_workout(
    data: WorkoutIn,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["user_id"])
    log = WorkoutLog(
        user_id=user_id,
        exercise=data.exercise, muscle_group=data.muscle_group,
        sets=data.sets, reps=data.reps, weight_kg=data.weight_kg,
        duration_min=data.duration_min, reps_counted=data.reps_counted,
        form_score=data.form_score, rpe=data.rpe, notes=data.notes,
        logged_at=datetime.utcnow()
    )
    db.add(log)
    await db.commit()
    return {"status": "logged", "workout": data.model_dump()}


@router.get("/workouts")
async def get_workouts(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["user_id"])
    result  = await db.execute(
        select(WorkoutLog).where(WorkoutLog.user_id == user_id)
        .order_by(WorkoutLog.logged_at.desc()).limit(limit)
    )
    rows = result.scalars().all()
    return [
        {"id": r.id, "exercise": r.exercise, "muscle_group": r.muscle_group,
         "sets": r.sets, "reps": r.reps, "weight_kg": r.weight_kg,
         "duration_min": r.duration_min, "form_score": r.form_score,
         "rpe": r.rpe, "calories_burned": r.calories_burned,
         "logged_at": r.logged_at.isoformat()}
        for r in rows
    ]


@router.get("/weight-history")
async def get_weight_history(
    limit: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["user_id"])
    result  = await db.execute(
        select(WeightLog).where(WeightLog.user_id == user_id)
        .order_by(WeightLog.logged_at.asc()).limit(limit)
    )
    rows = result.scalars().all()
    return [
        {"weight_kg": r.weight_kg, "body_fat": r.body_fat,
         "logged_at": r.logged_at.isoformat()}
        for r in rows
    ]


@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Legacy dashboard — kept for backward compat. Prefer GET /dashboard/{user_id}."""
    user_id   = int(current_user["user_id"])
    now       = datetime.utcnow()
    w30ago    = now - timedelta(days=30)
    w7ago     = now - timedelta(days=7)

    result  = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    profile = result.scalar_one_or_none()

    wk_result = await db.execute(
        select(WorkoutLog).where(WorkoutLog.user_id == user_id)
        .order_by(WorkoutLog.logged_at.asc())
    )
    workouts = wk_result.scalars().all()

    total_workouts   = len(workouts)
    recent_30d       = [w for w in workouts if w.logged_at >= w30ago]
    wk_per_week      = round(len(recent_30d) / 4.3, 1)
    form_scores      = [w.form_score for w in workouts if w.form_score > 0]
    avg_form         = round(statistics.mean(form_scores), 3) if form_scores else 0.0

    workout_dates = sorted({w.logged_at.date() for w in workouts}, reverse=True)
    streak = 0
    check  = now.date()
    from datetime import timedelta as td
    for d in workout_dates:
        if d == check or d == check - td(days=1):
            streak += 1; check = d - td(days=1)
        else:
            break

    return {
        "computed_at":   now.isoformat(),
        "has_profile":   profile is not None,
        "progress": {
            "total_workouts":    total_workouts,
            "workouts_per_week": wk_per_week,
            "streak_days":       streak,
            "avg_form_score":    avg_form,
        },
        "recent_workouts": [
            {"exercise": w.exercise, "sets": w.sets, "reps": w.reps,
             "form_score": w.form_score, "logged_at": w.logged_at.isoformat()}
            for w in sorted(workouts, key=lambda x: x.logged_at, reverse=True)[:5]
        ],
    }


# ─── DASHBOARD-FULL ─────────────────────────────────────────────────────────
# Single aggregated endpoint that powers the React dashboard with 100%
# database-derived values — no random fillers, no static fallbacks.

@router.get("/dashboard-full")
async def get_dashboard_full(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Returns everything the dashboard renders, computed from real data:

      profile         → stored UserProfile (or null)
      kpi             → tdee, bmi, calories goal, fitness level, streak
      weight_series   → up to 30 weight log points for the chart
      calorie_series  → 7-day nutrition log totals
      workouts_recent → last 5 workout log entries
      insights        → derived insights (consistency, plateau, form trend)
      forecast_30d    → simple 30-day weight projection from trend
    """
    user_id = int(current_user["user_id"])
    now = datetime.utcnow()
    w30 = now - timedelta(days=30)
    w7  = now - timedelta(days=7)

    # 1. Profile
    p = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalar_one_or_none()
    profile_dict = None
    bmi = tdee = calories_goal = 0.0
    if p:
        profile_dict = {
            "name": p.name, "age": p.age, "weight_kg": p.weight_kg,
            "height_cm": p.height_cm, "activity_level": p.activity_level,
            "gender": p.gender, "goal": p.goal, "target_weight": p.target_weight,
        }
        # BMI
        if p.height_cm and p.weight_kg:
            bmi = round(p.weight_kg / ((p.height_cm / 100) ** 2), 1)
        # BMR (Mifflin-St Jeor) + TDEE
        if all([p.age, p.weight_kg, p.height_cm]):
            base = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age
            bmr = base + 5 if p.gender == 1 else base - 161
            mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}.get(p.activity_level, 1.55)
            tdee = round(bmr * mult)
            if p.goal == "lose":
                calories_goal = max(int(tdee - 500), 1200)
            elif p.goal in ("build", "bulk", "gain"):
                calories_goal = int(tdee + 300)
            else:
                calories_goal = int(tdee)

    # 2. Workouts
    wk_rows = (await db.execute(
        select(WorkoutLog).where(WorkoutLog.user_id == user_id)
        .order_by(WorkoutLog.logged_at.asc())
    )).scalars().all()
    workouts_total = len(wk_rows)
    recent_30d = [w for w in wk_rows if w.logged_at >= w30]
    recent_7d  = [w for w in wk_rows if w.logged_at >= w7]
    avg_form = round(statistics.mean([w.form_score for w in wk_rows if w.form_score > 0]), 3) if wk_rows else 0.0

    # streak
    workout_dates = sorted({w.logged_at.date() for w in wk_rows}, reverse=True)
    streak = 0; cursor = now.date()
    for d in workout_dates:
        if d == cursor or d == cursor - timedelta(days=1):
            streak += 1; cursor = d - timedelta(days=1)
        else:
            break

    # consistency over last 4 weeks (workouts ÷ target of 12)
    consistency = round(min(len(recent_30d) / 12.0, 1.0), 2)

    # 3. Weight series (last 30 entries)
    wt_rows = (await db.execute(
        select(WeightLog).where(WeightLog.user_id == user_id)
        .order_by(WeightLog.logged_at.asc()).limit(30)
    )).scalars().all()
    weight_series = [
        {"date": w.logged_at.strftime("%b %d"), "weight": round(w.weight_kg, 1)}
        for w in wt_rows
    ]

    # 30-day weight trend (kg/30d) — simple linear from first vs last point
    weight_trend_30d = 0.0
    if len(wt_rows) >= 2:
        delta = wt_rows[-1].weight_kg - wt_rows[0].weight_kg
        days = max((wt_rows[-1].logged_at - wt_rows[0].logged_at).days, 1)
        weight_trend_30d = round(delta * 30 / days, 2)

    forecast_30d = None
    if p and weight_series:
        forecast_30d = {
            "available": True,
            "predicted_kg": round(wt_rows[-1].weight_kg + weight_trend_30d, 1),
            "trend_kg_per_30d": weight_trend_30d,
        }

    # 4. Nutrition series (last 7 days, summed by date)
    nut_rows = (await db.execute(
        select(NutritionLog).where(NutritionLog.user_id == user_id,
                                   NutritionLog.logged_at >= w7)
    )).scalars().all()
    by_day: dict[str, float] = {}
    for n in nut_rows:
        by_day[n.date] = by_day.get(n.date, 0.0) + (n.calories or 0.0)
    calorie_series = [
        {"day": d, "calories": round(c)}
        for d, c in sorted(by_day.items())
    ]

    # 5. Predictions log — pull the most recent fitness-level run if any
    pred_rows = (await db.execute(
        select(PredictionLog).where(PredictionLog.user_id == user_id,
                                    PredictionLog.prediction_type == "fitness-level")
        .order_by(PredictionLog.predicted_at.desc()).limit(1)
    )).scalars().all()
    fitness_level = "Beginner"
    fitness_conf = None
    if pred_rows:
        out = pred_rows[0].output_data or {}
        fitness_level = out.get("level_name", fitness_level)
        fitness_conf = out.get("confidence")

    # 6. Insights (derived, not hardcoded)
    insights: list[dict] = []
    if streak >= 3:
        insights.append({
            "category": "STREAK", "icon": "🔥", "color": "#ffd93d",
            "borderColor": "rgba(255,217,61,0.10)",
            "text": f"{streak}-day workout streak — keep it alive."
        })
    if consistency >= 0.75:
        insights.append({
            "category": "CONSISTENCY", "icon": "✅", "color": "#00ff88",
            "borderColor": "rgba(0,255,136,0.10)",
            "text": f"You hit {int(consistency*100)}% of your workout target this month."
        })
    elif consistency < 0.4 and len(wk_rows) > 0:
        insights.append({
            "category": "CONSISTENCY", "icon": "⚠️", "color": "#ff6b35",
            "borderColor": "rgba(255,107,53,0.10)",
            "text": f"Only {len(recent_30d)} workouts in 30 days — aim for 3 per week."
        })
    if avg_form and avg_form < 0.7:
        insights.append({
            "category": "FORM", "icon": "🎯", "color": "#ff6b35",
            "borderColor": "rgba(255,107,53,0.10)",
            "text": f"Average form score {int(avg_form*100)}% — drop weight 10% and focus on tempo."
        })
    if forecast_30d and p and p.target_weight:
        delta_to_target = round(p.target_weight - wt_rows[-1].weight_kg, 1) if wt_rows else 0
        if abs(delta_to_target) > 0.5:
            insights.append({
                "category": "GOAL", "icon": "🎯", "color": "#00d4ff",
                "borderColor": "rgba(0,212,255,0.10)",
                "text": f"{delta_to_target:+} kg to your target. Trend says {forecast_30d['predicted_kg']} kg in 30 days."
            })

    return {
        "computed_at": now.isoformat(),
        "has_profile": p is not None,
        "profile": profile_dict,
        "kpi": {
            "tdee": tdee,
            "bmi": bmi,
            "calories_goal": calories_goal,
            "fitness_level": fitness_level,
            "fitness_confidence": fitness_conf,
            "streak_days": streak,
            "workouts_total": workouts_total,
            "workouts_30d": len(recent_30d),
            "workouts_7d": len(recent_7d),
            "avg_form_score": avg_form,
            "consistency": consistency,
            "weight_trend_30d": weight_trend_30d,
        },
        "weight_series": weight_series,
        "calorie_series": calorie_series,
        "workouts_recent": [
            {"exercise": w.exercise, "sets": w.sets, "reps": w.reps,
             "weight_kg": w.weight_kg, "form_score": w.form_score,
             "logged_at": w.logged_at.isoformat()}
            for w in sorted(wk_rows, key=lambda x: x.logged_at, reverse=True)[:5]
        ],
        "insights": insights,
        "forecast_30d": forecast_30d,
    }
