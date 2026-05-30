"""
backend/routes/recommend.py
─────────────────────────────
Personalised recommendation endpoints. Auth is required; the user's profile
drives the fitness level so two users with different goals get different
exercises and rationales. Each call is persisted to recommendation_logs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import RecommendationLog, UserProfile, get_db
from backend.middleware.auth_guard import get_current_user
from backend.services.ml_service import get_ml_service


router = APIRouter(prefix="/recommend", tags=["Recommendations"])


async def _profile(db: AsyncSession, user_id: int) -> dict:
    p = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )).scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Complete your profile first")
    return {
        "name": p.name, "age": p.age, "weight_kg": p.weight_kg,
        "height_cm": p.height_cm, "activity_level": p.activity_level,
        "gender": int(p.gender), "goal": p.goal, "target_weight": p.target_weight,
        "dietary_pref": p.dietary_pref,
    }


@router.get("")
async def recommend(
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """
    Personalised exercise recommendations.

    Pipeline
      1. Load profile from DB (NEVER from query string)
      2. Run the fitness-level classifier on the real profile
      3. Ask the recommender for the right slate, with rationale per item
      4. Persist the slate so we can track engagement later
    """
    prof = await _profile(db, user["user_id"])
    svc  = get_ml_service()
    fit  = svc.classify_fitness(
        age=int(prof["age"]), weight_kg=float(prof["weight_kg"]),
        height_cm=float(prof["height_cm"]),
        activity_level=int(prof["activity_level"]),
        gender=int(prof["gender"]),
    )
    items = svc.recommend_exercises(fit["level_id"], top_k=top_k, profile=prof)

    # Persist this slate
    try:
        db.add(RecommendationLog(
            user_id=user["user_id"],
            rec_type="exercise",
            items=items,
            rationale={
                "fitness_level":   fit["level_name"],
                "fitness_explanation": fit["explanation"]["headline"],
                "goal":            prof.get("goal"),
                "based_on":        ["profile.activity_level", "profile.weight_kg",
                                     "profile.age", "ml.fitness_classifier"],
            },
            fitness_level_id=fit["level_id"],
            served_at=datetime.utcnow(),
        ))
        await db.commit()
    except Exception:
        await db.rollback()

    return {
        "level_id":        fit["level_id"],
        "level_name":      fit["level_name"],
        "fitness_confidence": fit.get("confidence"),
        "fitness_explanation": fit["explanation"],
        "recommendations": items,
        "personalised_for": {
            "goal":           prof.get("goal"),
            "activity_level": prof.get("activity_level"),
        },
    }


@router.get("/history")
async def history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    rows = (await db.execute(
        select(RecommendationLog)
        .where(RecommendationLog.user_id == user["user_id"])
        .order_by(RecommendationLog.served_at.desc())
        .limit(limit)
    )).scalars().all()
    return [
        {
            "id":            r.id,
            "rec_type":      r.rec_type,
            "items":         r.items,
            "rationale":     r.rationale,
            "fitness_level": r.fitness_level_id,
            "served_at":     r.served_at.isoformat() if r.served_at else None,
        }
        for r in rows
    ]
