"""
backend/routes/predict.py
───────────────────────────
ML prediction endpoints. Every endpoint is cookie-session protected and pulls the user's
profile from the DB (rather than trusting client-supplied stats), so two users
will *always* see different predictions.

Endpoints
  POST /predict/calories        body: {} (uses signed-in user's profile)
  POST /predict/weight-change   body: {}
  POST /predict/fitness-level   body: {}
  POST /predict/all             body: {}            ← convenience aggregator
  POST /predict/explain         body: {}            ← all 3 + explanations only

If a request body is provided it can override profile fields (handy for
"what-if" exploration), but the user_id is always taken from the session.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import PredictionLog, UserProfile, get_db
from backend.middleware.auth_guard import get_current_user
from backend.services.ml_service import get_ml_service


router = APIRouter(prefix="/predict", tags=["ML Predictions"])


class WhatIfOverrides(BaseModel):
    """Optional per-call overrides for what-if analysis."""
    age:            Optional[int]   = Field(default=None, ge=10, le=100)
    weight_kg:      Optional[float] = Field(default=None, ge=20, le=400)
    height_cm:      Optional[float] = Field(default=None, ge=100, le=250)
    activity_level: Optional[int]   = Field(default=None, ge=1, le=5)
    gender:         Optional[int]   = Field(default=None, ge=0, le=1)


async def _resolve_features(
    db: AsyncSession, user_id: int, overrides: Optional[WhatIfOverrides] = None
) -> dict:
    p = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )).scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Complete your profile first to unlock predictions")

    feats = {
        "age": p.age,
        "weight_kg": p.weight_kg,
        "height_cm": p.height_cm,
        "activity_level": p.activity_level,
        "gender": int(p.gender),
    }
    if overrides:
        for k, v in overrides.model_dump(exclude_none=True).items():
            feats[k] = v
    return feats


async def _record(
    db: AsyncSession, user_id: int, kind: str, inp: dict, outp: dict, conf: Optional[float] = None
):
    try:
        db.add(PredictionLog(
            user_id=user_id,
            prediction_type=kind,
            input_data=inp,
            output_data={k: v for k, v in outp.items() if k != "explanation"},
            explanation=outp.get("explanation"),
            confidence=conf,
            predicted_at=datetime.utcnow(),
        ))
        await db.commit()
    except Exception:
        await db.rollback()


@router.post("/calories")
async def calories(
    overrides: WhatIfOverrides = WhatIfOverrides(),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    feats = await _resolve_features(db, user["user_id"], overrides)
    out = get_ml_service().predict_calories(**feats)
    await _record(db, user["user_id"], "calories", feats, out)
    return out


@router.post("/weight-change")
async def weight_change(
    overrides: WhatIfOverrides = WhatIfOverrides(),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    feats = await _resolve_features(db, user["user_id"], overrides)
    out = get_ml_service().predict_weight_change(**feats)
    await _record(db, user["user_id"], "weight_change", feats, out)
    return out


@router.post("/fitness-level")
async def fitness_level(
    overrides: WhatIfOverrides = WhatIfOverrides(),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    feats = await _resolve_features(db, user["user_id"], overrides)
    out = get_ml_service().classify_fitness(**feats)
    await _record(db, user["user_id"], "fitness_level", feats, out, conf=out.get("confidence"))
    return out


@router.post("/all")
async def predict_all(
    overrides: WhatIfOverrides = WhatIfOverrides(),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """One call returns every prediction — convenient for the dashboard."""
    feats = await _resolve_features(db, user["user_id"], overrides)
    svc = get_ml_service()
    out = {
        "calories":      svc.predict_calories(**feats),
        "weight_change": svc.predict_weight_change(**feats),
        "fitness":       svc.classify_fitness(**feats),
    }
    # Snapshot the bundle so the user has a history of "what the AI told me"
    await _record(
        db, user["user_id"], "ai_bundle", feats,
        {
            "calories":      out["calories"]["calories"],
            "weight_change": out["weight_change"]["weight_change_kg_30d"],
            "fitness_level": out["fitness"]["level_name"],
        },
    )
    return out


@router.post("/explain")
async def explain(
    overrides: WhatIfOverrides = WhatIfOverrides(),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Return only the explanation payloads — useful for the 'Why?' panel."""
    feats = await _resolve_features(db, user["user_id"], overrides)
    svc = get_ml_service()
    cal = svc.predict_calories(**feats)
    chg = svc.predict_weight_change(**feats)
    fit = svc.classify_fitness(**feats)
    return {
        "features":        feats,
        "calories":        cal["explanation"],
        "weight_change":   chg["explanation"],
        "fitness_level":   fit["explanation"],
    }


@router.get("/history")
async def prediction_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    rows = (await db.execute(
        select(PredictionLog)
        .where(PredictionLog.user_id == user["user_id"])
        .order_by(PredictionLog.predicted_at.desc())
        .limit(min(max(limit, 1), 100))
    )).scalars().all()
    return [
        {
            "id":              r.id,
            "prediction_type": r.prediction_type,
            "input_data":      r.input_data,
            "output_data":     r.output_data,
            "explanation":     r.explanation,
            "confidence":      r.confidence,
            "predicted_at":    r.predicted_at.isoformat() if r.predicted_at else None,
        }
        for r in rows
    ]
