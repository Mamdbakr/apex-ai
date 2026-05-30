"""
backend/data_pipeline/pipeline.py
───────────────────────────────────
Data pipeline orchestrator — four stages, each isolated and testable.

    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  COLLECTION  │──►│ PREPROCESS   │──►│   STORAGE    │──►│   SERVING    │
    │  /ingest     │   │  validate,   │   │  SQL + raw   │   │  feature API │
    │  /cv events  │   │  enrich      │   │  JSONL files │   │  + batch ETL │
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘

Two processing modes:
  - REAL-TIME: API endpoints call `pipeline.ingest_event(...)` → immediate
    validation, derived-feature computation, DB insert, and a row appended
    to data/events/{date}.jsonl for durable re-processing.
  - BATCH: `pipeline.run_batch_etl()` sweeps the JSONL files, recomputes
    aggregates per user (7d/30d rolling stats), and writes a feature row
    into user_feature_vectors.

For production throughput you swap the in-process queue for Kafka/Redis-streams
by replacing `_EVENT_QUEUE.put(...)` with a producer call — nothing else changes.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the DB module carefully — existing schema lives in backend.database.db
from backend.database.db import (AsyncSessionLocal, NutritionLog, PredictionLog,
                                  UserFeatureVector, WeightLog, WorkoutLog)


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
EVENTS_DIR = DATA_ROOT / "events"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
for d in (EVENTS_DIR, RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─── EVENT SCHEMAS (validation) ───────────────────────────────────────────────

class WorkoutEvent(BaseModel):
    user_id: int
    exercise: str
    sets: int = 3
    reps: int = 10
    weight_kg: float = 0.0
    duration_min: int = 30
    reps_counted: int = 0
    form_score: float = Field(ge=0.0, le=1.0, default=1.0)
    rpe: Optional[float] = Field(ge=0.0, le=10.0, default=None)
    notes: str = ""

    @field_validator("form_score")
    @classmethod
    def _norm_form(cls, v): return max(0.0, min(1.0, v))


class WeightEvent(BaseModel):
    user_id: int
    weight_kg: float = Field(gt=20.0, lt=400.0)
    body_fat: Optional[float] = Field(default=None, ge=2.0, le=70.0)


class NutritionEvent(BaseModel):
    user_id: int
    date: str
    calories: float = Field(ge=0, le=15000)
    protein_g: float = Field(ge=0, le=1500)
    carbs_g:   float = Field(ge=0, le=2500)
    fat_g:     float = Field(ge=0, le=1000)
    water_ml:  float = Field(ge=0, le=20000)


class CVFrameEvent(BaseModel):
    user_id: int
    session_id: str
    exercise_id: str
    confidence: float
    reps: int
    form_score: float


EVENT_SCHEMAS = {
    "workout": WorkoutEvent,
    "weight": WeightEvent,
    "nutrition": NutritionEvent,
    "cv_frame": CVFrameEvent,
}


# ─── STAGE 1: collection & validation ────────────────────────────────────────

def _validate(event_type: str, raw: Dict[str, Any]) -> BaseModel:
    schema = EVENT_SCHEMAS.get(event_type)
    if not schema:
        raise ValueError(f"Unknown event type: {event_type}")
    return schema.model_validate(raw)


# ─── STAGE 2: preprocess (derived fields) ────────────────────────────────────

# MET values in the compendium of physical activities.
# Rough averages across "light/moderate/vigorous" variants of each lift.
_MET_TABLE = {
    "squat":               5.0,
    "deadlift":            6.0,
    "bench_press":         4.0,
    "push_up":             3.8,
    "pull_up":             8.0,
    "shoulder_press":      4.0,
    "plank":               3.0,
    "lat_pulldown":        4.0,
    "lateral_raise":       3.5,
    "leg_extension":       3.5,
    "leg_raises":          3.0,
    "romanian_deadlift":   5.5,
    "t_bar_row":           5.0,
    "tricep_dips":         5.0,
    "barbell_biceps_curl": 3.5,
}


async def _user_weight_kg(user_id: int, fallback: float = 75.0) -> float:
    """Real lookup of the logged-in user's body weight for calorie math.

    Order of precedence:
      1. Most recent WeightLog row for the user (last entry wins).
      2. UserProfile.weight_kg.
      3. `fallback` (only used if a brand-new user has neither — the value
         is still noted in logs so bad data is obvious in monitoring).
    """
    from backend.database.db import UserProfile  # local import avoids cycle

    async with AsyncSessionLocal() as db:  # type: AsyncSession
        # 1 — latest weight log
        row = (await db.execute(
            select(WeightLog.weight_kg)
            .where(WeightLog.user_id == user_id)
            .order_by(WeightLog.logged_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if row is not None:
            return float(row)

        # 2 — user profile
        row = (await db.execute(
            select(UserProfile.weight_kg)
            .where(UserProfile.user_id == user_id)
            .limit(1)
        )).scalar_one_or_none()
        if row is not None:
            return float(row)

    logger.warning(f"No weight on file for user_id={user_id} — falling back to {fallback} kg")
    return fallback


async def _enrich_workout(evt: WorkoutEvent) -> Dict[str, Any]:
    """Add `calories_burned` using the user's REAL body weight."""
    met = _MET_TABLE.get(evt.exercise, 4.5)
    weight_kg = await _user_weight_kg(evt.user_id)
    # Kcal = MET × 3.5 × body_mass_kg / 200 × duration_min  (ACSM formula)
    calories = round(met * 3.5 * weight_kg / 200 * evt.duration_min, 1)
    return {**evt.model_dump(), "calories_burned": calories}


# ─── STAGE 3: storage ────────────────────────────────────────────────────────

async def _persist(event_type: str, payload: Dict[str, Any]) -> None:
    """Durable write: DB row + JSONL append."""
    # 3a — append to daily JSONL (replay-safe)
    day = datetime.utcnow().strftime("%Y-%m-%d")
    line = json.dumps({"ts": datetime.utcnow().isoformat(),
                       "type": event_type, "data": payload})
    (EVENTS_DIR / f"{day}.jsonl").open("a", encoding="utf-8").write(line + "\n")

    # 3b — structured DB insert
    async with AsyncSessionLocal() as db:  # type: AsyncSession
        if event_type == "workout":
            db.add(WorkoutLog(**{k: v for k, v in payload.items()
                                  if k in WorkoutLog.__table__.columns.keys()}))
        elif event_type == "weight":
            db.add(WeightLog(**payload))
        elif event_type == "nutrition":
            db.add(NutritionLog(**payload))
        elif event_type == "cv_frame":
            # CV frames don't have their own table — we log as prediction
            db.add(PredictionLog(
                user_id=payload["user_id"], prediction_type="cv_frame",
                input_data={"session_id": payload["session_id"]},
                output_data={"exercise": payload["exercise_id"],
                             "reps": payload["reps"],
                             "form_score": payload["form_score"]},
                confidence=payload["confidence"],
                model_version="exercisenet_v1",
            ))
        await db.commit()


# ─── PUBLIC: single-event ingest (called from API routes) ────────────────────

async def ingest_event(event_type: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Real-time path. Validates, enriches, persists, returns the stored payload."""
    model = _validate(event_type, raw)
    payload = model.model_dump()
    if event_type == "workout":
        payload = await _enrich_workout(model)
    await _persist(event_type, payload)
    return payload


# ─── STAGE 4: batch ETL (rolling aggregates per user) ────────────────────────

async def run_batch_etl() -> Dict[str, int]:
    """
    Recompute user_feature_vectors from the last 30 days of logs.
    Designed to be idempotent — run it hourly by cron/Celery.
    """
    now = datetime.utcnow()
    start_30d = now - timedelta(days=30)
    start_7d = now - timedelta(days=7)
    updated = 0

    async with AsyncSessionLocal() as db:  # type: AsyncSession
        user_ids = (await db.execute(select(WorkoutLog.user_id).distinct())).scalars().all()
        for uid in user_ids:
            wl_30 = (await db.execute(
                select(WorkoutLog).where(
                    WorkoutLog.user_id == uid,
                    WorkoutLog.logged_at >= start_30d,
                )
            )).scalars().all()

            weights = (await db.execute(
                select(WeightLog).where(
                    WeightLog.user_id == uid,
                    WeightLog.logged_at >= start_30d,
                ).order_by(WeightLog.logged_at)
            )).scalars().all()

            avg_form = float(np.mean([w.form_score for w in wl_30])) if wl_30 else 0.0
            avg_dur = float(np.mean([w.duration_min for w in wl_30])) if wl_30 else 0.0
            wkly = max(1, len({w.logged_at.date() for w in wl_30}))
            consistency = min(1.0, wkly / 21.0)     # 3/wk over 30d ≈ 1.0

            def trend(series, since):
                pts = [w for w in weights if w.logged_at >= since]
                if len(pts) < 2: return 0.0
                xs = np.arange(len(pts))
                ys = np.array([p.weight_kg for p in pts])
                return float(np.polyfit(xs, ys, 1)[0])    # kg per log-interval

            vec = (await db.execute(
                select(UserFeatureVector).where(UserFeatureVector.user_id == uid)
            )).scalar_one_or_none()
            if vec is None:
                vec = UserFeatureVector(user_id=uid)
                db.add(vec)

            vec.workouts_30d = len(wl_30)
            vec.avg_duration = round(avg_dur, 1)
            vec.avg_form_score = round(avg_form, 3)
            vec.consistency_score = round(consistency, 3)
            vec.weight_trend_7d = round(trend(weights, start_7d), 3)
            vec.weight_trend_30d = round(trend(weights, start_30d), 3)
            vec.updated_at = now
            updated += 1

        await db.commit()

    logger.info(f"Batch ETL complete · {updated} users updated")
    return {"users_updated": updated}
