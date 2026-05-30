"""
backend/routes/data.py
────────────────────────
Public data-pipeline endpoints — cookie-session protected.

The user_id is *always* taken from the access token, even if the request body
contains one. Clients can no longer write data on behalf of another user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from backend.data_pipeline import (CVFrameEvent, NutritionEvent, WeightEvent,
                                     WorkoutEvent, ingest_event, run_batch_etl)
from backend.middleware.auth_guard import get_current_user, require_role


router = APIRouter(prefix="/data", tags=["Data Pipeline"])


def _force_user_id(payload: dict, user_id: int) -> dict:
    """Always overwrite user_id in the payload with the authenticated user's id."""
    payload = dict(payload)
    payload["user_id"] = int(user_id)
    return payload


@router.post("/workout")
async def ingest_workout(
    evt: WorkoutEvent,
    user: dict = Depends(get_current_user),
):
    try:
        stored = await ingest_event(
            "workout", _force_user_id(evt.model_dump(), user["user_id"])
        )
        return {"status": "ok", "stored": stored}
    except ValidationError as e:
        raise HTTPException(422, e.errors())


@router.post("/weight")
async def ingest_weight(
    evt: WeightEvent,
    user: dict = Depends(get_current_user),
):
    stored = await ingest_event(
        "weight", _force_user_id(evt.model_dump(), user["user_id"])
    )
    return {"status": "ok", "stored": stored}


@router.post("/nutrition")
async def ingest_nutrition(
    evt: NutritionEvent,
    user: dict = Depends(get_current_user),
):
    stored = await ingest_event(
        "nutrition", _force_user_id(evt.model_dump(), user["user_id"])
    )
    return {"status": "ok", "stored": stored}


@router.post("/cv-frame")
async def ingest_cv_frame(
    evt: CVFrameEvent,
    user: dict = Depends(get_current_user),
):
    stored = await ingest_event(
        "cv_frame", _force_user_id(evt.model_dump(), user["user_id"])
    )
    return {"status": "ok", "stored": stored}


@router.post("/batch-etl")
async def trigger_batch_etl(
    user: dict = Depends(require_role("admin")),
):
    """Manually kick the nightly aggregator. Admin only — runs heavy SQL."""
    return await run_batch_etl()
