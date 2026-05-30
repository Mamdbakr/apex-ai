"""
backend/routes/vision.py  (high-FPS rewrite)
──────────────────────────────────────────────
Key change vs the original: the WebSocket loop is split into two concurrent
tasks so YOLO inference never blocks frame ingestion or result delivery.

  ┌─ recv_task  ──────────────────────────────────────────────────────┐
  │  Reads frames from the WebSocket as fast as the client sends them.│
  │  Always keeps only the LATEST frame in `frame_slot` (a 1-item     │
  │  asyncio.Queue with maxsize=1 — old frames are dropped).          │
  └───────────────────────────────────────────────────────────────────┘
                          │ latest frame
                          ▼
  ┌─ infer_task ──────────────────────────────────────────────────────┐
  │  Pulls from frame_slot, runs YOLO in a thread-pool executor,      │
  │  updates `last_result`, then immediately sends the JSON back.     │
  │  While YOLO is running the recv_task keeps draining incoming      │
  │  frames so the WebSocket buffer never backs up.                   │
  └───────────────────────────────────────────────────────────────────┘

  ┌─ heartbeat_task ──────────────────────────────────────────────────┐
  │  Every 40 ms sends the last_result again so the frontend UI       │
  │  animates at 25 FPS even when YOLO is slower than that.           │
  └───────────────────────────────────────────────────────────────────┘

Result: UI always runs at 25 FPS; YOLO runs as fast as the CPU allows
(typically 6-12 FPS on CPU) but that's invisible to the user because
the heartbeat fills the gaps with the last known pose/reps/form data.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.cv.pipeline import get_cv_pipeline
from backend.cv.rep_counter import get_rep_counter
from backend.cv.exercise_profiles import EXERCISE_PROFILES, all_exercise_ids
from backend.data_pipeline import ingest_event
from backend.database.db import CVAnalysis, get_db, AsyncSessionLocal
from backend.middleware.auth_guard import get_current_user, get_user_for_websocket


router = APIRouter(prefix="/vision", tags=["Computer Vision"])

# How often the heartbeat re-sends the last result to keep UI at 25 FPS
_HEARTBEAT_INTERVAL = 0.040   # 40 ms → 25 FPS


# ─── helpers ──────────────────────────────────────────────────────────────────

async def _persist_analysis(
    db: AsyncSession, user_id: int, session_id: str, result_dict: dict, frame_count: int = 1,
) -> None:
    try:
        cues = result_dict.get("form_cues") or []
        feedback = "; ".join(cues[:3]) if cues else ""
        db.add(CVAnalysis(
            user_id=user_id,
            session_id=session_id,
            exercise=result_dict.get("exercise_name") or result_dict.get("exercise_id") or "unknown",
            confidence=float(result_dict.get("confidence", 0.0)),
            reps=int(result_dict.get("reps", 0)),
            form_score=float(result_dict.get("form_score", 0.0)) / 100.0
                if result_dict.get("form_score", 0.0) > 1.0
                else float(result_dict.get("form_score", 0.0)),
            feedback=feedback,
            suggestions=cues,
            duration_s=float(result_dict.get("hold_seconds", 0.0)),
            frame_count=frame_count,
            keypoint_summary={
                "phase": result_dict.get("phase"),
                "fps":   result_dict.get("fps"),
                "top_3": result_dict.get("top_3", []),
            },
            analysed_at=datetime.utcnow(),
        ))
        await db.commit()
    except Exception as e:
        logger.debug(f"CVAnalysis persist skipped: {e}")
        await db.rollback()


# ─── Single-image analysis ────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    session_id_q:    str = Query("default", alias="session_id"),
    session_id_f:    Optional[str] = Form(default=None, alias="session_id"),
    exercise_hint_q: Optional[str] = Query(default=None, alias="exercise_hint"),
    exercise_hint_f: Optional[str] = Form(default=None,  alias="exercise_hint"),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp",
                                  "image/jpg", None):
        raise HTTPException(422, f"Unsupported content-type: {file.content_type}")

    data = await file.read()
    if len(data) < 200:
        raise HTTPException(422, "Image is too small or empty.")

    session_id    = session_id_f    or session_id_q    or "default"
    exercise_hint = exercise_hint_f or exercise_hint_q or None

    pipe   = get_cv_pipeline()
    result = pipe.analyze_frame(data, session_id=session_id,
                                exercise_hint=exercise_hint)
    out = result.to_dict()

    if result.detected:
        await _persist_analysis(db, user["user_id"], session_id, out, frame_count=1)
        try:
            await ingest_event("cv_frame", {
                "user_id": user["user_id"], "session_id": session_id,
                "exercise_id": result.exercise_id,
                "confidence": result.confidence,
                "reps": result.reps,
                "form_score": result.form_score / 100.0
                    if result.form_score > 1.0 else result.form_score,
            })
        except Exception as e:
            logger.debug(f"cv_frame ingest skipped: {e}")

    return out


# ─── Reset ────────────────────────────────────────────────────────────────────

@router.post("/reset")
async def reset(
    session_id: str = Query("default"),
    user: dict = Depends(get_current_user),
):
    get_rep_counter().reset(session_id)
    try:
        from apex_ml.integrations.apex_ai_bridge import get_temporal_sessions
        get_temporal_sessions().reset(session_id)
    except Exception as _e:
        logger.debug(f"apex_ml temporal reset skipped: {_e}")
    return {"status": "ok", "session_id": session_id, "user_id": user["user_id"]}


# ─── Finish ───────────────────────────────────────────────────────────────────

@router.post("/session/finish")
async def finish_session(
    session_id: str = Query(...),
    sets: int = Query(1, ge=1, le=20),
    duration_min: int = Query(5, ge=1, le=240),
    notes: str = Query("", max_length=500),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    counter = get_rep_counter()
    st = counter.get(session_id)

    if st.exercise in (None, "", "unknown"):
        raise HTTPException(422, "No exercise detected in this session yet.")

    form_0_to_1 = st.avg_form_score if st.form_score_n > 0 else 0.0

    payload = {
        "user_id": user["user_id"],
        "exercise": st.exercise,
        "sets": sets,
        "reps": int(st.reps) if st.reps else 0,
        "weight_kg": 0.0,
        "duration_min": duration_min,
        "reps_counted": int(st.reps),
        "form_score": form_0_to_1,
        "notes": notes or f"CV session {session_id} ({st.form_score_n} frames analysed)",
    }
    stored = await ingest_event("workout", payload)

    await _persist_analysis(
        db, user["user_id"], session_id,
        {
            "exercise_name": st.exercise,
            "confidence": 1.0,
            "reps": int(st.reps),
            "form_score": form_0_to_1,
            "form_cues": [],
            "phase": "done",
            "hold_seconds": duration_min * 60.0,
        },
        frame_count=st.form_score_n,
    )

    counter.reset(session_id)
    return {
        "status": "ok",
        "workout_logged": stored,
        "frames_analysed": st.form_score_n,
        "avg_form_score": form_0_to_1,
    }


# ─── History ──────────────────────────────────────────────────────────────────

@router.get("/history")
async def cv_history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    rows = (await db.execute(
        select(CVAnalysis)
        .where(CVAnalysis.user_id == user["user_id"])
        .order_by(CVAnalysis.analysed_at.desc())
        .limit(limit)
    )).scalars().all()
    return [
        {
            "id":          r.id,
            "session_id":  r.session_id,
            "exercise":    r.exercise,
            "confidence":  r.confidence,
            "reps":        r.reps,
            "form_score":  r.form_score,
            "feedback":    r.feedback,
            "suggestions": r.suggestions,
            "duration_s":  r.duration_s,
            "frame_count": r.frame_count,
            "analysed_at": r.analysed_at.isoformat() if r.analysed_at else None,
        }
        for r in rows
    ]


# ─── Discovery ───────────────────────────────────────────────────────────────

@router.get("/exercises")
async def list_exercises():
    return [
        {"id": p.name, "name": p.display_name, "mode": p.mode}
        for p in (EXERCISE_PROFILES[k] for k in all_exercise_ids())
    ]


# ─── Real-time WebSocket — decoupled recv / infer / heartbeat ────────────────

@router.websocket("/stream")
async def stream(
    ws: WebSocket,
    sid: str = "default",
    session: str = "",
    exercise_hint: str = "",
):
    # ── Auth ────────────────────────────────────────────────────────────────
    cookies = dict(ws.cookies) if hasattr(ws, "cookies") else {}
    async with AsyncSessionLocal() as db:
        user = await get_user_for_websocket(cookies, session or None, db)

    if not user:
        await ws.close(code=4401)
        logger.info("WS stream rejected — missing/invalid session")
        return

    user_id = user["user_id"]
    await ws.accept()
    pipe = get_cv_pipeline()
    logger.info(f"WS stream opened · sid={sid} uid={user_id} hint={exercise_hint!r}")

    # ── Shared state between tasks ──────────────────────────────────────────
    # frame_slot: maxsize=1 queue — old frames are dropped when a new one
    # arrives before YOLO has finished the previous one.
    frame_slot: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    # last_result_holder: mutable container so heartbeat_task can always
    # send the most recent result without needing a lock.
    last_result: list[Optional[dict]] = [None]   # [0] = latest dict or None

    current_hint: list[Optional[str]] = [exercise_hint or None]
    stop_event   = asyncio.Event()
    frame_count  = 0
    infer_count  = 0
    persist_every = 30

    # ── Task 1: receive frames from WebSocket ───────────────────────────────
    async def recv_task():
        nonlocal frame_count
        try:
            while not stop_event.is_set():
                msg = await ws.receive()

                # Text control messages
                if msg.get("text"):
                    text = msg["text"].strip()
                    low  = text.lower()
                    if low == "reset":
                        get_rep_counter().reset(sid)
                        last_result[0] = None
                        await ws.send_json({"type": "control", "status": "reset"})
                    elif low == "close":
                        stop_event.set()
                        break
                    elif low.startswith("hint:"):
                        new_hint = text.split(":", 1)[1].strip() or None
                        current_hint[0] = new_hint
                        await ws.send_json({"type": "control",
                                            "status": "hint_set",
                                            "exercise_hint": new_hint})
                    continue

                frame_bytes = msg.get("bytes")
                if not frame_bytes:
                    continue

                frame_count += 1

                # Drop oldest frame if YOLO hasn't consumed it yet — we only
                # ever want the freshest frame in the slot.
                if frame_slot.full():
                    try:
                        frame_slot.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_slot.put(frame_bytes)

        except WebSocketDisconnect:
            stop_event.set()
        except Exception as e:
            logger.debug(f"recv_task exit: {e}")
            stop_event.set()

    # ── Task 2: YOLO inference (runs in thread-pool, never blocks the loop) ─
    async def infer_task():
        nonlocal infer_count
        loop = asyncio.get_event_loop()
        try:
            while not stop_event.is_set():
                # Wait up to 1 s for a frame; if nothing arrives just loop
                try:
                    frame_bytes = await asyncio.wait_for(
                        frame_slot.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Run YOLO in the default thread-pool executor so it doesn't
                # block the event loop (other tasks keep running).
                hint = current_hint[0]
                result = await loop.run_in_executor(
                    None,  # uses default ThreadPoolExecutor
                    pipe.analyze_frame, frame_bytes, sid, hint,
                )
                infer_count += 1

                payload = result.to_dict()
                payload.pop("keypoints", None)   # strip bulky raw keypoints
                payload["ui_fps"] = 25           # tell frontend target FPS

                last_result[0] = payload

                # Persist occasionally
                if result.detected and infer_count % persist_every == 0:
                    async with AsyncSessionLocal() as db:
                        await _persist_analysis(
                            db, user_id, sid, payload,
                            frame_count=persist_every,
                        )

        except Exception as e:
            logger.debug(f"infer_task exit: {e}")
            stop_event.set()

    # ── Task 3: heartbeat — sends last result at 25 FPS regardless of YOLO──
    async def heartbeat_task():
        try:
            while not stop_event.is_set():
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                r = last_result[0]
                if r is not None:
                    try:
                        await ws.send_json(r)
                    except Exception:
                        stop_event.set()
                        break
        except Exception as e:
            logger.debug(f"heartbeat_task exit: {e}")
            stop_event.set()

    # ── Run all three tasks concurrently ────────────────────────────────────
    try:
        await asyncio.gather(
            recv_task(),
            infer_task(),
            heartbeat_task(),
        )
    except Exception as e:
        logger.exception(f"WS stream error: {e}")
    finally:
        stop_event.set()
        logger.info(
            f"WS stream closed · sid={sid} · "
            f"frames_received={frame_count} frames_inferred={infer_count}"
        )
        try:
            await ws.close()
        except Exception:
            pass
