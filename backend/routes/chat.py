"""
backend/routes/chat.py
────────────────────────
APEX AI v10 — Unified Chat API.

External contract (POST /chat, /chat/stream, /chat/history) is unchanged
from v9, but internally requests now go through ChatbotService which
prefers the LangGraph + FAISS + tool-calling engine when keys are present
and falls back to the v9 OpenAI/Anthropic/Ollama engine otherwise.

User identity priority: session cookie > nothing (auth is required).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import ChatHistory, UserProfile, get_db
from backend.middleware.auth_guard import get_current_user
from backend.services.chatbot_service import get_chatbot_service


router = APIRouter(prefix="/chat", tags=["AI Chatbot"])


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_data: Optional[dict] = None     # optional what-if profile overrides


class SourceSnippet(BaseModel):
    source: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[SourceSnippet] = Field(default_factory=list)
    model: str
    provider: str
    calories: Optional[int] = None
    macros: Optional[dict] = None
    workout_plan: list[str] = Field(default_factory=list)
    meal_plan: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _resolve_user_id(current_user: dict) -> int:
    """user_id is always taken from the authenticated session — never from the body."""
    return int(current_user["user_id"])


_ACTIVITY = {1: "sedentary", 2: "light", 3: "moderate", 4: "active", 5: "very_active"}


async def _load_profile(db: AsyncSession, user_id: int) -> dict:
    res = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    p = res.scalar_one_or_none()
    if not p:
        return {}
    return {
        "name": p.name,
        "age": p.age,
        "weight_kg": p.weight_kg,
        "weight": p.weight_kg,           # alias for rag_coach
        "height_cm": p.height_cm,
        "height": p.height_cm,           # alias for rag_coach
        "gender": "male" if p.gender == 1 else "female",
        "goal": p.goal,
        "activity_level": _ACTIVITY.get(int(p.activity_level or 2), "moderate"),
        "target_weight": p.target_weight,
    }


def _merge_profile(stored: dict, override: Optional[dict]) -> dict:
    out = dict(stored)
    if override:
        for k, v in override.items():
            if v not in (None, ""):
                out[k] = v
    return out


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = _resolve_user_id(current_user)
    profile = _merge_profile(await _load_profile(db, user_id), req.user_data)

    service = get_chatbot_service()
    result: dict[str, Any] = await service.chat(
        user_id=str(user_id),
        message=req.message,
        profile=profile,
    )

    # Persist for history (best-effort).
    try:
        now = datetime.utcnow()
        db.add_all([
            ChatHistory(user_id=user_id, role="user",
                        content=req.message, timestamp=now),
            ChatHistory(user_id=user_id, role="assistant",
                        content=result["reply"], model=result.get("model", ""),
                        timestamp=now),
        ])
        await db.commit()
    except Exception as e:
        logger.warning(f"Chat history persist failed: {e}")

    return ChatResponse(**result)


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    user_id = _resolve_user_id(current_user)
    profile = _merge_profile(await _load_profile(db, user_id), req.user_data)
    service = get_chatbot_service()

    async def event_gen():
        collected: list[str] = []
        nl = "\n"
        try:
            async for tok in service.stream(str(user_id), req.message, profile):
                collected.append(tok)
                safe = tok.replace(nl, "\\n")
                yield f"data: {safe}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.exception(f"Chat stream failed: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"
        try:
            reply = "".join(collected).strip()
            if reply:
                now = datetime.utcnow()
                db.add_all([
                    ChatHistory(user_id=user_id, role="user",
                                content=req.message, timestamp=now),
                    ChatHistory(user_id=user_id, role="assistant",
                                content=reply, model="stream", timestamp=now),
                ])
                await db.commit()
        except Exception:
            pass

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/history/{user_id}")
async def get_history(
    user_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if current_user and int(current_user["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Cannot read another user's history")

    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {"role": r.role, "content": r.content,
         "model": r.model, "timestamp": r.timestamp.isoformat()}
        for r in reversed(rows)
    ]


@router.delete("/history/{user_id}")
async def clear_history(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if current_user and int(current_user["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Cannot clear another user's history")
    await db.execute(delete(ChatHistory).where(ChatHistory.user_id == user_id))
    await db.commit()
    get_chatbot_service().clear_session(str(user_id))
    return {"status": "ok", "message": f"History cleared for user {user_id}"}


@router.get("/stats")
async def chat_stats():
    return get_chatbot_service().stats()
