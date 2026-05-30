"""
backend/chatbot/memory.py
───────────────────────────
Per-user conversation memory with:
  - Sliding window (last N turns kept verbatim)
  - Optional running summary of older turns (LLM-compressed — cheap)
  - Thread-safe in-memory store (swap for Redis in a multi-worker deploy)

Exposes the minimal API the chatbot service needs:
    mem = get_memory()
    history = await mem.get_history(user_id)     # list[ChatMessage]
    await mem.append(user_id, role, content)
    await mem.reset(user_id)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from time import time
from typing import Deque, Dict, List, Optional

from loguru import logger

from backend.chatbot.types import ChatMessage


# ─── CONFIG ───────────────────────────────────────────────────────────────────

MAX_TURNS = 12          # user+assistant turns kept verbatim (so 24 messages)
SUMMARY_TRIGGER = 24    # when history exceeds this, summarise older turns
SUMMARY_KEEP = 8        # keep this many recent turns unsummarised
SESSION_TTL_SECONDS = 60 * 60 * 6   # purge sessions idle > 6h


# ─── DATA ─────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    messages: Deque[ChatMessage] = field(default_factory=deque)
    summary: str = ""
    last_access: float = field(default_factory=time)


# ─── MEMORY MANAGER ───────────────────────────────────────────────────────────

class MemoryManager:
    """In-memory; swap the dict for Redis in production without touching callers."""

    def __init__(self):
        self._sessions: Dict[int, Session] = defaultdict(Session)
        self._locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _touch(self, user_id: int) -> Session:
        s = self._sessions[user_id]
        s.last_access = time()
        return s

    def purge_idle(self) -> int:
        cutoff = time() - SESSION_TTL_SECONDS
        stale = [uid for uid, s in self._sessions.items() if s.last_access < cutoff]
        for uid in stale:
            self._sessions.pop(uid, None)
        if stale:
            logger.info(f"Purged {len(stale)} idle chat sessions")
        return len(stale)

    # ── reads ────────────────────────────────────────────────────────────────

    async def get_history(self, user_id: int) -> List[ChatMessage]:
        s = self._touch(user_id)
        out: List[ChatMessage] = []
        if s.summary:
            out.append(ChatMessage(role="system",
                                   content=f"[CONVERSATION_SUMMARY]\n{s.summary}"))
        out.extend(list(s.messages))
        return out

    # ── writes ───────────────────────────────────────────────────────────────

    async def append(self, user_id: int, role: str, content: str) -> None:
        async with self._locks[user_id]:
            s = self._touch(user_id)
            s.messages.append(ChatMessage(role=role, content=content))
            # keep at most MAX_TURNS * 2 messages verbatim
            while len(s.messages) > MAX_TURNS * 2:
                s.messages.popleft()

    async def maybe_summarise(self, user_id: int, llm) -> None:
        """
        If history grew past the threshold, ask the LLM for a compressed summary
        of the older turns and drop them from the verbatim buffer.
        """
        s = self._sessions.get(user_id)
        if not s or len(s.messages) < SUMMARY_TRIGGER:
            return

        async with self._locks[user_id]:
            # re-check after acquiring lock
            if len(s.messages) < SUMMARY_TRIGGER:
                return
            old = [m for m in list(s.messages)[:-SUMMARY_KEEP]]
            keep = list(s.messages)[-SUMMARY_KEEP:]

            prompt = ChatMessage(
                role="user",
                content=(
                    "Summarise this fitness-coaching conversation in 4–6 short bullets. "
                    "Preserve: the user's goal, their stats, any programme you proposed, "
                    "and any constraints (injuries, preferences). Do not invent details.\n\n"
                    + "\n".join(f"{m.role.upper()}: {m.content}" for m in old)
                ),
            )
            try:
                result = await llm.acomplete(
                    [ChatMessage(role="system",
                                 content="You compress chat histories faithfully and concisely."),
                     prompt],
                    temperature=0.2, max_tokens=300,
                )
                s.summary = (s.summary + "\n" + result.content).strip() if s.summary else result.content
                s.messages = deque(keep)
                logger.debug(f"Summarised memory for user_id={user_id}")
            except Exception as e:
                logger.warning(f"Memory summarisation failed: {e}")

    async def reset(self, user_id: int) -> None:
        async with self._locks[user_id]:
            self._sessions.pop(user_id, None)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "total_messages": sum(len(s.messages) for s in self._sessions.values()),
        }


@lru_cache(maxsize=1)
def get_memory() -> MemoryManager:
    return MemoryManager()
