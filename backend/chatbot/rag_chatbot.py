"""
backend/chatbot/rag_chatbot.py
────────────────────────────────
RAG-augmented conversational fitness coach.

Flow:
  1. Accept (user_id, message, user_data).
  2. Pull recent conversation from MemoryManager.
  3. Embed the message and retrieve top-k chunks from the vector store.
  4. Build a structured user turn: [USER_CONTEXT] + [KNOWLEDGE] + [USER_QUESTION].
  5. Call the LLM (OpenAI / Anthropic / Ollama — decided by settings).
  6. Append both turns to memory; maybe summarise older turns.
  7. Return reply + citations + metadata.

Provider failure (network, rate-limit, missing key) does NOT crash the
request — a graceful fallback message is returned instead, and the error
is logged with enough context to debug.

No rule-based branching, no keyword dispatch — everything is LLM-generated.
This replaces the old rule_engine.py entirely.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from loguru import logger

from backend.chatbot.llm_provider import get_llm
from backend.chatbot.types import ChatMessage
from backend.chatbot.memory import get_memory
from backend.chatbot.prompts import (
    SYSTEM_PROMPT_COACH,
    build_rag_user_message,
)
from backend.chatbot.vector_store import get_vector_store


# ─── CONFIG ───────────────────────────────────────────────────────────────────

RAG_TOP_K = 4
RAG_MIN_SCORE = 0.25        # drop chunks below this cosine similarity
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 700

FALLBACK_REPLY = (
    "I'm having trouble reaching the language model right now. "
    "In the meantime — tell me your weight, height, age, and main goal "
    "and I'll work out your calorie and protein targets the moment I'm back."
)


# ─── RESPONSE SHAPE ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    reply: str
    sources: list[dict]         # list of {"source": str, "score": float, "snippet": str}
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

class RAGChatbot:
    def __init__(self):
        self.llm = get_llm()
        self.memory = get_memory()
        self.vstore = get_vector_store()

    # ── retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, message: str) -> list[dict]:
        try:
            raw = self.vstore.query(message, k=RAG_TOP_K)
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []
        return [r for r in raw if r.get("score", 0) >= RAG_MIN_SCORE]

    # ── message assembly ─────────────────────────────────────────────────────

    async def _build_messages(
        self,
        user_id: int,
        message: str,
        user_data: Optional[dict],
        kb: list[dict],
    ) -> list[ChatMessage]:
        history = await self.memory.get_history(user_id)
        user_turn = build_rag_user_message(message, user_data, kb)
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT_COACH),
            *history,
            ChatMessage(role="user", content=user_turn),
        ]

    @staticmethod
    def _sources_payload(kb: list[dict]) -> list[dict]:
        return [
            {
                "source": s.get("metadata", {}).get("source", "kb"),
                "score": round(float(s.get("score", 0.0)), 3),
                "snippet": s["text"][:240] + ("…" if len(s["text"]) > 240 else ""),
            }
            for s in kb
        ]

    # ── public: full (non-streaming) response ────────────────────────────────

    async def chat(
        self,
        user_id: int,
        message: str,
        user_data: Optional[dict] = None,
    ) -> RAGResponse:
        message = (message or "").strip()
        if not message:
            return RAGResponse(
                reply="Type a message and I'll coach you through it.",
                sources=[], model=self.llm.model, provider=self.llm.provider,
                prompt_tokens=0, completion_tokens=0,
            )

        kb = self._retrieve(message)
        messages = await self._build_messages(user_id, message, user_data, kb)

        try:
            result = await self.llm.acomplete(
                messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            reply = result.content.strip()
        except Exception as e:
            logger.exception(f"LLM completion failed: {e}")
            return RAGResponse(
                reply=FALLBACK_REPLY, sources=self._sources_payload(kb),
                model=self.llm.model, provider=self.llm.provider,
                prompt_tokens=0, completion_tokens=0,
            )

        # persist to memory (fire-and-forget the summariser)
        await self.memory.append(user_id, "user", message)
        await self.memory.append(user_id, "assistant", reply)
        asyncio.create_task(self.memory.maybe_summarise(user_id, self.llm))

        return RAGResponse(
            reply=reply,
            sources=self._sources_payload(kb),
            model=result.model,
            provider=result.provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    # ── public: streaming response ───────────────────────────────────────────

    async def stream(
        self,
        user_id: int,
        message: str,
        user_data: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        message = (message or "").strip()
        if not message:
            yield "Type a message and I'll coach you through it."
            return

        kb = self._retrieve(message)
        messages = await self._build_messages(user_id, message, user_data, kb)

        collected: list[str] = []
        try:
            async for tok in self.llm.acomplete_stream(
                messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            ):
                collected.append(tok)
                yield tok
        except Exception as e:
            logger.exception(f"LLM stream failed: {e}")
            yield FALLBACK_REPLY
            return

        full = "".join(collected).strip()
        if full:
            await self.memory.append(user_id, "user", message)
            await self.memory.append(user_id, "assistant", full)
            asyncio.create_task(self.memory.maybe_summarise(user_id, self.llm))


# ─── singleton ────────────────────────────────────────────────────────────────

_chatbot: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot()
    return _chatbot
