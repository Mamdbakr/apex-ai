"""
backend/services/chatbot_service.py
─────────────────────────────────────
Single entry point for chat. Picks the best available engine based on
which API keys are configured:

    PRIMARY    rag_coach (LangGraph + FAISS + tool-calling)
               needs: GROQ_API_KEY  (Llama 3.3 via Groq, free tier)
                      or GOOGLE_API_KEY (Gemini, free tier)

    FALLBACK   v9 RAG chatbot (OpenAI / Anthropic / Ollama)
               needs: OPENAI_API_KEY  or  ANTHROPIC_API_KEY
                      or a running Ollama daemon

If none of those is configured the service returns a clear, helpful
"please configure a key" reply — never crashes the request.

Both engines emit the same response dict, so the route layer doesn't
need to know which one served the call.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _has_rag_coach_keys() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _has_v9_keys() -> bool:
    """v9 engine works if any of OpenAI / Anthropic / Ollama are reachable."""
    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        return True
    if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
        return True
    # Ollama is the last-resort local option; we don't probe it here, only
    # return True if the user explicitly selected it.
    return os.environ.get("LLM_PROVIDER", "").lower() == "ollama"


_NO_KEYS_REPLY = (
    "I'd love to coach you, but no language-model keys are configured yet. "
    "Add GROQ_API_KEY (free at https://console.groq.com), or "
    "OPENAI_API_KEY / ANTHROPIC_API_KEY to your .env file and restart the server."
)


# ─── PRIMARY ENGINE: rag_coach (LangGraph) ───────────────────────────────────

class _RagCoachEngine:
    """Wraps backend.rag_coach for async use behind a uniform interface."""

    name = "rag_coach"

    def __init__(self):
        # Imports are local so Groq/Gemini libs are loaded only when used.
        from backend.rag_coach.graph import FitnessGraph
        from backend.rag_coach.memory import SessionMemory
        from backend.rag_coach.rag import RAGService

        docs_dir = os.environ.get("RAG_DOCS_DIR", "knowledge_base/raw")
        index_dir = os.environ.get("RAG_INDEX_DIR", "knowledge_base/faiss_index")

        self._memory = SessionMemory()
        self._rag = RAGService(docs_dir=docs_dir, index_dir=index_dir)
        # Loads prebuilt FAISS index from disk (or builds it on first call).
        self._rag.build_or_load_index(force_rebuild=False)
        self._graph = FitnessGraph(self._rag)

        provider = os.environ.get("RAG_LLM_PROVIDER", "groq").lower()
        if provider == "gemini":
            self._provider = "gemini"
            self._model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            self._provider = "groq"
            self._model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    async def chat(self, user_id: str, message: str, profile: dict) -> dict[str, Any]:
        # rag_coach is sync; offload to a thread so we don't block the event loop.
        import asyncio

        # update session memory with the latest profile
        self._memory.upsert_profile(user_id, profile)
        latest = self._memory.get_profile_dict(user_id)

        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None, lambda: self._graph.invoke(message=message, profile=latest)
        )

        # Map FAISS retrieved sources into apex chat schema
        sources: list[dict] = []
        try:
            hits = self._rag.search(message, top_k=4)
            for h in hits:
                src = os.path.basename(str(h.get("source", "")))
                snippet = str(h.get("text", ""))[:240]
                if len(str(h.get("text", ""))) > 240:
                    snippet += "…"
                sources.append({
                    "source": src or "kb",
                    "score": float(h.get("score", 0.0)),
                    "snippet": snippet,
                })
        except Exception:
            pass

        return {
            "reply": payload.get("response", "I could not generate a response."),
            "sources": sources,
            "model": self._model,
            "provider": self._provider,
            "calories": payload.get("calories"),
            "macros": payload.get("macros"),
            "workout_plan": payload.get("workout_plan", []),
            "meal_plan": payload.get("meal_plan", []),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    async def stream(self, user_id: str, message: str, profile: dict) -> AsyncIterator[str]:
        # LangGraph runs as a single transaction; we emit the final reply at once.
        result = await self.chat(user_id, message, profile)
        yield result["reply"]

    def clear(self, user_id: str) -> None:
        self._memory._profiles.pop(user_id, None)

    def stats(self) -> dict:
        return {
            "engine": self.name,
            "provider": self._provider,
            "model": self._model,
            "vector_store": "faiss",
            "kb_files": len(self._rag.list_source_files()),
        }


# ─── FALLBACK ENGINE: v9 RAG chatbot ─────────────────────────────────────────

class _V9Engine:
    name = "v9_rag"

    def __init__(self):
        from backend.chatbot.rag_chatbot import get_chatbot
        self._bot = get_chatbot()

    async def chat(self, user_id: str, message: str, profile: dict) -> dict[str, Any]:
        # v9 chatbot already handles user_id as int and accepts free-form user_data
        try:
            uid_int = int(user_id)
        except Exception:
            uid_int = 1
        result = await self._bot.chat(user_id=uid_int, message=message, user_data=profile)
        return {
            "reply": result.reply,
            "sources": result.sources,
            "model": result.model,
            "provider": result.provider,
            "calories": None,
            "macros": None,
            "workout_plan": [],
            "meal_plan": [],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    async def stream(self, user_id: str, message: str, profile: dict) -> AsyncIterator[str]:
        try:
            uid_int = int(user_id)
        except Exception:
            uid_int = 1
        async for tok in self._bot.stream(user_id=uid_int, message=message, user_data=profile):
            yield tok

    def clear(self, user_id: str) -> None:
        try:
            uid_int = int(user_id)
        except Exception:
            return
        from backend.chatbot.memory import get_memory
        import asyncio
        try:
            asyncio.get_event_loop().create_task(get_memory().reset(uid_int))
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "engine": self.name,
            "provider": self._bot.llm.provider,
            "model": self._bot.llm.model,
            "vector_store": "chromadb",
            "kb_size": self._bot.vstore.count(),
        }


# ─── NULL ENGINE: no keys configured ─────────────────────────────────────────

class _NullEngine:
    name = "none"

    async def chat(self, user_id, message, profile):
        return {
            "reply": _NO_KEYS_REPLY, "sources": [],
            "model": "none", "provider": "none",
            "calories": None, "macros": None,
            "workout_plan": [], "meal_plan": [],
            "prompt_tokens": 0, "completion_tokens": 0,
        }

    async def stream(self, user_id, message, profile):
        yield _NO_KEYS_REPLY

    def clear(self, user_id): pass

    def stats(self): return {"engine": self.name, "provider": "none", "model": "none"}


# ─── PUBLIC SERVICE ──────────────────────────────────────────────────────────

class ChatbotService:
    """Picks the best engine on first use and remembers the choice."""

    def __init__(self):
        self._engine: Optional[Any] = None
        self._engine_choice: str = "lazy"

    def _select_engine(self):
        if _has_rag_coach_keys():
            try:
                eng = _RagCoachEngine()
                logger.info(f"  ✅  Chatbot engine: rag_coach ({eng._provider}/{eng._model})")
                return eng
            except Exception as e:
                logger.warning(f"rag_coach init failed: {e} — trying v9 fallback")

        if _has_v9_keys():
            try:
                eng = _V9Engine()
                logger.info(f"  ✅  Chatbot engine: v9 ({eng._bot.llm.provider}/{eng._bot.llm.model})")
                return eng
            except Exception as e:
                logger.warning(f"v9 chatbot init failed: {e}")

        logger.warning("  ⚠️   Chatbot engine: NONE — no LLM keys configured")
        return _NullEngine()

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._select_engine()
            self._engine_choice = self._engine.name
        return self._engine

    async def chat(self, user_id: str, message: str, profile: dict) -> dict:
        return await self.engine.chat(user_id, message, profile)

    async def stream(self, user_id: str, message: str, profile: dict) -> AsyncIterator[str]:
        async for tok in self.engine.stream(user_id, message, profile):
            yield tok

    def clear_session(self, user_id: str) -> None:
        self.engine.clear(user_id)

    def stats(self) -> dict:
        return self.engine.stats()


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService()
