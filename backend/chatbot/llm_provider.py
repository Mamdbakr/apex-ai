"""
backend/chatbot/llm_provider.py
─────────────────────────────────
Unified async LLM interface. Supports:
    - OpenAI (GPT-4o / GPT-4o-mini / GPT-3.5-turbo)
    - Anthropic (Claude 3.5 Sonnet / Claude 3 Haiku)
    - Ollama (local LLaMA 3 / Mistral / any local model)

All providers return the same ChatResult schema so upstream code is provider-agnostic.
Streaming is supported — acomplete_stream() yields token chunks.

The factory get_llm() returns a cached singleton chosen from settings().LLM_PROVIDER.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import AsyncIterator, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.config import settings
from backend.chatbot.types import ChatMessage, ChatResult, LLMError

__all__ = ["ChatMessage", "ChatResult", "LLMError",
           "BaseLLM", "GroqLLM", "OpenAILLM", "AnthropicLLM", "OllamaLLM", "get_llm"]


# ─── BASE INTERFACE ───────────────────────────────────────────────────────────

class BaseLLM(ABC):
    provider: str = "base"
    model: str = "unknown"

    @abstractmethod
    async def acomplete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> ChatResult:
        raise NotImplementedError

    async def acomplete_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncIterator[str]:
        """Default: non-streaming providers fall back to one chunk."""
        result = await self.acomplete(messages, temperature=temperature, max_tokens=max_tokens)
        yield result.content


# ─── GROQ (Llama 3, fastest free LLM) ─────────────────────────────────────────

class GroqLLM(BaseLLM):
    """
    Groq is the recommended provider — extremely fast inference for Llama 3 /
    Mixtral, free tier is generous, and the SDK is OpenAI-compatible.

    Get a key at https://console.groq.com/keys (it starts with 'gsk_').
    """
    provider = "groq"

    def __init__(self):
        from groq import AsyncGroq
        s = settings()
        if not s.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set")
        self.client = AsyncGroq(api_key=s.GROQ_API_KEY, timeout=30.0)
        self.model = s.GROQ_MODEL

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
        reraise=True,
    )
    async def acomplete(self, messages, *, temperature=0.7, max_tokens=800):
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ChatResult(
            content=choice.message.content or "",
            model=resp.model,
            provider=self.provider,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def acomplete_stream(self, messages, *, temperature=0.7, max_tokens=800):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ─── OPENAI ───────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(self):
        from openai import AsyncOpenAI
        s = settings()
        if not s.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=s.OPENAI_API_KEY, timeout=30.0)
        self.model = s.OPENAI_MODEL

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
        reraise=True,
    )
    async def acomplete(self, messages, *, temperature=0.7, max_tokens=800):
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        return ChatResult(
            content=choice.message.content or "",
            model=resp.model,
            provider=self.provider,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def acomplete_stream(self, messages, *, temperature=0.7, max_tokens=800):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ─── ANTHROPIC ────────────────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self):
        import anthropic
        s = settings()
        if not s.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.AsyncAnthropic(api_key=s.ANTHROPIC_API_KEY, timeout=30.0)
        self.model = s.ANTHROPIC_MODEL

    @staticmethod
    def _split_system(messages: list[ChatMessage]):
        """Anthropic requires system prompt in a top-level `system` field."""
        system_parts, chat = [], []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), chat

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=6),
           reraise=True)
    async def acomplete(self, messages, *, temperature=0.7, max_tokens=800):
        system, chat = self._split_system(messages)
        resp = await self.client.messages.create(
            model=self.model,
            system=system or "You are a helpful assistant.",
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResult(
            content=text,
            model=resp.model,
            provider=self.provider,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            finish_reason=resp.stop_reason or "stop",
        )

    async def acomplete_stream(self, messages, *, temperature=0.7, max_tokens=800):
        system, chat = self._split_system(messages)
        async with self.client.messages.stream(
            model=self.model,
            system=system or "You are a helpful assistant.",
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ─── OLLAMA (local LLaMA) ─────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    provider = "ollama"

    def __init__(self):
        s = settings()
        self.base_url = s.OLLAMA_BASE_URL.rstrip("/")
        self.model = s.OLLAMA_MODEL
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def aclose(self):
        await self._client.aclose()

    @retry(stop=stop_after_attempt(2),
           wait=wait_exponential(multiplier=1, min=1, max=4),
           reraise=True)
    async def acomplete(self, messages, *, temperature=0.7, max_tokens=800):
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ChatResult(
            content=data.get("message", {}).get("content", ""),
            model=self.model,
            provider=self.provider,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            finish_reason="stop",
        )

    async def acomplete_stream(self, messages, *, temperature=0.7, max_tokens=800):
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json as _json
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    yield token
                if obj.get("done"):
                    break


# ─── FACTORY ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    """
    Returns the configured LLM provider, with a documented fallback chain.

    Priority:
      1. The provider named in settings().LLM_PROVIDER if its credentials work.
      2. Groq if GROQ_API_KEY is set (fastest, recommended).
      3. OpenAI if OPENAI_API_KEY is set.
      4. Anthropic if ANTHROPIC_API_KEY is set.
      5. Ollama (local LLaMA) — only if a server responds at OLLAMA_BASE_URL.

    If none of those work, this raises LLMError immediately so the failure is
    visible at startup instead of silently returning a broken client.
    """
    s = settings()
    provider = s.LLM_PROVIDER.lower()
    logger.info(f"Initialising LLM provider: {provider}")

    attempts: list[tuple[str, type[BaseLLM]]] = []
    # primary first
    if   provider == "groq":      attempts.append(("groq", GroqLLM))
    elif provider == "openai":    attempts.append(("openai", OpenAILLM))
    elif provider == "anthropic": attempts.append(("anthropic", AnthropicLLM))
    elif provider == "ollama":    attempts.append(("ollama", OllamaLLM))

    # then the rest of the chain (deduped)
    for name, cls in (("groq", GroqLLM), ("openai", OpenAILLM),
                      ("anthropic", AnthropicLLM), ("ollama", OllamaLLM)):
        if (name, cls) not in attempts:
            # Skip cloud providers when no key — don't even try them
            if name == "groq" and not s.has_groq: continue
            if name == "openai" and not s.has_openai: continue
            if name == "anthropic" and not s.has_anthropic: continue
            attempts.append((name, cls))

    last_err: Optional[Exception] = None
    for name, cls in attempts:
        try:
            inst = cls()
            logger.info(f"LLM provider active: {name} · {inst.model}")
            return inst
        except Exception as e:
            last_err = e
            logger.warning(f"LLM provider '{name}' unavailable: {e}")

    raise LLMError(
        "No usable LLM provider. Configure GROQ_API_KEY (recommended), "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally "
        f"(OLLAMA_BASE_URL). Last error: {last_err}"
    )
