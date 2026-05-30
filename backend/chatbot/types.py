"""
backend/chatbot/types.py
──────────────────────────
Shared, dependency-free data shapes used across the chatbot package.

Kept in its own module so lightweight consumers (memory, prompts, tests)
don't pull in httpx / openai / anthropic just to get a dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResult:
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class LLMError(Exception):
    """Raised when every provider attempt fails."""
