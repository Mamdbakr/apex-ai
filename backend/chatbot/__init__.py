"""APEX AI v8 — LLM + RAG chatbot package.

Imports are intentionally lazy: importing backend.chatbot.prompts or .memory
should not force httpx / openai / chroma to load. Call get_chatbot() when
you actually need the orchestrator.
"""
from __future__ import annotations


def get_chatbot():
    from backend.chatbot.rag_chatbot import get_chatbot as _g
    return _g()


__all__ = ["get_chatbot"]
