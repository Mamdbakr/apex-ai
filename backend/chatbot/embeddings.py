"""
backend/chatbot/embeddings.py
───────────────────────────────
Sentence-transformer embeddings for RAG. Loaded once, reused for every query.

Default model (all-MiniLM-L6-v2): 384-dim, ~80MB, ~10ms per query on CPU.
Override via env EMBEDDING_MODEL.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from loguru import logger

from backend.core.config import settings


class EmbeddingModel:
    """Thin wrapper with batching and normalisation."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready · dim={self.dim}")

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.astype(np.float32).tolist() for v in vecs]


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingModel:
    s = settings()
    return EmbeddingModel(s.EMBEDDING_MODEL, device=s.EMBEDDING_DEVICE)
