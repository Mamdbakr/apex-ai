"""
backend/chatbot/vector_store.py
─────────────────────────────────
Chroma-backed vector store. Persistent on disk at settings().VECTOR_DB_PATH.

Public API:
    store = get_vector_store()
    store.upsert(ids, texts, metadatas)
    results = store.query(query_text, k=5)     # returns [{text, metadata, score}]
    store.count()
    store.reset()
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from backend.chatbot.embeddings import get_embedder
from backend.core.config import settings


class VectorStore:
    def __init__(self, persist_path: str, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedder()
        logger.info(
            f"Vector store ready · collection='{collection_name}' "
            f"path='{persist_path}' items={self._collection.count()}"
        )

    # ── Writes ───────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> int:
        if not texts:
            return 0
        embeddings = self._embedder.embed_batch(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in texts],
        )
        return len(ids)

    def reset(self) -> None:
        """Delete all vectors and recreate an empty collection."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Reads ────────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        if self._collection.count() == 0:
            return []
        q_emb = self._embedder.embed(text)
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, self._collection.count()),
            where=where,
        )
        if not res.get("documents") or not res["documents"][0]:
            return []
        docs = res["documents"][0]
        metas = res.get("metadatas", [[]])[0] or [{} for _ in docs]
        dists = res.get("distances", [[]])[0] or [0.0] * len(docs)
        return [
            {
                "text": d,
                "metadata": m,
                "score": float(1.0 - dist),   # cosine distance → similarity
            }
            for d, m, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    s = settings()
    return VectorStore(s.VECTOR_DB_PATH, s.VECTOR_COLLECTION)
