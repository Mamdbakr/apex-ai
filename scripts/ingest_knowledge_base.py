"""
scripts/ingest_knowledge_base.py
──────────────────────────────────
Ingest markdown files from ./knowledge_base/ into the Chroma vector store.

Chunks by H2 sections (## heading), falls back to ~800-char sliding windows
for any section longer than the limit. Idempotent — upserts by stable id.

CLI:
    python -m scripts.ingest_knowledge_base                # ingest everything
    python -m scripts.ingest_knowledge_base --reset        # wipe first
    python -m scripts.ingest_knowledge_base --path ./docs  # custom source dir
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from loguru import logger

# allow `python -m scripts.ingest_knowledge_base` and plain `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.chatbot.vector_store import get_vector_store
from backend.core.logging import setup_logging


MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 120
SLIDING_OVERLAP = 150


def _sliding_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Fallback when an H2 section is too long."""
    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def chunk_markdown(content: str) -> list[str]:
    """Split a markdown file on H2 boundaries; fall back to sliding windows."""
    # Split on lines starting with "## " but keep the heading with its section
    parts = re.split(r"(?m)^(?=## )", content)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < MIN_CHUNK_CHARS:
            continue
        if len(part) <= MAX_CHUNK_CHARS:
            out.append(part)
        else:
            out.extend(_sliding_chunks(part, MAX_CHUNK_CHARS, SLIDING_OVERLAP))
    return out


def stable_id(source: str, chunk_idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}::{chunk_idx}::{digest}"


def ingest(kb_path: Path, reset: bool = False) -> dict:
    store = get_vector_store()
    if reset:
        logger.warning("Resetting vector store …")
        store.reset()

    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base path does not exist: {kb_path}")

    md_files = sorted(kb_path.rglob("*.md"))
    if not md_files:
        logger.warning(f"No .md files found under {kb_path}")
        return {"files": 0, "chunks": 0}

    total_chunks = 0
    for md in md_files:
        content = md.read_text(encoding="utf-8")
        chunks = chunk_markdown(content)
        if not chunks:
            logger.info(f"  · {md.name}: no usable chunks, skipping")
            continue

        source = md.stem
        ids = [stable_id(source, i, c) for i, c in enumerate(chunks)]
        metas = [{"source": source, "path": str(md.relative_to(kb_path))} for _ in chunks]

        store.upsert(ids, chunks, metas)
        total_chunks += len(chunks)
        logger.info(f"  · {md.name}: upserted {len(chunks)} chunks")

    logger.info(f"✅ Ingestion complete · files={len(md_files)} · chunks={total_chunks} · "
                f"collection size={store.count()}")
    return {"files": len(md_files), "chunks": total_chunks, "collection_size": store.count()}


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="./knowledge_base",
                    help="Folder containing markdown files (default: ./knowledge_base)")
    ap.add_argument("--reset", action="store_true",
                    help="Delete all vectors before ingesting")
    args = ap.parse_args()

    ingest(Path(args.path).resolve(), reset=args.reset)


if __name__ == "__main__":
    main()
