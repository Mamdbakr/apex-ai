from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGService:
    """Loads system documents, chunks them, and serves retrieval with FAISS."""

    def __init__(
        self,
        docs_dir: str = "knowledge_base/raw",
        index_dir: str = "knowledge_base/faiss_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=600,
        )
        self.vstore: FAISS | None = None

    def _load_txt(self, path: Path) -> List[Document]:
        return TextLoader(str(path), encoding="utf-8").load()

    def _load_pdf(self, path: Path) -> List[Document]:
        return PyPDFLoader(str(path)).load()

    def _load_json(self, path: Path) -> List[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))
        docs: List[Document] = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                docs.append(
                    Document(
                        page_content=json.dumps(item, ensure_ascii=False),
                        metadata={"source": str(path), "idx": idx},
                    )
                )
        elif isinstance(data, dict):
            docs.append(
                Document(
                    page_content=json.dumps(data, ensure_ascii=False),
                    metadata={"source": str(path), "idx": 0},
                )
            )
        else:
            docs.append(
                Document(page_content=str(data), metadata={"source": str(path), "idx": 0})
            )
        return docs

    def load_documents(self) -> List[Document]:
        docs: List[Document] = []
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.docs_dir.glob("*.txt"):
            docs.extend(self._load_txt(file_path))
        for file_path in self.docs_dir.glob("*.pdf"):
            docs.extend(self._load_pdf(file_path))
        for file_path in self.docs_dir.glob("*.json"):
            docs.extend(self._load_json(file_path))

        for doc in docs:
            doc.metadata = doc.metadata or {}
            doc.metadata.setdefault("source", "unknown")
        return docs

    def build_or_load_index(self, force_rebuild: bool = False) -> int:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if not force_rebuild and faiss_file.exists() and pkl_file.exists():
            self.vstore = FAISS.load_local(
                str(self.index_dir),
                self.embedder,
                allow_dangerous_deserialization=True,
            )
            return 0

        docs = self.load_documents()
        if not docs:
            self.vstore = None
            return 0

        chunks = self.splitter.split_documents(docs)
        self.vstore = FAISS.from_documents(chunks, self.embedder)
        self.vstore.save_local(str(self.index_dir))
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, object]]:
        if self.vstore is None:
            return []
        hits = self.vstore.similarity_search_with_score(query, k=top_k)
        results: List[Dict[str, object]] = []
        for doc, score in hits:
            results.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )
        return results

    def list_source_files(self) -> List[str]:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            [p.name for p in self.docs_dir.glob("*.pdf")]
            + [p.name for p in self.docs_dir.glob("*.txt")]
            + [p.name for p in self.docs_dir.glob("*.json")]
        )
        return files
