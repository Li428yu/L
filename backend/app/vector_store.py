from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from backend.app.document_processing import ChunkRecord
from backend.app.models import EvidenceItem


class ChromaPaperStore:
    def __init__(self, persist_dir: Path, collection_name: str = "paper_chunks") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on installed env
            raise RuntimeError(
                "缺少 chromadb 依赖。请先安装 requirements.txt，再启动 FastAPI 后端。"
            ) from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        return "Chroma PersistentClient"

    def add_chunks(
        self,
        *,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "paper_name": chunk.paper_name,
                    "page": chunk.page,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "source": chunk.source,
                    "file_hash": chunk.file_hash,
                    "quote": chunk.quote,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "parent_id": chunk.parent_id,
                    "parent_text": chunk.parent_text,
                    "parent_page_start": chunk.parent_page_start,
                    "parent_page_end": chunk.parent_page_end,
                    "parent_char_start": chunk.parent_char_start,
                    "parent_char_end": chunk.parent_char_end,
                    "parent_token_count": chunk.parent_token_count,
                    "image_id": chunk.image_id,
                    "image_path": chunk.image_path,
                    "bbox_json": chunk.bbox_json,
                    "embedding_model": embedding_model,
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        evidence: list[EvidenceItem] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            evidence.append(
                EvidenceItem(
                    citation_id=f"E{index + 1}",
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    document_id=str(metadata.get("document_id", "")),
                    paper_name=str(metadata.get("paper_name", "")),
                    page=int(metadata.get("page", 0) or 0),
                    page_start=int(metadata.get("page_start", metadata.get("page", 0)) or 0),
                    page_end=int(metadata.get("page_end", metadata.get("page", 0)) or 0),
                    section=str(metadata.get("section") or ""),
                    source=str(metadata.get("source", "")),
                    file_hash=str(metadata.get("file_hash", "")),
                    score=score,
                    vector_score=score,
                    final_score=score,
                    score_source="vector",
                    text=str(documents[index] if index < len(documents) else ""),
                    quote=str(metadata.get("quote", "")),
                    char_start=int(metadata.get("char_start", 0) or 0),
                    char_end=int(metadata.get("char_end", 0) or 0),
                    token_count=int(metadata.get("token_count", 0) or 0),
                    chunk_type=str(metadata.get("chunk_type") or "text"),
                    parent_id=str(metadata.get("parent_id") or "") or None,
                    image_id=str(metadata.get("image_id") or "") or None,
                    image_path=str(metadata.get("image_path") or "") or None,
                    bbox_json=str(metadata.get("bbox_json") or "") or None,
                )
            )
        return [self._expand_evidence_context(item) for item in evidence]

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
        if self.count() == 0:
            self._reset_collection()

    def _reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def get_document_chunks(self, document_id: str, limit: int = 50) -> list[dict[str, Any]]:
        result = self.collection.get(
            where={"document_id": document_id},
            limit=limit,
            include=["documents", "metadatas"],
        )
        rows: list[dict[str, Any]] = []
        for chunk_id, document, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            rows.append({"id": chunk_id, "text": document, "metadata": metadata or {}})
        return sorted(rows, key=lambda row: str(row["id"]))

    def _expand_evidence_context(self, item: EvidenceItem) -> EvidenceItem:
        if item.image_id or "image" in (item.chunk_type or ""):
            return item

        rows = self.get_document_chunks(item.document_id, limit=1000)
        if not rows:
            return item

        current_index = next(
            (
                index
                for index, row in enumerate(rows)
                if str(row["id"]) == item.chunk_id
                or str(row["metadata"].get("chunk_id", "")) == item.chunk_id
            ),
            -1,
        )
        if current_index < 0:
            return item

        current_row = rows[current_index]
        current_metadata = current_row["metadata"]
        parent_text = str(current_metadata.get("parent_text") or "").strip()
        if parent_text:
            item.text = parent_text
            item.quote = self._best_quote(str(current_row["text"]))
            item.page_start = int(current_metadata.get("parent_page_start", item.page_start or item.page) or item.page)
            item.page_end = int(current_metadata.get("parent_page_end", item.page_end or item.page) or item.page)
            item.char_start = int(current_metadata.get("parent_char_start", item.char_start or 0) or 0)
            item.char_end = int(current_metadata.get("parent_char_end", item.char_end or 0) or 0)
            token_count = int(current_metadata.get("parent_token_count", item.token_count or 0) or 0)
            item.token_count = token_count or item.token_count
            return item

        start = max(current_index - 1, 0)
        end = min(current_index + 2, len(rows))
        context_rows = rows[start:end]
        context_text = "\n\n".join(str(row["text"]).strip() for row in context_rows if row["text"])
        if not context_text:
            return item

        item.text = context_text
        item.quote = self._best_quote(str(rows[current_index]["text"]))
        return item

    def _best_quote(self, text: str, limit: int = 280) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit and not self._is_table_like_text(normalized):
            return normalized
        sentence_parts = []
        current = ""
        for char in normalized:
            current += char
            if char in "。！？.!?；;":
                sentence_parts.append(current.strip())
                current = ""
        if current.strip():
            sentence_parts.append(current.strip())

        quote = ""
        for sentence in sentence_parts:
            if self._is_table_like_text(sentence):
                continue
            if len(quote) + len(sentence) > limit:
                break
            quote += sentence
        return quote or normalized[:limit]

    def _is_table_like_text(self, text: str) -> bool:
        normalized = " ".join(text.split())
        pipe_count = normalized.count("|")
        digit_ratio = len(re.findall(r"\d", normalized)) / max(len(normalized), 1)
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
        return pipe_count >= 6 or (digit_ratio > 0.28 and cjk_count < 80 and len(normalized) > 100)
