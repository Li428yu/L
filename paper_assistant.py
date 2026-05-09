from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from typing import Iterable, Sequence

from docx import Document
import fitz
import httpx
import numpy as np
from openai import OpenAI


@dataclass
class Chunk:
    chunk_id: int
    paper_id: str
    paper_name: str
    page: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class PaperOverview:
    paper_id: str
    paper_name: str
    page_count: int
    chunk_count: int


class PaperAssistant:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("API_BASE_URL")

        self.api_key = resolved_api_key
        self.base_url = (resolved_base_url or "").rstrip("/")
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

    def extract_text_from_file(self, file_name: str, file_bytes: bytes) -> list[tuple[int, str]]:
        lowered_name = file_name.lower()
        if lowered_name.endswith(".pdf"):
            return self.extract_text(file_bytes)
        if lowered_name.endswith(".docx"):
            return self.extract_text_from_docx(file_bytes)
        raise ValueError(f"暂不支持的文件类型：{file_name}")

    def extract_text(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((index, text))
        return pages

    def extract_text_from_docx(self, docx_bytes: bytes) -> list[tuple[int, str]]:
        document = Document(BytesIO(docx_bytes))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
        full_text = "\n".join(text for text in paragraphs if text)
        if not full_text:
            return []
        return [(1, full_text)]

    def chunk_pages(
        self,
        pages: Iterable[tuple[int, str]],
        chunk_size: int = 900,
        overlap: int = 180,
        paper_id: str = "paper-1",
        paper_name: str = "未命名论文",
        start_chunk_id: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_id = start_chunk_id

        for page_number, text in pages:
            start = 0
            step = max(chunk_size - overlap, 1)
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            paper_id=paper_id,
                            paper_name=paper_name,
                            page=page_number,
                            text=chunk_text,
                        )
                    )
                    chunk_id += 1
                start += step

        return chunks

    def list_papers(self, chunks: Sequence[Chunk]) -> list[PaperOverview]:
        paper_stats: dict[str, PaperOverview] = {}

        for chunk in chunks:
            overview = paper_stats.get(chunk.paper_id)
            if overview is None:
                paper_stats[chunk.paper_id] = PaperOverview(
                    paper_id=chunk.paper_id,
                    paper_name=chunk.paper_name,
                    page_count=chunk.page,
                    chunk_count=1,
                )
                continue

            overview.page_count = max(overview.page_count, chunk.page)
            overview.chunk_count += 1

        return list(paper_stats.values())

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        vectors = self._embed_texts(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self._embed_texts([query])
        return np.array(vectors[0], dtype=np.float32)

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        top_k: int = 4,
        paper_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks or len(chunk_vectors) == 0:
            return []

        eligible_indices = [
            index
            for index, chunk in enumerate(chunks)
            if paper_ids is None or chunk.paper_id in paper_ids
        ]
        if not eligible_indices:
            return []

        query_vector = self.embed_query(query)
        candidate_vectors = chunk_vectors[eligible_indices]
        chunk_norms = np.linalg.norm(candidate_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        similarities = (candidate_vectors @ query_vector) / (
            np.maximum(chunk_norms * query_norm, 1e-12)
        )

        top_positions = np.argsort(similarities)[::-1][: min(top_k, len(eligible_indices))]
        results: list[RetrievedChunk] = []
        for position in top_positions:
            chunk_index = eligible_indices[int(position)]
            results.append(
                RetrievedChunk(
                    chunk=chunks[chunk_index],
                    score=float(similarities[position]),
                )
            )
        return results

    def answer_question(self, question: str, evidence: Sequence[RetrievedChunk]) -> str:
        if not evidence:
            return "当前检索范围内没有找到可用证据。你可以扩大检索范围，或换一个更具体的问题。"

        context = self._format_retrieved_context(evidence)
        system_prompt = (
            "You are a careful paper reading assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, explicitly say what is missing. "
            "Respond in Chinese. "
            "Start with a concise answer, then add a section titled '证据依据'. "
            "Cite claims inline using the format [paper_name p.page]."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Please keep the answer grounded in the context."
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return (response.choices[0].message.content or "").strip()

    def generate_paper_digest(
        self,
        paper_name: str,
        chunks: Sequence[Chunk],
        max_chunks: int = 8,
    ) -> str:
        if not chunks:
            return "当前论文还没有可用内容，暂时无法生成摘要卡片。"

        representative_chunks = self._pick_representative_chunks(chunks, max_chunks=max_chunks)
        context = self._format_plain_chunks(representative_chunks)
        system_prompt = (
            "You are a careful paper reading assistant. "
            "Summarize only from the provided paper excerpts. "
            "If some information is missing from the excerpts, say so clearly instead of guessing. "
            "Respond in Chinese."
        )
        user_prompt = (
            f"请为论文《{paper_name}》生成一张结构化阅读卡片。\n"
            "输出格式请严格包含以下小标题：\n"
            "1. 一句话总结\n"
            "2. 研究问题\n"
            "3. 核心方法\n"
            "4. 主要贡献\n"
            "5. 实验与结果\n"
            "6. 局限性\n"
            "7. 推荐追问\n\n"
            "如果原文片段没有覆盖某部分，请直接写“原文片段未覆盖”。\n\n"
            f"论文片段：\n{context}"
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return (response.choices[0].message.content or "").strip()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if "embedding-vision" in self.embedding_model.lower():
            return self._embed_texts_with_multimodal_api(texts)

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            error_text = str(exc).lower()
            if (
                "does not support this api" in error_text
                or "invalidparameter" in error_text
                or "embeddings/multimodal" in error_text
            ):
                return self._embed_texts_with_multimodal_api(texts)
            raise

    def _embed_texts_with_multimodal_api(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key or not self.base_url:
            raise ValueError("多模态 embedding 接口需要同时配置 API_KEY 和 API_BASE_URL。")

        all_vectors: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                payload = {
                    "model": self.embedding_model,
                    "input": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }

                response = client.post(
                    f"{self.base_url}/embeddings/multimodal",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                )

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise ValueError(
                        "多模态 embedding 接口请求失败，"
                        f"状态码 {response.status_code}，返回内容：{response.text}"
                    ) from exc

                data = response.json()
                all_vectors.append(self._extract_embedding_from_multimodal_response(data))

        return all_vectors

    def _extract_embedding_from_multimodal_response(self, payload: dict) -> list[float]:
        candidates = []

        data_field = payload.get("data")
        if isinstance(data_field, list):
            candidates.extend(data_field)
        elif isinstance(data_field, dict):
            candidates.append(data_field)

        result_field = payload.get("result")
        if isinstance(result_field, dict):
            result_data = result_field.get("data")
            if isinstance(result_data, list):
                candidates.extend(result_data)
            elif isinstance(result_data, dict):
                candidates.append(result_data)

        for item in candidates:
            if not isinstance(item, dict):
                continue
            if "embedding" in item:
                return item["embedding"]
            if "dense" in item:
                return item["dense"]

        raise ValueError(f"无法从多模态 embedding 响应中解析向量：{payload}")

    def _format_retrieved_context(self, evidence: Sequence[RetrievedChunk]) -> str:
        context_blocks = []
        for item in evidence:
            context_blocks.append(
                f"[{item.chunk.paper_name} | Page {item.chunk.page} | Score {item.score:.3f}]\n"
                f"{item.chunk.text}"
            )
        return "\n\n".join(context_blocks)

    def _format_plain_chunks(self, chunks: Sequence[Chunk]) -> str:
        context_blocks = []
        for chunk in chunks:
            context_blocks.append(f"[{chunk.paper_name} | Page {chunk.page}]\n{chunk.text}")
        return "\n\n".join(context_blocks)

    def _pick_representative_chunks(
        self,
        chunks: Sequence[Chunk],
        max_chunks: int,
    ) -> list[Chunk]:
        if len(chunks) <= max_chunks:
            return list(chunks)
        if max_chunks <= 1:
            return [chunks[0]]

        positions = np.linspace(0, len(chunks) - 1, num=max_chunks, dtype=int)
        selected_chunks: list[Chunk] = []
        seen_positions: set[int] = set()
        for position in positions:
            resolved = int(position)
            if resolved in seen_positions:
                continue
            selected_chunks.append(chunks[resolved])
            seen_positions.add(resolved)
        return selected_chunks


def estimate_cost_hint(chunk_count: int, average_chars_per_chunk: int = 900) -> str:
    estimated_tokens = math.ceil(chunk_count * average_chars_per_chunk / 4)
    return f"本次建立索引预计会处理约 {estimated_tokens} 个文本 token。"
