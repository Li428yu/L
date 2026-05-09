from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from typing import Iterable

from docx import Document
import fitz
import httpx
import numpy as np
from openai import OpenAI


@dataclass
class Chunk:
    chunk_id: int
    page: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


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
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_id = 0

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
                            page=page_number,
                            text=chunk_text,
                        )
                    )
                    chunk_id += 1
                start += step

        return chunks

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
        chunks: list[Chunk],
        chunk_vectors: np.ndarray,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        query_vector = self.embed_query(query)
        chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        similarities = (chunk_vectors @ query_vector) / (
            np.maximum(chunk_norms * query_norm, 1e-12)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for index in top_indices:
            results.append(
                RetrievedChunk(
                    chunk=chunks[index],
                    score=float(similarities[index]),
                )
            )
        return results

    def answer_question(self, question: str, evidence: list[RetrievedChunk]) -> str:
        context_blocks = []
        for item in evidence:
            context_blocks.append(
                f"[Page {item.chunk.page} | Score {item.score:.3f}]\n{item.chunk.text}"
            )

        context = "\n\n".join(context_blocks)
        system_prompt = (
            "You are a paper reading assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, say what is missing. "
            "Give a concise answer first, then a short evidence section with page numbers."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Respond in Chinese."
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content or ""

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
            raise ValueError("多模态向量接口需要 API_KEY 和 API_BASE_URL。")

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
                        f"多模态向量接口请求失败，状态码 {response.status_code}，返回内容：{response.text}"
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

        raise ValueError(f"无法从多模态向量接口返回中解析 embedding。返回内容：{payload}")


def estimate_cost_hint(chunk_count: int, average_chars_per_chunk: int = 900) -> str:
    estimated_tokens = math.ceil(chunk_count * average_chars_per_chunk / 4)
    return f"本次建立检索大约会处理 {estimated_tokens} 个文本 token。"
