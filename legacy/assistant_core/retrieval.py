from __future__ import annotations

import time
from typing import Sequence

import httpx
import numpy as np

from assistant_core.types import Chunk, RetrievedChunk


class PaperRetrievalMixin:
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

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._should_use_multimodal_embeddings():
            return self._embed_texts_with_multimodal_api(texts)

        try:
            return self.embedding_client.embed_documents(texts)
        except Exception as exc:
            if self._should_fallback_to_multimodal(exc):
                return self._embed_texts_with_multimodal_api(texts)
            raise

    def _should_use_multimodal_embeddings(self) -> bool:
        return "embedding-vision" in self.embedding_model.lower()

    def _should_fallback_to_multimodal(self, exc: Exception) -> bool:
        error_text = str(exc).lower()
        return (
            "does not support this api" in error_text
            or "invalidparameter" in error_text
            or "embeddings/multimodal" in error_text
        )

    def _embed_texts_with_multimodal_api(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key or not self.base_url:
            raise ValueError("多模态 embedding 接口需要同时配置 API_KEY 和 API_BASE_URL。")

        all_vectors: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                response = self._request_multimodal_embedding_with_retry(
                    client=client,
                    text=text,
                )

                data = response.json()
                all_vectors.append(self._extract_embedding_from_multimodal_response(data))

        return all_vectors

    def _request_multimodal_embedding_with_retry(
        self,
        client: httpx.Client,
        text: str,
        max_attempts: int = 4,
    ) -> httpx.Response:
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.post(
                    f"{self.base_url}/embeddings/multimodal",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.embedding_model,
                        "input": [{"type": "text", "text": text}],
                    },
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                raise ValueError(
                    "多模态 embedding 接口请求失败："
                    f"状态码 {exc.response.status_code}，返回内容：{exc.response.text}"
                ) from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                time.sleep(min(0.6 * attempt, 1.8))

        raise ValueError(
            "连接 embedding 接口时出现临时网络抖动，已经自动重试多次但仍失败。"
            "这通常是上游 TLS/网络瞬时不稳定导致的，请稍后重试。"
        ) from last_error

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
