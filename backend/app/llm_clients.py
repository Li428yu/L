from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

import httpx
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.app.config import Settings


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    used_fallback: bool


class ModelClients:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat(self, model: str | None = None) -> ChatOpenAI:
        self._require_api_key()
        return ChatOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.api_base_url,
            model=model or self.settings.default_chat_model,
            temperature=0,
            timeout=self.settings.model_timeout_seconds,
            max_retries=self.settings.model_max_retries,
        )

    def embeddings(self, model: str | None = None) -> OpenAIEmbeddings:
        self._require_embedding_api_key()
        return OpenAIEmbeddings(
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_api_base_url,
            model=model or self.settings.default_embedding_model,
            timeout=self.settings.model_timeout_seconds,
            max_retries=self.settings.model_max_retries,
            retry_min_seconds=1,
            retry_max_seconds=1,
            check_embedding_ctx_length=False,
        )

    def chat_text(self, messages: list[BaseMessage], model: str | None = None) -> str:
        try:
            response = self.chat(model).invoke(messages)
        except Exception as exc:
            raise RuntimeError(
                "对话模型暂时不可用或响应太慢，已切换为基于原文证据的快速回答。"
            ) from exc
        return str(response.content).strip()

    def chat_text_stream(self, messages: list[BaseMessage], model: str | None = None):
        try:
            for chunk in self.chat(model).stream(messages):
                text = self._message_content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text
        except Exception as exc:
            raise RuntimeError(
                "对话模型流式响应失败或响应太慢。"
            ) from exc

    def embed_documents(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return self.embed_documents_with_info(texts, model=model).vectors

    def embed_documents_with_info(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], provider="none", used_fallback=False)
        try:
            client = self.embeddings(model)
            vectors = self._with_retry(lambda: client.embed_documents(texts))
            return EmbeddingResult(
                vectors=vectors,
                provider=model or self.settings.default_embedding_model,
                used_fallback=False,
            )
        except Exception as exc:
            if self._should_try_multimodal_embeddings(exc, model):
                try:
                    resolved_model = model or self.settings.default_embedding_model
                    return EmbeddingResult(
                        vectors=self._embed_texts_with_multimodal_api(texts, resolved_model),
                        provider=resolved_model,
                        used_fallback=False,
                    )
                except Exception:
                    pass
            return EmbeddingResult(
                vectors=[self._local_embedding(text) for text in texts],
                provider="本地备用检索",
                used_fallback=True,
            )

    def embed_query(self, text: str, model: str | None = None) -> list[float]:
        if model == "本地备用检索":
            return self._local_embedding(text)
        try:
            client = self.embeddings(model)
            return self._with_retry(lambda: client.embed_query(text))
        except Exception as exc:
            if self._should_try_multimodal_embeddings(exc, model):
                try:
                    resolved_model = model or self.settings.default_embedding_model
                    return self._embed_texts_with_multimodal_api([text], resolved_model)[0]
                except Exception:
                    pass
            return self._local_embedding(text)

    def _with_retry(self, fn):
        last_error: Exception | None = None
        attempts = max(self.settings.model_max_retries + 1, 1)
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - provider/network dependent
                last_error = exc
                if attempt == attempts:
                    break
                time.sleep(min(0.4 * attempt, 0.8))
        raise RuntimeError(
            "模型接口暂时不可用，已经自动重试但仍失败。请检查 API key、模型名、额度或网络。"
        ) from last_error

    def _require_api_key(self) -> None:
        if not self.settings.api_key:
            raise RuntimeError("缺少 API_KEY 或 OPENAI_API_KEY，无法调用模型服务。")

    def _require_embedding_api_key(self) -> None:
        if not self.settings.embedding_api_key:
            raise RuntimeError("缺少 EMBEDDING_API_KEY 或 API_KEY，无法调用 embedding 服务。")

    def _should_try_multimodal_embeddings(
        self,
        exc: Exception,
        model: str | None,
    ) -> bool:
        model_name = (model or self.settings.default_embedding_model).lower()
        cause = getattr(exc, "__cause__", None)
        error_text = f"{exc} {cause or ''}".lower()
        return (
            "embedding-vision" in model_name
            or "multimodal" in error_text
            or "does not support this api" in error_text
            or "invalidparameter" in error_text
        )

    def _embed_texts_with_multimodal_api(
        self,
        texts: list[str],
        model: str,
    ) -> list[list[float]]:
        if not self.settings.embedding_api_key or not self.settings.embedding_api_base_url:
            raise RuntimeError("缺少模型服务配置，无法调用多模态 embedding。")

        endpoint = f"{self.settings.embedding_api_base_url.rstrip('/')}/embeddings/multimodal"
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.settings.model_timeout_seconds) as client:
            for text in texts:
                response = client.post(
                    endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.settings.embedding_api_key}",
                    },
                    json={
                        "model": model,
                        "input": [{"type": "text", "text": text}],
                    },
                )
                response.raise_for_status()
                vectors.append(self._extract_embedding_from_payload(response.json()))
        return vectors

    def _extract_embedding_from_payload(self, payload: dict) -> list[float]:
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
            if isinstance(item.get("embedding"), list):
                return item["embedding"]
            if isinstance(item.get("dense"), list):
                return item["dense"]

        raise RuntimeError("多模态 embedding 接口没有返回可用向量。")

    def _message_content_to_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return str(content) if content else ""

    def _local_embedding(self, text: str, dimensions: int = 384) -> list[float]:
        vector = [0.0] * dimensions
        tokens = self._tokenize(text)
        if not tokens:
            tokens = [text[:64] or "empty"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]", lowered)
        cjk = [token for token in words if re.match(r"[\u4e00-\u9fff]", token)]
        if len(cjk) >= 2:
            words.extend("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
        return words[:1200]
