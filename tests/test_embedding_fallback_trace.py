from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.config import Settings
from backend.app.llm_clients import (
    LOCAL_FALLBACK_EMBEDDING_PROVIDER,
    LOCAL_HASH_EMBEDDING_PROVIDER,
    ModelClients,
)


class RetrievalTraceHarness(AgentRetrievalMixin):
    def __init__(self, provider_by_id: dict[str, str] | None = None) -> None:
        provider_by_id = provider_by_id or {}
        self.store = SimpleNamespace(
            get_document=lambda document_id: SimpleNamespace(
                embedding_model=provider_by_id.get(
                    document_id,
                    LOCAL_FALLBACK_EMBEDDING_PROVIDER if document_id == "fallback-doc" else "online-embedding",
                )
            )
        )


class EmbeddingFallbackTraceTests(unittest.TestCase):
    def test_query_embedding_fallback_reports_provider_and_reason(self) -> None:
        clients = ModelClients(
            Settings(
                api_key=None,
                embedding_api_key=None,
                default_embedding_model="online-embedding",
            )
        )

        result = clients.embed_query_with_info("test query", model="online-embedding")

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.provider, LOCAL_FALLBACK_EMBEDDING_PROVIDER)
        self.assertTrue(result.vector)
        self.assertIn("RuntimeError", result.fallback_reason)

    def test_explicit_local_hash_embedding_is_not_fallback(self) -> None:
        clients = ModelClients(Settings(default_embedding_model=LOCAL_HASH_EMBEDDING_PROVIDER))

        result = clients.embed_query_with_info("test query", model=LOCAL_HASH_EMBEDDING_PROVIDER)

        self.assertFalse(result.used_fallback)
        self.assertEqual(result.provider, LOCAL_HASH_EMBEDDING_PROVIDER)
        self.assertTrue(result.vector)

    def test_document_fallback_is_reflected_in_embedding_trace(self) -> None:
        harness = RetrievalTraceHarness()

        trace = harness._build_embedding_trace(
            requested_model="online-embedding",
            document_ids=["fallback-doc", "normal-doc"],
            query_events=[
                {
                    "provider": LOCAL_FALLBACK_EMBEDDING_PROVIDER,
                    "used_fallback": True,
                    "fallback_reason": "目标文档使用本地备用检索索引，查询向量需保持同一向量空间。",
                }
            ],
        )

        self.assertTrue(trace["embedding_used_fallback"])
        self.assertEqual(trace["embedding_provider"], LOCAL_FALLBACK_EMBEDDING_PROVIDER)
        self.assertEqual(trace["embedding_document_fallback_count"], 1)
        self.assertIn("fallback-doc", trace["embedding_document_providers"])
        self.assertTrue(trace["embedding_fallback_reason"])

    def test_indexed_local_hash_provider_is_comparable(self) -> None:
        harness = RetrievalTraceHarness({"local-doc": LOCAL_HASH_EMBEDDING_PROVIDER})

        resolved_model = harness._resolve_query_embedding_model(
            requested_model="online-embedding",
            document_ids=["local-doc"],
        )
        trace = harness._build_embedding_trace(
            requested_model="online-embedding",
            document_ids=["local-doc"],
            query_events=[
                {
                    "provider": resolved_model,
                    "used_fallback": False,
                    "fallback_reason": "",
                }
            ],
        )

        self.assertEqual(resolved_model, LOCAL_HASH_EMBEDDING_PROVIDER)
        self.assertFalse(trace["embedding_used_fallback"])
        self.assertEqual(trace["embedding_provider"], LOCAL_HASH_EMBEDDING_PROVIDER)


if __name__ == "__main__":
    unittest.main()
