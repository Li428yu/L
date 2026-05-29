from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


def make_evidence(chunk_id: str, *, document_id: str = "doc-1", score: float = 0.7) -> EvidenceItem:
    return EvidenceItem(
        citation_id="",
        chunk_id=chunk_id,
        document_id=document_id,
        paper_name=f"{document_id}.pdf",
        page=1,
        source=f"{document_id}.pdf",
        file_hash="hash",
        score=score,
        vector_score=score,
        final_score=score,
        score_source="vector",
        text=f"Evidence for {chunk_id}.",
        quote=f"Evidence for {chunk_id}.",
    )


class MultiQueryDenseHarness(AgentRetrievalMixin, AgentTextUtilityMixin):
    def __init__(self, *, multi_document: bool = False) -> None:
        self.multi_document = multi_document
        self.vector_calls: list[tuple[str, tuple[str, ...]]] = []
        self.vector_store = SimpleNamespace(get_document_chunks=lambda document_id, limit=1000: [])

    def _should_balance_multi_document_retrieval(self, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        return self.multi_document

    def _targeted_evidence_candidates(self, **kwargs) -> list[EvidenceItem]:  # type: ignore[no-untyped-def]
        return []

    def _bm25_sparse_evidence(self, **kwargs) -> list[EvidenceItem]:  # type: ignore[no-untyped-def]
        return []

    def _vector_similarity_evidence(self, **kwargs) -> list[EvidenceItem]:  # type: ignore[no-untyped-def]
        question = str(kwargs["question"])
        document_ids = tuple(kwargs["document_ids"])
        self.vector_calls.append((question, document_ids))
        return [make_evidence(f"{document_ids[0]}:{question}", document_id=document_ids[0])]

    def _rrf_fuse_evidence_candidates(self, *, candidate_lists, weights, limit):  # type: ignore[no-untyped-def]
        del weights
        return [item for candidates in candidate_lists for item in candidates][:limit]

    def _select_rrf_ranked_evidence(self, *, question, evidence, limit):  # type: ignore[no-untyped-def]
        del question
        return evidence[:limit]

    def _select_balanced_multi_document_evidence(self, *, question, evidence, target_document_ids, limit):  # type: ignore[no-untyped-def]
        del question, target_document_ids
        return evidence[:limit]

    def _build_embedding_trace(self, *, requested_model, document_ids, query_events):  # type: ignore[no-untyped-def]
        return {
            "embedding_requested_model": requested_model,
            "embedding_document_providers": {document_id: requested_model for document_id in document_ids},
            "query_event_count": len(query_events),
        }


class MultiQueryDenseRetrievalTests(unittest.TestCase):
    def test_hybrid_retrieval_runs_dense_for_expanded_queries(self) -> None:
        harness = MultiQueryDenseHarness()
        queries = [
            "What efficiency benefits are reported?",
            "What efficiency benefits are reported? trainable parameters memory",
            "What efficiency benefits are reported? experimental results benchmark",
        ]

        _, pipeline, _, _ = harness._hybrid_evidence(
            question=queries[0],
            retrieval_queries=queries,
            soft_intent={},
            document_ids=["doc-1"],
            top_k=2,
            embedding_model="remote-embedding",
            retrieval_strategy="hybrid_soft",
        )

        self.assertEqual(harness.vector_calls, [(query, ("doc-1",)) for query in queries])
        self.assertIn("multi_query_dense_vector", pipeline)

    def test_multi_document_retrieval_runs_dense_per_document_with_query_cap(self) -> None:
        harness = MultiQueryDenseHarness(multi_document=True)
        queries = ["base query", "expanded query", "role query", "extra query"]

        harness._hybrid_evidence(
            question=queries[0],
            retrieval_queries=queries,
            soft_intent={},
            document_ids=["doc-a", "doc-b"],
            top_k=2,
            embedding_model="remote-embedding",
            retrieval_strategy="hybrid_comparison",
        )

        expected = [
            ("base query", ("doc-a",)),
            ("expanded query", ("doc-a",)),
            ("role query", ("doc-a",)),
            ("base query", ("doc-b",)),
            ("expanded query", ("doc-b",)),
            ("role query", ("doc-b",)),
        ]
        self.assertEqual(harness.vector_calls, expected)


if __name__ == "__main__":
    unittest.main()
