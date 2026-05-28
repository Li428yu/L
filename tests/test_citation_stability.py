from __future__ import annotations

import unittest

from backend.app.agent_parts.citation_stability import AgentCitationStabilityMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class CitationStabilityHarness(
    AgentCitationStabilityMixin,
    AgentRetrievalFilterMixin,
    AgentTextUtilityMixin,
):
    pass


def make_evidence(
    chunk_id: str,
    *,
    text: str,
    score: float = 0.7,
    page: int = 1,
    document_id: str = "doc-1",
    chunk_type: str = "text",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id="",
        chunk_id=chunk_id,
        document_id=document_id,
        paper_name="paper.pdf",
        page=page,
        page_start=page,
        page_end=page,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        final_score=score,
        text=text,
        quote=text,
        chunk_type=chunk_type,
    )


class CitationStabilityTests(unittest.TestCase):
    def test_no_document_specific_profile_reorders_generic_candidates(self) -> None:
        harness = CitationStabilityHarness()
        question = "What result does the paper report for the benchmark table?"
        selected_table = make_evidence(
            "selected-table",
            page=3,
            chunk_type="table",
            text="Table 1 reports benchmark accuracy of 84.2 and error rate of 15.8.",
        )
        alternate_table = make_evidence(
            "alternate-table",
            page=4,
            chunk_type="table",
            text="Table 2 reports latency and memory usage for a separate ablation.",
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[selected_table],
            candidates=[selected_table, alternate_table],
            limit=1,
        )

        self.assertEqual(selected[0].chunk_id, "selected-table")
        self.assertEqual(selected[0].quote, selected_table.quote)

    def test_no_document_specific_profile_preserves_document_coverage(self) -> None:
        harness = CitationStabilityHarness()
        question = "Compare the training setup across the uploaded papers."
        first_doc = make_evidence(
            "first-doc",
            document_id="doc-a",
            text="Paper A trains on a public corpus and reports validation accuracy.",
        )
        second_doc = make_evidence(
            "second-doc",
            document_id="doc-b",
            text="Paper B trains on sensor logs and reports validation accuracy.",
        )
        same_doc_candidate = make_evidence(
            "same-doc-candidate",
            document_id="doc-a",
            text="Paper A also includes a longer appendix about optimizer settings.",
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[first_doc, second_doc],
            candidates=[first_doc, second_doc, same_doc_candidate],
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in selected], ["first-doc", "second-doc"])
        self.assertEqual({item.document_id for item in selected}, {"doc-a", "doc-b"})


if __name__ == "__main__":
    unittest.main()
