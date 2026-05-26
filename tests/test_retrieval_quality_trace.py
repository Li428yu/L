from __future__ import annotations

import unittest

from backend.app.agent_parts.retrieval_quality import AgentRetrievalQualityMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class RetrievalQualityHarness(AgentRetrievalQualityMixin, AgentTextUtilityMixin):
    pass


def make_evidence(
    chunk_id: str,
    *,
    citation_id: str = "",
    score: float = 0.5,
    text: str = "The paper proposes an attention mechanism for sequence modeling.",
    score_source: str = "rrf_fusion",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        final_score=score,
        score_source=score_source,
        text=text,
        quote=text,
    )


class RetrievalQualityTraceTests(unittest.TestCase):
    def test_selected_and_filtered_candidates_receive_explicit_statuses(self) -> None:
        harness = RetrievalQualityHarness()
        strong = make_evidence("strong", score=0.82)
        weak = make_evidence("weak", score=0.12, text="Unrelated appendix boilerplate.")

        selected, trace = harness._annotate_retrieval_quality(
            question="How does attention help sequence modeling?",
            candidates=[strong, weak],
            selected=[strong],
            top_k=1,
            retrieval_strategy="hybrid_soft",
        )

        self.assertEqual(selected[0].selection_status, "selected_by_retrieval_filter")
        self.assertEqual(selected[0].quality_label, "strong")
        self.assertEqual(trace[0]["selection_status"], "selected_by_retrieval_filter")
        self.assertEqual(trace[1]["selection_status"], "filtered_out")
        self.assertTrue(trace[1]["rejection_reason"])

    def test_evidence_judge_rejection_is_reflected_in_quality_trace(self) -> None:
        harness = RetrievalQualityHarness()
        item = make_evidence("candidate", citation_id="E1", score=0.8)
        _, trace = harness._annotate_retrieval_quality(
            question="How does attention help sequence modeling?",
            candidates=[item],
            selected=[item],
            top_k=1,
            retrieval_strategy="hybrid_soft",
        )

        merged = harness._merge_evidence_judgments_into_quality_trace(
            trace=trace,
            judgments=[
                {
                    "citation_id": "E1",
                    "chunk_id": "candidate",
                    "verdict": "reject",
                    "confidence": 0.9,
                    "reason": "forced reject",
                }
            ],
            kept=[],
        )

        self.assertEqual(merged[0]["selection_status"], "rejected_by_evidence_judge")
        self.assertEqual(merged[0]["judge_verdict"], "reject")
        self.assertIn("forced reject", merged[0]["rejection_reason"])

    def test_kept_evidence_is_marked_selected_for_answer(self) -> None:
        harness = RetrievalQualityHarness()
        item = make_evidence("candidate", citation_id="E1", score=0.8)
        _, trace = harness._annotate_retrieval_quality(
            question="How does attention help sequence modeling?",
            candidates=[item],
            selected=[item],
            top_k=1,
            retrieval_strategy="hybrid_soft",
        )
        item.citation_id = "E1"

        merged = harness._merge_evidence_judgments_into_quality_trace(
            trace=trace,
            judgments=[
                {
                    "citation_id": "E1",
                    "chunk_id": "candidate",
                    "verdict": "direct",
                    "confidence": 0.9,
                    "reason": "direct support",
                }
            ],
            kept=[item],
        )

        self.assertEqual(merged[0]["selection_status"], "selected_for_answer")
        self.assertEqual(item.selection_status, "selected_for_answer")
        self.assertEqual(item.quality_label, "strong")


if __name__ == "__main__":
    unittest.main()
