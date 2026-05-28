from __future__ import annotations

import unittest

from backend.app.agent_parts.answering import AgentAnsweringMixin
from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class SnippetHarness(AgentAnsweringMixin, AgentRetrievalMixin, AgentTextUtilityMixin):
    def _looks_like_reference_question(self, question: str) -> bool:
        return False

    def _looks_like_document_wide_question(self, question: str) -> bool:
        return False

    def _looks_like_field_lookup_question(self, question: str) -> bool:
        return False

    def _looks_like_metric_result_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["result", "metric", "score", "accuracy", "error", "benchmark"])

    def _looks_like_table_question(self, question: str) -> bool:
        return "table" in question.lower()

    def _looks_like_visual_evidence_question(self, question: str) -> bool:
        return False


def make_evidence(*, text: str, quote: str = "", chunk_type: str = "text") -> EvidenceItem:
    return EvidenceItem(
        citation_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=0.9,
        text=text,
        quote=quote or text,
        chunk_type=chunk_type,
    )


class EvidenceSnippetTests(unittest.TestCase):
    def test_best_quote_prefers_direct_sentence_over_neighboring_context(self) -> None:
        harness = SnippetHarness()
        text = (
            "The introduction describes the motivation for the system. "
            "The method uses a lightweight encoder for sensor calibration. "
            "Experimental results show accuracy improves to 91 percent while error falls to 9 percent."
        )

        quote = harness._best_quote_for_question("What accuracy result is reported?", text)

        self.assertIn("accuracy improves to 91 percent", quote)
        self.assertNotIn("introduction describes", quote)

    def test_table_quote_keeps_header_and_relevant_rows(self) -> None:
        harness = SnippetHarness()
        table = "\n".join(
            [
                "| Model | Accuracy | Error |",
                "| --- | --- | --- |",
                "| Baseline | 80% | 20% |",
                "| Proposed | 91% | 9% |",
                "| Ablation | 86% | 14% |",
            ]
        )

        quote = harness._best_quote_for_question("What accuracy does the Proposed model report?", table)

        self.assertIn("Model | Accuracy | Error", quote)
        self.assertIn("Proposed | 91% | 9%", quote)
        self.assertNotIn("Baseline | 80%", quote)

    def test_model_prompt_compacts_table_to_row_level_evidence(self) -> None:
        harness = SnippetHarness()
        table = "\n".join(
            [
                "| Dataset | Score | Error |",
                "| --- | --- | --- |",
                "| Indoor | 72 | 28 |",
                "| Outdoor | 88 | 12 |",
                "| Synthetic | 65 | 35 |",
            ]
        )
        item = make_evidence(text=table, chunk_type="table")

        summary = harness._summarize_evidence_for_model_prompt(
            question="What score is reported for the Outdoor benchmark table row?",
            item=item,
        )

        self.assertIn("Dataset | Score | Error", summary)
        self.assertIn("Outdoor | 88 | 12", summary)
        self.assertNotIn("Synthetic | 65", summary)


if __name__ == "__main__":
    unittest.main()
