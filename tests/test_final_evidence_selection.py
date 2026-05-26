from __future__ import annotations

import unittest

from backend.app.agent_parts.answering import AgentAnsweringMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.retrieval_scoring import AgentRetrievalScoringMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class FinalEvidenceHarness(
    AgentAnsweringMixin,
    AgentRetrievalFilterMixin,
    AgentRetrievalScoringMixin,
    AgentTextUtilityMixin,
):
    def _looks_like_reference_question(self, question: str) -> bool:
        return False

    def _looks_like_field_lookup_question(self, question: str) -> bool:
        return False

    def _looks_like_broad_overview_question(self, question: str) -> bool:
        return False

    def _looks_like_overview_question(self, question: str) -> bool:
        return False

    def _looks_like_compare_question(self, question: str) -> bool:
        return False

    def _looks_like_multi_document_topic_question(self, question: str) -> bool:
        return False

    def _looks_like_framework_function_question(self, question: str) -> bool:
        normalized = question.lower()
        return "function" in normalized and ("csf" in normalized or "rmf" in normalized or "framework" in normalized)

    def _looks_like_trustworthy_characteristics_question(self, question: str) -> bool:
        normalized = question.lower()
        return "trustworthy" in normalized and "characteristic" in normalized

    def _looks_like_metric_result_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["bleu", "wmt", "table", "result"])

    def _looks_like_visual_retrieval_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["figure", "image", "visual", "chart"])


def make_evidence(
    chunk_id: str,
    *,
    citation_id: str = "",
    score: float = 0.8,
    text: str,
    chunk_type: str = "text",
    image_id: str | None = None,
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
        text=text,
        quote=text,
        chunk_type=chunk_type,
        image_id=image_id,
    )


class FinalEvidenceSelectionTests(unittest.TestCase):
    def test_framework_text_evidence_replaces_image_for_non_visual_question(self) -> None:
        harness = FinalEvidenceHarness()
        image = make_evidence(
            "image-core",
            score=0.95,
            chunk_type="figure_image",
            image_id="img-1",
            text="Fig. 2 shows CSF Functions: GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER.",
        )
        text = make_evidence(
            "text-core",
            score=0.82,
            text=(
                "The CSF Core Functions - GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER - "
                "organize cybersecurity outcomes at their highest level."
            ),
        )

        repaired = harness._repair_final_evidence_selection(
            question="What are the NIST CSF core functions?",
            selected=[image],
            candidates=[image, text],
            limit=1,
        )

        self.assertEqual([item.chunk_id for item in repaired], ["text-core"])

    def test_framework_repair_does_not_promote_partial_function_chunk(self) -> None:
        harness = FinalEvidenceHarness()
        image = make_evidence(
            "image-core",
            score=0.95,
            chunk_type="figure_image",
            image_id="img-1",
            text="Fig. 5 shows AI RMF Core Functions: GOVERN, MAP, MEASURE, and MANAGE.",
        )
        partial = make_evidence(
            "partial-manage",
            score=0.98,
            text="MANAGE 1.1 says AI systems should be assessed before deployment.",
        )

        repaired = harness._repair_final_evidence_selection(
            question="What are the AI RMF core functions?",
            selected=[image],
            candidates=[image, partial],
            limit=1,
        )

        self.assertEqual([item.chunk_id for item in repaired], ["image-core"])

    def test_framework_selector_requires_complete_function_set(self) -> None:
        harness = FinalEvidenceHarness()
        complete = make_evidence(
            "complete",
            score=0.7,
            text="AI RMF Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE.",
        )
        partial = make_evidence(
            "partial",
            score=0.99,
            text="MANAGE 1.1 says AI systems should be assessed before deployment.",
        )

        selected = harness._select_framework_function_evidence(
            question="What are the AI RMF core functions?",
            evidence=[partial, complete],
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in selected], ["complete"])

    def test_framework_selector_prefers_text_over_image_for_non_visual_question(self) -> None:
        harness = FinalEvidenceHarness()
        image = make_evidence(
            "image-core",
            score=0.95,
            chunk_type="figure_image",
            image_id="img-1",
            text="Fig. 2 shows CSF Functions: GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER.",
        )
        text = make_evidence(
            "text-core",
            score=0.8,
            text=(
                "The CSF Core Functions - GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER - "
                "organize cybersecurity outcomes at their highest level."
            ),
        )

        selected = harness._select_framework_function_evidence(
            question="What are the NIST CSF core functions?",
            evidence=[image, text],
            limit=2,
        )

        self.assertEqual(selected[0].chunk_id, "text-core")

    def test_metric_result_keeps_table_and_adds_explanatory_result_chunk(self) -> None:
        harness = FinalEvidenceHarness()
        table = make_evidence(
            "table-bleu",
            score=0.95,
            chunk_type="table",
            text="Table 2: Transformer (big) 28.4 EN-DE and 41.8 EN-FR BLEU.",
        )
        result_text = make_evidence(
            "result-bleu",
            score=0.72,
            text=(
                "On the WMT 2014 English-to-German task, Transformer (big) establishes "
                "a new state-of-the-art BLEU score of 28.4. On English-to-French it achieves 41.0 BLEU."
            ),
        )

        repaired = harness._repair_final_evidence_selection(
            question="What BLEU results does the Transformer report?",
            selected=[table],
            candidates=[table, result_text],
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in repaired], ["result-bleu", "table-bleu"])
        self.assertIn("28.4", repaired[0].quote)

    def test_prompt_evidence_prefers_direct_text_over_non_visual_image(self) -> None:
        harness = FinalEvidenceHarness()
        image = make_evidence(
            "image-core",
            citation_id="E1",
            score=0.95,
            chunk_type="figure_image",
            image_id="img-1",
            text="Fig. 2 shows CSF Functions: GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER.",
        )
        text = make_evidence(
            "text-core",
            citation_id="E2",
            score=0.82,
            text=(
                "The CSF Core Functions - GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER - "
                "organize cybersecurity outcomes at their highest level."
            ),
        )

        prompt_evidence = harness._select_model_prompt_evidence(
            question="What are the NIST CSF core functions?",
            evidence=[image, text, make_evidence("other", score=0.4, text="Appendix background.")],
            soft_intent={},
        )

        self.assertEqual(prompt_evidence[0].chunk_id, "text-core")

    def test_prompt_evidence_keeps_direct_support_even_when_noise_penalty_is_high(self) -> None:
        harness = FinalEvidenceHarness()
        harness._model_prompt_noise_penalty = lambda **kwargs: 2.0
        direct = make_evidence(
            "direct-core",
            citation_id="E1",
            score=0.8,
            text="AI RMF Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE.",
        )
        decoy = make_evidence(
            "decoy",
            citation_id="E2",
            score=0.95,
            text="General background about AI risk management.",
        )

        prompt_evidence = harness._select_model_prompt_evidence(
            question="What are the AI RMF core functions?",
            evidence=[decoy, direct, make_evidence("other", score=0.4, text="Appendix background.")],
            soft_intent={},
        )

        self.assertIn("direct-core", [item.chunk_id for item in prompt_evidence])


if __name__ == "__main__":
    unittest.main()
