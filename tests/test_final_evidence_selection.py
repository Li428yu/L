from __future__ import annotations

import unittest

from backend.app.agent_parts.answering import AgentAnsweringMixin
from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.retrieval_scoring import AgentRetrievalScoringMixin
from backend.app.agent_parts.state import AnswerPlan
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

    def _looks_like_document_wide_question(self, question: str) -> bool:
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

    def _looks_like_pretraining_objective_question(self, question: str) -> bool:
        normalized = question.lower()
        return "pretraining objective" in normalized or "pre-training objective" in normalized

    def _looks_like_dataset_or_scale_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["dataset", "scale", "数据", "规模"])

    def _looks_like_visual_retrieval_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["figure", "image", "visual", "chart"])

    def _renumber_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        for index, item in enumerate(evidence, start=1):
            item.citation_id = f"E{index}"
        return evidence


class MultiDocumentEvidenceHarness(FinalEvidenceHarness):
    def _looks_like_compare_question(self, question: str) -> bool:
        return "compare" in question.lower() or "比较" in question

    def _looks_like_multi_document_topic_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["gpt-2", "clip", "sam", "multi-document"])


class VisualQuestionHarness(AgentAnsweringMixin, AgentRetrievalMixin, AgentTextUtilityMixin):
    pass


def make_evidence(
    chunk_id: str,
    *,
    citation_id: str = "",
    score: float = 0.8,
    text: str,
    chunk_type: str = "text",
    image_id: str | None = None,
    document_id: str = "doc-1",
    paper_name: str = "paper.pdf",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id=document_id,
        paper_name=paper_name,
        page=1,
        source=paper_name,
        file_hash="hash",
        score=score,
        final_score=score,
        text=text,
        quote=text,
        chunk_type=chunk_type,
        image_id=image_id,
    )


class FinalEvidenceSelectionTests(unittest.TestCase):
    def test_visual_detector_keeps_explicit_architecture_image_question(self) -> None:
        harness = VisualQuestionHarness()
        question = "Transformer 架构图展示了哪些主要组件？请引用图片证据。"

        self.assertTrue(harness._looks_like_visual_evidence_question(question))
        self.assertTrue(harness._looks_like_visual_retrieval_question(question))

    def test_visual_detector_does_not_treat_image_text_dataset_scale_as_visual(self) -> None:
        harness = VisualQuestionHarness()
        question = "CLIP 使用了多大规模的图像-文本数据进行训练？"

        self.assertFalse(harness._looks_like_visual_evidence_question(question))
        self.assertFalse(harness._looks_like_visual_retrieval_question(question))

    def test_unsupported_medical_proof_question_gets_clear_refusal(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "csf-scope",
            citation_id="E1",
            text="The Cybersecurity Framework helps organizations manage cybersecurity risk.",
        )

        answer = harness._build_unsupported_proof_refusal_answer(
            question="Does NIST CSF 2.0 prove that a medical diagnosis model improved accuracy?",
            evidence=[evidence],
        )

        self.assertIn("不能证明", answer)
        self.assertIn("medical diagnosis", answer)
        self.assertIn("accuracy", answer)
        self.assertIn("[E1]", answer)

    def test_model_failure_for_unsupported_proof_returns_refusal_not_api_only(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "scanned-product",
            citation_id="E1",
            chunk_type="image",
            image_id="img-1",
            text="图片实际内容 这是 LinnSequencer 32 轨 MIDI 序列录音机的产品说明页。",
        )

        plan = harness._model_failure_answer_plan(
            {
                "question": "Can ocrmypdf_cardinal_scanned.pdf prove that GPT-2 achieved new SOTA on LAMBADA?",
                "needs_retrieval": True,
                "evidence": [evidence],
            },
            AnswerPlan(
                mode="model",
                answer_strategy="model_answer",
                runtime_detail="",
                final_prompt_evidence=["[E1] scanned.pdf p.1 score=1.000"],
                prompt_evidence=[evidence],
            ),
            RuntimeError("APIStatusError"),
        )

        self.assertEqual(plan.answer_strategy, "missing_evidence_refusal")
        self.assertTrue(plan.fallback_used)
        self.assertIn("不能证明", plan.local_answer)
        self.assertIn("LinnSequencer", plan.local_answer)
        self.assertIn("GPT-2", plan.local_answer)

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

    def test_final_repair_keeps_required_visual_candidate(self) -> None:
        harness = FinalEvidenceHarness()
        visual = make_evidence(
            "sam-figure",
            score=1.0,
            chunk_type="figure_image",
            image_id="img-1",
            text=(
                "Figure 4: Segment Anything Model (SAM) overview. "
                "promptable segmentation prompt image valid mask object masks."
            ),
        )
        selected = [
            make_evidence(f"text-{index}", score=1.0, text="Related segmentation background.")
            for index in range(5)
        ]

        repaired = harness._repair_final_evidence_selection(
            question="Segment Anything 的图片证据如何展示 promptable segmentation？",
            selected=selected,
            candidates=[visual, *selected],
            limit=5,
        )

        self.assertEqual(repaired[0].chunk_id, "sam-figure")
        self.assertTrue(any(item.chunk_id == "sam-figure" for item in repaired))

    def test_multi_document_coverage_restores_missing_target_document(self) -> None:
        harness = MultiDocumentEvidenceHarness()
        doc_a_1 = make_evidence(
            "gpt2-main",
            score=1.0,
            document_id="doc-gpt2",
            paper_name="gpt2.pdf",
            text="GPT-2 uses WebText and language modeling for pretraining.",
        )
        doc_a_2 = make_evidence(
            "gpt2-extra",
            score=0.98,
            document_id="doc-gpt2",
            paper_name="gpt2.pdf",
            text="GPT-2 evaluates zero-shot transfer with WebText language modeling.",
        )
        doc_b = make_evidence(
            "clip-main",
            score=0.88,
            document_id="doc-clip",
            paper_name="clip.pdf",
            text="CLIP uses image-text pairs and contrastive learning.",
        )
        doc_c = make_evidence(
            "sam-main",
            score=0.78,
            document_id="doc-sam",
            paper_name="sam.pdf",
            text="SAM uses promptable segmentation with prompts and masks.",
        )

        selected = harness._ensure_multi_document_coverage_if_needed(
            question="Compare GPT-2, CLIP, and SAM pretraining or model objectives.",
            selected=[doc_a_1, doc_a_2, doc_b],
            evidence=[doc_a_1, doc_a_2, doc_b, doc_c],
            target_document_ids=["doc-gpt2", "doc-clip", "doc-sam"],
            limit=3,
        )

        self.assertEqual(
            {"doc-gpt2", "doc-clip", "doc-sam"},
            {item.document_id for item in selected},
        )

    def test_multi_document_filter_uses_explicit_target_documents_for_final_coverage(self) -> None:
        harness = MultiDocumentEvidenceHarness()
        evidence = [
            make_evidence(
                "gpt2-main",
                score=1.0,
                document_id="doc-gpt2",
                paper_name="gpt2.pdf",
                text="GPT-2 uses WebText and language modeling for pretraining.",
            ),
            make_evidence(
                "gpt2-extra",
                score=0.99,
                document_id="doc-gpt2",
                paper_name="gpt2.pdf",
                text="GPT-2 scales language modeling for zero-shot transfer.",
            ),
            make_evidence(
                "clip-main",
                score=0.86,
                document_id="doc-clip",
                paper_name="clip.pdf",
                text="CLIP uses image-text pairs and contrastive learning.",
            ),
            make_evidence(
                "sam-main",
                score=0.77,
                document_id="doc-sam",
                paper_name="sam.pdf",
                text="SAM uses promptable segmentation with prompts and masks.",
            ),
        ]

        selected = harness._filter_evidence_for_question(
            "Compare GPT-2, CLIP, and SAM pretraining or model objectives.",
            evidence,
            top_k=3,
            target_document_ids=["doc-gpt2", "doc-clip", "doc-sam"],
        )

        self.assertEqual(
            {"doc-gpt2", "doc-clip", "doc-sam"},
            {item.document_id for item in selected},
        )
        self.assertEqual(
            [f"E{index}" for index in range(1, len(selected) + 1)],
            [item.citation_id for item in selected],
        )

    def test_local_framework_answer_uses_complete_core_evidence(self) -> None:
        harness = FinalEvidenceHarness()
        direct = make_evidence(
            "direct-core",
            citation_id="E2",
            score=0.8,
            text="AI RMF Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What are the AI RMF core functions?",
            evidence=[direct],
        )

        self.assertIn("GOVERN、MAP、MEASURE、MANAGE", answer)
        self.assertIn("[E2]", answer)

    def test_local_bleu_answer_requires_table_values(self) -> None:
        harness = FinalEvidenceHarness()
        table = make_evidence(
            "table-bleu",
            citation_id="E1",
            score=0.9,
            chunk_type="table",
            text="Table 2: Transformer (big) 28.4 EN-DE and 41.8 EN-FR BLEU.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What BLEU results are reported in the WMT table?",
            evidence=[table],
        )

        self.assertIn("28.4 BLEU", answer)
        self.assertIn("41.8 BLEU", answer)
        self.assertIn("[E1]", answer)

    def test_local_clip_dataset_answer_uses_exact_pair_scale(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "clip-scale",
            citation_id="E1",
            score=1.0,
            text="We constructed a new dataset of 400 million (image, text) pairs.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="CLIP 使用了多大规模的图像-文本数据进行训练？",
            evidence=[evidence],
        )

        self.assertIn("400 million (image, text) pairs", answer)
        self.assertIn("CLIP", answer)
        self.assertIn("[E1]", answer)

    def test_local_sam_visual_answer_uses_image_evidence(self) -> None:
        harness = FinalEvidenceHarness()
        visual = make_evidence(
            "sam-figure",
            citation_id="E1",
            score=1.0,
            chunk_type="figure_image",
            image_id="img-1",
            text=(
                "Figure 4: Segment Anything Model (SAM) overview. "
                "The task is promptable segmentation with a prompt image and valid mask."
            ),
        )

        answer = harness._build_local_structured_evidence_answer(
            question="Segment Anything 的图片证据如何展示 promptable segmentation？",
            evidence=[visual],
        )

        self.assertIn("promptable segmentation", answer)
        self.assertIn("valid mask", answer)
        self.assertIn("[E1]", answer)

    def test_local_scanned_cardinal_answer_uses_visual_summary(self) -> None:
        harness = FinalEvidenceHarness()
        visual = make_evidence(
            "cardinal-image",
            citation_id="E1",
            score=1.0,
            chunk_type="image",
            image_id="img-1",
            text="图片实际内容 这是 LinnSequencer 32轨 MIDI 序列录音机的产品说明页，文字清晰可识别。",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="扫描型 PDF `ocrmypdf_cardinal_scanned.pdf` 的图片内容是什么？",
            evidence=[visual],
        )

        self.assertIn("LinnSequencer 32 轨 MIDI", answer)
        self.assertIn("[E1]", answer)


if __name__ == "__main__":
    unittest.main()
