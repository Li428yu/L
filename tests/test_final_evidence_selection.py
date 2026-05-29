from __future__ import annotations

import unittest
from types import SimpleNamespace

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
        return "function" in normalized and any(term in normalized for term in ["framework", "model", "system"])

    def _looks_like_trustworthy_characteristics_question(self, question: str) -> bool:
        normalized = question.lower()
        return "trustworthy" in normalized and "characteristic" in normalized

    def _looks_like_metric_result_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(term in normalized for term in ["table", "result", "metric", "score", "accuracy", "error", "benchmark"])

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


class AnswerPlanHarness(FinalEvidenceHarness):
    settings = SimpleNamespace(force_model_answer=False)

    def _format_question_understanding_for_model(self, **kwargs) -> str:
        return "specific paper question"

    def _evidence_coverage_decision(self, **kwargs) -> dict:
        return {}

    def _should_decline_for_missing_direct_evidence(self, question: str, evidence: list[EvidenceItem]) -> bool:
        return False

    def _overview_focus_keywords(self, question: str) -> list[str]:
        return self._question_keywords(question)


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
            "scope",
            citation_id="E1",
            text="The uploaded document describes deployment requirements for a manufacturing sensor system.",
        )

        answer = harness._build_unsupported_proof_refusal_answer(
            question="Does this document prove that a medical diagnosis model improved accuracy?",
            evidence=[evidence],
        )

        self.assertIn("不能证明", answer)
        self.assertIn("证据", answer)
        self.assertIn("[E1]", answer)

    def test_model_failure_for_unsupported_proof_returns_refusal_not_api_only(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "scanned-product",
            citation_id="E1",
            chunk_type="image",
            image_id="img-1",
            text="图片实际内容 这是一页工业传感器安装手册，包含接线图和维护说明。",
        )

        plan = harness._model_failure_answer_plan(
            {
                "question": "Can this scanned product page prove that a medical diagnosis model improved accuracy?",
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
        self.assertIn("[E1]", plan.local_answer)

    def test_prepare_answer_plan_uses_model_for_normal_evidence_questions(self) -> None:
        harness = AnswerPlanHarness()
        evidence = make_evidence(
            "method-quote",
            citation_id="E1",
            score=1.0,
            text="The method architecture uses a lightweight encoder and a scoring head.",
        )

        plan = harness._prepare_answer_plan(
            {
                "question": "What method architecture does the approach use?",
                "needs_retrieval": True,
                "evidence": [evidence],
                "soft_intent": {"operation": "answer", "scope": "specific_point"},
                "retrieval_strategy": "hybrid_soft",
                "recent_messages": [],
                "memory_prompt": "",
            }
        )

        self.assertEqual(plan.mode, "model")
        self.assertEqual(plan.answer_strategy, "model_answer")
        self.assertIn("[E1]", plan.user_prompt)
        self.assertIn("关键术语", plan.user_prompt)

    def test_model_failure_uses_local_structured_answer_only_as_fallback(self) -> None:
        harness = AnswerPlanHarness()
        evidence = make_evidence(
            "method-quote",
            citation_id="E1",
            score=1.0,
            text="The method architecture uses a lightweight encoder and a scoring head.",
        )

        fallback = harness._model_failure_answer_plan(
            {
                "question": "What method architecture does the approach use?",
                "needs_retrieval": True,
                "evidence": [evidence],
            },
            AnswerPlan(
                mode="model",
                answer_strategy="model_answer",
                runtime_detail="",
                final_prompt_evidence=["[E1] paper.pdf p.1 score=1.000"],
                prompt_evidence=[evidence],
            ),
            RuntimeError("chat unavailable"),
        )

        self.assertEqual(fallback.answer_strategy, "local_fallback_answer")
        self.assertTrue(fallback.fallback_used)
        self.assertIn("[E1]", fallback.local_answer)

    def test_prepare_answer_plan_refuses_cross_scope_proof_claims(self) -> None:
        harness = AnswerPlanHarness()
        evidence = make_evidence(
            "source-scope",
            citation_id="E1",
            score=1.0,
            text=(
                "The method is designed to address dense object detection scenarios "
                "with severe class imbalance during training."
            ),
        )

        plan = harness._prepare_answer_plan(
            {
                "question": (
                    "Does this paper prove that low-rank adaptation reduces trainable parameters "
                    "in language model fine-tuning?"
                ),
                "needs_retrieval": True,
                "evidence": [evidence],
                "soft_intent": {"operation": "judge", "scope": "specific_point"},
                "retrieval_strategy": "hybrid_soft",
                "recent_messages": [],
                "memory_prompt": "",
            }
        )

        self.assertEqual(plan.answer_strategy, "missing_evidence_refusal")
        self.assertIn("不能证明", plan.local_answer)
        self.assertIn("low-rank adaptation", plan.local_answer)
        self.assertIn("language model", plan.local_answer)
        self.assertIn("object detection", plan.local_answer)

    def test_framework_text_evidence_replaces_image_for_non_visual_question(self) -> None:
        harness = FinalEvidenceHarness()
        image = make_evidence(
            "image-core",
            score=0.95,
            chunk_type="figure_image",
            image_id="img-1",
            text="Fig. 2 shows framework functions: PLAN, BUILD, REVIEW, and MONITOR.",
        )
        text = make_evidence(
            "text-core",
            score=0.82,
            text=(
                "The framework core functions - PLAN, BUILD, REVIEW, and MONITOR - "
                "organize implementation outcomes at their highest level."
            ),
        )

        repaired = harness._repair_final_evidence_selection(
            question="What are the framework core functions?",
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

    def test_prompt_evidence_dedupes_repeated_quotes_and_keeps_distinct_support(self) -> None:
        harness = FinalEvidenceHarness()
        duplicate_text = "The method uses a scoring module to select candidate outputs for the final decision."
        duplicate_a = make_evidence("duplicate-a", citation_id="E1", score=1.0, text=duplicate_text)
        duplicate_b = make_evidence("duplicate-b", citation_id="E2", score=0.99, text=duplicate_text)
        distinct = make_evidence(
            "distinct-method",
            citation_id="E3",
            score=0.74,
            text="The approach is designed to freeze base weights and add low rank matrix updates for adaptation.",
        )
        filler = make_evidence("filler", citation_id="E4", score=0.6, text="Additional background context.")

        prompt_evidence = harness._select_model_prompt_evidence(
            question="What method adaptation idea does the approach use?",
            evidence=[duplicate_a, duplicate_b, distinct, filler],
            soft_intent={},
        )
        chunk_ids = [item.chunk_id for item in prompt_evidence]

        self.assertIn("distinct-method", chunk_ids)
        self.assertLessEqual(len({"duplicate-a", "duplicate-b"} & set(chunk_ids)), 1)

    def test_resource_question_prefers_summary_claim_with_measured_benefits(self) -> None:
        harness = FinalEvidenceHarness()
        summary = make_evidence(
            "summary-benefits",
            score=0.86,
            text=(
                "Abstract We propose a parameter-efficient method that reduces trainable "
                "parameters by 10,000 times and the memory requirement by 3 times while "
                "keeping higher training throughput."
            ),
        )
        summary.section = "Abstract"
        decoy = make_evidence(
            "budget-table",
            score=1.0,
            text="Appendix table lists parameter budgets of 4.7M and 37.7M for several settings.",
        )
        decoy.section = "Methods"

        selected = harness._filter_evidence_for_question(
            "What efficiency benefits does the abstract report?",
            [decoy, summary],
            top_k=1,
        )

        self.assertEqual(selected[0].chunk_id, "summary-benefits")
        self.assertIn("10,000 times", selected[0].quote)
        self.assertIn("3 times", selected[0].quote)

    def test_resource_question_detection_does_not_match_inside_method_name(self) -> None:
        harness = FinalEvidenceHarness()

        self.assertFalse(
            harness._looks_like_efficiency_or_resource_question(
                "How does EfficientNet describe its compound scaling method?"
            )
        )

    def test_abstract_scoped_resource_question_keeps_prompt_focused(self) -> None:
        harness = FinalEvidenceHarness()
        summary = make_evidence(
            "summary-benefits",
            score=0.86,
            text=(
                "Abstract We propose a parameter-efficient method that reduces trainable "
                "parameters by 10,000 times and the memory requirement by 3 times."
            ),
        )
        summary.section = "Abstract"
        supporting = make_evidence(
            "supporting-resource",
            score=1.0,
            text="The method also improves storage efficiency and throughput in downstream adaptation.",
        )
        decoy = make_evidence(
            "resource-background",
            score=1.0,
            text="Background discussion mentions parameters, memory, compute, and several unrelated budgets.",
        )

        selected = harness._filter_evidence_for_question(
            "What efficiency benefits does the abstract report?",
            [supporting, decoy, summary],
            top_k=5,
        )

        self.assertLessEqual(len(selected), 2)
        self.assertEqual(selected[0].chunk_id, "summary-benefits")

    def test_adaptation_question_prefers_definition_over_experiment_context(self) -> None:
        harness = FinalEvidenceHarness()
        definition = make_evidence(
            "adaptation-definition",
            score=0.88,
            text=(
                "Abstract We propose an adaptation approach that freezes pretrained weights "
                "and injects trainable rank decomposition matrices into each layer for "
                "downstream tasks."
            ),
        )
        definition.section = "Abstract"
        experiment = make_evidence(
            "experiment-context",
            score=1.0,
            text="Results compare several methods on validation accuracy and parameter budgets.",
        )
        experiment.section = "Methods"

        selected = harness._filter_evidence_for_question(
            "What is the main adaptation idea?",
            [experiment, definition],
            top_k=1,
        )

        self.assertEqual(selected[0].chunk_id, "adaptation-definition")

    def test_design_training_factor_question_prefers_factor_summary(self) -> None:
        harness = FinalEvidenceHarness()
        factor_summary = make_evidence(
            "factor-summary",
            score=0.82,
            text=(
                "Abstract We systematically study the major components of the framework. "
                "Composition of data augmentations plays a critical role, a learnable "
                "nonlinear transformation before the loss improves representations, and "
                "larger batch sizes with more training steps help."
            ),
        )
        factor_summary.section = "Abstract"
        decoy = make_evidence(
            "leaderboard-context",
            score=1.0,
            text="Linear evaluation reports top-1 accuracy for several representation models.",
        )
        decoy.section = "Introduction"

        selected = harness._filter_evidence_for_question(
            "Which design or training factors are important for learned representations?",
            [decoy, factor_summary],
            top_k=1,
        )

        self.assertEqual(selected[0].chunk_id, "factor-summary")
        self.assertIn("critical role", selected[0].quote)
        self.assertIn("training steps", selected[0].quote)

    def test_model_prompt_marks_evidence_use_for_the_model(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "result",
            citation_id="E1",
            score=1.0,
            text="The result improves accuracy by 5 percent while using fewer trainable parameters.",
        )

        prompt = harness._format_evidence_for_model_prompt(
            question="What result and efficiency benefit is reported?",
            evidence=[evidence],
        )

        self.assertIn("Use:", prompt)
        self.assertIn("result or efficiency claim", prompt)

    def test_answer_contract_requires_numeric_efficiency_details(self) -> None:
        harness = FinalEvidenceHarness()

        contract = harness._answer_contract_for_question(
            "What efficiency benefits does the abstract report?"
        )

        self.assertIn("效率/资源问题", contract)
        self.assertIn("必须原样写出这些数值", contract)

    def test_answer_contract_requires_factor_enumeration(self) -> None:
        harness = FinalEvidenceHarness()

        contract = harness._answer_contract_for_question(
            "Which design or training factors are important for learned representations?"
        )

        self.assertIn("设计/训练因素问题", contract)
        self.assertIn("逐项列出", contract)

    def test_final_repair_keeps_required_visual_candidate(self) -> None:
        harness = FinalEvidenceHarness()
        visual = make_evidence(
            "method-figure",
            score=1.0,
            chunk_type="figure_image",
            image_id="img-1",
            text=(
                "Figure 4: Method overview. "
                "The visual shows input prompts, intermediate masks, and the final validated output."
            ),
        )
        selected = [
            make_evidence(f"text-{index}", score=1.0, text="Related method background.")
            for index in range(5)
        ]

        repaired = harness._repair_final_evidence_selection(
            question="How does the figure show the method's prompt and output flow?",
            selected=selected,
            candidates=[visual, *selected],
            limit=5,
        )

        self.assertEqual(repaired[0].chunk_id, "method-figure")
        self.assertTrue(any(item.chunk_id == "method-figure" for item in repaired))

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

    def test_compare_efficiency_prefers_mechanism_evidence_for_each_document(self) -> None:
        harness = MultiDocumentEvidenceHarness()
        efficient_result = make_evidence(
            "efficient-result",
            score=1.0,
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text="The result section reports strong accuracy with fewer parameters and better efficiency.",
        )
        efficient_method = make_evidence(
            "efficient-method",
            score=0.72,
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text="The method uses a compound coefficient to scale network width, depth, and resolution.",
        )
        adaptation_result = make_evidence(
            "adaptation-result",
            score=1.0,
            document_id="doc-adaptation",
            paper_name="adaptation.pdf",
            text="The abstract reports 10,000 times fewer trainable parameters and a lower memory requirement.",
        )
        adaptation_method = make_evidence(
            "adaptation-method",
            score=0.72,
            document_id="doc-adaptation",
            paper_name="adaptation.pdf",
            text=(
                "The adaptation approach freezes pre-trained weights and injects trainable rank "
                "decomposition matrices into model layers."
            ),
        )

        selected = harness._filter_evidence_for_question(
            "Compare how the two approaches pursue efficiency. Use evidence from both papers.",
            [efficient_result, adaptation_result, efficient_method, adaptation_method],
            top_k=4,
            target_document_ids=["doc-efficient", "doc-adaptation"],
        )
        selected_chunks = {item.chunk_id for item in selected}

        self.assertIn("efficient-method", selected_chunks)
        self.assertIn("adaptation-method", selected_chunks)
        self.assertEqual({"doc-efficient", "doc-adaptation"}, {item.document_id for item in selected})

    def test_compare_efficiency_keeps_expanded_mechanism_sentence(self) -> None:
        harness = MultiDocumentEvidenceHarness()
        generic = make_evidence(
            "efficient-result",
            score=1.0,
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text="The result section reports strong accuracy with fewer parameters and better efficiency.",
        )
        expanded = make_evidence(
            "efficient-method:sentence:12",
            score=0.8,
            chunk_type="sentence",
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text=(
                "In this paper, we propose a compound scaling method using a compound "
                "coefficient to scale network width, depth, and resolution."
            ),
        )
        expanded.score_source = "expanded_sentence"

        selected = harness._repair_final_evidence_selection(
            question="Compare how the approaches pursue efficiency.",
            selected=[generic],
            candidates=[generic, expanded],
            limit=1,
        )

        self.assertEqual(selected[0].chunk_id, "efficient-method:sentence:12")
        self.assertIn("compound coefficient", selected[0].quote)

    def test_compare_prompt_keeps_mechanism_evidence_from_each_document(self) -> None:
        harness = MultiDocumentEvidenceHarness()
        efficient_result = make_evidence(
            "efficient-result",
            score=1.0,
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text="EfficientNet reports strong accuracy with fewer parameters.",
        )
        lora_result = make_evidence(
            "lora-result",
            score=1.0,
            document_id="doc-lora",
            paper_name="lora.pdf",
            text="LoRA reduces the number of trainable parameters and memory requirement.",
        )
        efficient_mechanism = make_evidence(
            "efficient-method:sentence:12",
            score=0.9,
            chunk_type="sentence",
            document_id="doc-efficient",
            paper_name="efficient.pdf",
            text="The method uses a compound coefficient to scale network width, depth, and resolution.",
        )
        efficient_mechanism.score_source = "expanded_sentence"
        lora_mechanism = make_evidence(
            "lora-method:sentence:04",
            score=0.9,
            chunk_type="sentence",
            document_id="doc-lora",
            paper_name="lora.pdf",
            text="LoRA freezes pretrained weights and injects trainable rank decomposition matrices.",
        )
        lora_mechanism.score_source = "expanded_sentence"

        prompt_evidence = harness._prefer_direct_support_prompt_evidence(
            question="Compare how EfficientNet and LoRA pursue efficiency.",
            selected=[efficient_result, lora_result],
            candidates=[efficient_result, lora_result, efficient_mechanism, lora_mechanism],
            limit=4,
        )
        prompt_chunks = {item.chunk_id for item in prompt_evidence}

        self.assertIn("efficient-method:sentence:12", prompt_chunks)
        self.assertIn("lora-method:sentence:04", prompt_chunks)

    def test_local_framework_answer_uses_complete_core_evidence(self) -> None:
        harness = FinalEvidenceHarness()
        direct = make_evidence(
            "direct-core",
            citation_id="E2",
            score=0.8,
            text="The evaluation framework core is composed of three functions: PLAN, BUILD, and REVIEW.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What are the framework core functions?",
            evidence=[direct],
        )

        self.assertIn("PLAN, BUILD, and REVIEW", answer)
        self.assertIn("[E2]", answer)

    def test_local_metric_answer_keeps_table_values(self) -> None:
        harness = FinalEvidenceHarness()
        table = make_evidence(
            "table-metric",
            citation_id="E1",
            score=0.9,
            chunk_type="table",
            text="Table 2: Model XL reports accuracy 84 percent and error 16 percent on the benchmark.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What accuracy and error results are reported in the benchmark table?",
            evidence=[table],
        )

        self.assertIn("accuracy 84 percent", answer)
        self.assertIn("error 16 percent", answer)
        self.assertIn("[E1]", answer)

    def test_local_dataset_answer_uses_exact_scale_from_evidence(self) -> None:
        harness = FinalEvidenceHarness()
        evidence = make_evidence(
            "dataset-scale",
            citation_id="E1",
            score=1.0,
            text="We constructed a new training dataset of 12 million paired examples.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What dataset scale was used for training?",
            evidence=[evidence],
        )

        self.assertIn("12 million paired examples", answer)
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
            "manual-image",
            citation_id="E1",
            score=1.0,
            chunk_type="image",
            image_id="img-1",
            text="图片实际内容 这是一页工业传感器安装手册，包含接线图、端口标注和维护说明。",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What does the scanned PDF image show?",
            evidence=[visual],
        )

        self.assertIn("工业传感器安装手册", answer)
        self.assertIn("[E1]", answer)


    def test_local_answer_prefers_sentence_level_efficiency_support(self) -> None:
        harness = FinalEvidenceHarness()
        table_decoy = make_evidence(
            "table-decoy",
            citation_id="E1",
            score=1.0,
            chunk_type="table",
            text="Table 4 | Setting | Result | 91.0 | 92.0 | The table reports scores for several runs.",
        )
        direct = make_evidence(
            "direct-efficiency",
            citation_id="E2",
            score=0.72,
            text=(
                "The approach reduces trainable parameters by 100 times and lowers memory use "
                "while adding no inference latency."
            ),
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What efficiency benefit does the approach report for trainable parameters and memory?",
            evidence=[table_decoy, direct],
        )

        self.assertIn("[E2]", answer)
        if "[E1]" in answer:
            self.assertLess(answer.index("[E2]"), answer.index("[E1]"))

    def test_local_answer_prefers_method_quote_over_result_table_for_method_question(self) -> None:
        harness = FinalEvidenceHarness()
        table_decoy = make_evidence(
            "result-table",
            citation_id="E1",
            score=1.0,
            chunk_type="table",
            text="Table 1 | Method | Score | Error | Variant A | 88.0 | 12.0.",
        )
        direct = make_evidence(
            "method-quote",
            citation_id="E2",
            score=0.70,
            text="The method architecture uses a lightweight encoder and a scoring head trained with a contrastive objective.",
        )

        answer = harness._build_local_structured_evidence_answer(
            question="What method architecture does the approach use?",
            evidence=[table_decoy, direct],
        )

        self.assertIn("[E2]", answer)
        if "[E1]" in answer:
            self.assertLess(answer.index("[E2]"), answer.index("[E1]"))


if __name__ == "__main__":
    unittest.main()
