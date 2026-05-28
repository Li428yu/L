from __future__ import annotations

import unittest

from backend.app.agent_parts.evidence_coverage import AgentEvidenceCoverageMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class EvidenceCoverageHarness(AgentEvidenceCoverageMixin, AgentTextUtilityMixin):
    def _looks_like_reference_question(self, question: str) -> bool:
        return False

    def _looks_like_field_lookup_question(self, question: str) -> bool:
        return False

    def _looks_like_compare_question(self, question: str) -> bool:
        return "compare" in question.lower()

    def _looks_like_document_wide_question(self, question: str) -> bool:
        return "summarize" in question.lower()

    def _looks_like_broad_overview_question(self, question: str) -> bool:
        return False

    def _looks_like_multi_document_topic_question(self, question: str) -> bool:
        return "compare" in question.lower()

    def _looks_like_visual_retrieval_question(self, question: str) -> bool:
        return "figure" in question.lower()

    def _looks_like_ocr_question(self, question: str) -> bool:
        return "ocr" in question.lower()


def make_evidence(
    *,
    score: float = 0.8,
    text: str = "The paper proposes an attention mechanism for sequence modeling.",
    quality_label: str = "strong",
    chunk_type: str = "text",
    image_id: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        citation_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        text=text,
        quote=text,
        quality_label=quality_label,
        selection_status="selected_for_answer",
        chunk_type=chunk_type,
        image_id=image_id,
    )


class EvidenceCoverageTests(unittest.TestCase):
    def test_no_evidence_after_judge_refuses(self) -> None:
        harness = EvidenceCoverageHarness()

        decision = harness._evidence_coverage_decision(
            state={
                "question": "What does the paper say about attention?",
                "needs_retrieval": True,
                "evidence": [],
                "evidence_quality": "none",
            },
            prompt_evidence=[],
        )

        self.assertTrue(decision["should_refuse"])
        self.assertEqual(decision["reason_code"], "no_evidence_after_judge")

    def test_strong_direct_evidence_passes(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [make_evidence()]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "How does attention help sequence modeling?",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "strong",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "direct"}],
            },
            prompt_evidence=evidence,
        )

        self.assertFalse(decision["should_refuse"])

    def test_all_weak_final_labels_refuse(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [
            make_evidence(
                score=0.32,
                text="A loosely related appendix paragraph.",
                quality_label="weak",
            )
        ]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "How does attention help sequence modeling?",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "medium",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "supporting"}],
            },
            prompt_evidence=evidence,
        )

        self.assertTrue(decision["should_refuse"])
        self.assertEqual(decision["reason_code"], "all_final_evidence_weak_or_noise")

    def test_weak_labels_pass_when_relevant_direct_support_is_still_usable(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [
            make_evidence(
                score=1.0,
                text=(
                    "The paper identifies important design and training factors for learned "
                    "representations, including data augmentation composition, nonlinear "
                    "transformation, batch size, and training steps."
                ),
                quality_label="noise",
            )
        ]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "Which design or training factors are important for learned representations?",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "strong",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "direct"}],
            },
            prompt_evidence=evidence,
        )

        self.assertFalse(decision["should_refuse"])

    def test_multi_document_incomplete_coverage_refuses(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [make_evidence()]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "Compare these documents.",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "strong",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "direct"}],
                "multi_document_coverage": {
                    "requested_document_count": 2,
                    "covered_document_count": 1,
                    "missing_document_names": ["other.pdf"],
                },
            },
            prompt_evidence=evidence,
        )

        self.assertTrue(decision["should_refuse"])
        self.assertEqual(decision["reason_code"], "multi_document_coverage_incomplete")

    def test_visual_question_requires_image_evidence(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [make_evidence()]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "What does the figure show?",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "strong",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "direct"}],
            },
            prompt_evidence=evidence,
        )

        self.assertTrue(decision["should_refuse"])
        self.assertEqual(decision["reason_code"], "missing_visual_evidence")

    def test_ocr_question_accepts_text_read_from_visual_summary(self) -> None:
        harness = EvidenceCoverageHarness()
        evidence = [
            make_evidence(
                chunk_type="image",
                image_id="image-1",
                text="图片实际内容 这是 LinnSequencer 32轨 MIDI 序列录音机的产品说明页，文字清晰可识别。",
            )
        ]

        decision = harness._evidence_coverage_decision(
            state={
                "question": "What does the OCR scanned image say?",
                "needs_retrieval": True,
                "evidence": evidence,
                "evidence_quality": "strong",
                "evidence_judgments": [{"chunk_id": "chunk-1", "verdict": "direct"}],
            },
            prompt_evidence=evidence,
        )

        self.assertFalse(decision["should_refuse"])


if __name__ == "__main__":
    unittest.main()
