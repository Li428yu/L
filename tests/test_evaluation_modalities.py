from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.evaluation import score_eval_case
from backend.app.models import EvaluationCase, EvidenceItem, RelatedImageInfo


def make_evidence(
    *,
    citation_id: str = "E1",
    chunk_type: str = "text",
    image_id: str | None = None,
    text: str = "Plain cited evidence.",
    related_images: list[RelatedImageInfo] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=0.8,
        text=text,
        quote=text,
        chunk_type=chunk_type,
        image_id=image_id,
        related_images=related_images or [],
    )


def make_response(*, evidence: list[EvidenceItem], answer: str = "Answer grounded in evidence. [E1]"):
    return SimpleNamespace(
        answer=answer,
        evidence=evidence,
        rag_trace=SimpleNamespace(
            top_k=len(evidence) or 1,
            verification={},
            document_relation_map=[],
            multi_document_coverage={},
            visual_ocr_warnings=[],
        ),
    )


class EvaluationModalityTests(unittest.TestCase):
    def test_ocr_expected_requires_actual_ocr_text(self) -> None:
        case = EvaluationCase(id="ocr-no-text", question="Read the scanned image.", expected_modalities=["ocr"])
        response = make_response(
            evidence=[
                make_evidence(
                    chunk_type="figure_image",
                    image_id="image-1",
                    text="Image evidence. Type: figure_image. Pages: 1-1.",
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertFalse(metrics["ocr_evidence_hit"])
        self.assertFalse(metrics["ocr_text_hit"])
        self.assertEqual(metrics["score_breakdown"]["cited_image_evidence"], 1.0)
        self.assertEqual(metrics["score_breakdown"]["cited_ocr_text"], 0.0)

    def test_ocr_expected_passes_with_related_ocr_text(self) -> None:
        case = EvaluationCase(id="ocr-with-text", question="Read the scanned image.", expected_modalities=["ocr"])
        response = make_response(
            evidence=[
                make_evidence(
                    related_images=[
                        RelatedImageInfo(
                            id="image-1",
                            document_id="doc-1",
                            page_start=1,
                            page_end=1,
                            kind="figure_image",
                            ocr_text="Total score: 95 points.",
                            status="ready",
                        )
                    ]
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertTrue(metrics["ocr_evidence_hit"])
        self.assertTrue(metrics["ocr_text_hit"])
        self.assertEqual(metrics["score_breakdown"]["cited_ocr_text"], 1.0)

    def test_vision_expected_requires_visual_summary_on_visual_evidence(self) -> None:
        case = EvaluationCase(id="vision-no-summary", question="What does the figure show?", expected_modalities=["vision"])
        response = make_response(
            evidence=[
                make_evidence(
                    chunk_type="figure_image",
                    image_id="image-1",
                    text="Image evidence. Type: figure_image. Pages: 1-1.",
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertFalse(metrics["visual_evidence_hit"])
        self.assertFalse(metrics["visual_summary_hit"])
        self.assertEqual(metrics["score_breakdown"]["cited_image_evidence"], 1.0)
        self.assertEqual(metrics["score_breakdown"]["cited_visual_summary"], 0.0)

    def test_vision_expected_passes_with_image_summary(self) -> None:
        case = EvaluationCase(id="vision-with-summary", question="What does the figure show?", expected_modalities=["vision"])
        response = make_response(
            evidence=[
                make_evidence(
                    chunk_type="figure_image",
                    image_id="image-1",
                    text=(
                        "Image evidence. Type: figure_image. Pages: 1-1.\n"
                        "Summary: The figure shows the model architecture and attention flow."
                    ),
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertTrue(metrics["visual_evidence_hit"])
        self.assertTrue(metrics["visual_summary_hit"])
        self.assertEqual(metrics["score_breakdown"]["cited_visual_summary"], 1.0)

    def test_vision_expected_passes_with_chinese_image_summary_markers(self) -> None:
        case = EvaluationCase(id="vision-with-chinese-summary", question="What does the figure show?", expected_modalities=["vision"])
        response = make_response(
            evidence=[
                make_evidence(
                    chunk_type="figure_image",
                    image_id="image-1",
                    text=(
                        "Image evidence. Type: figure_image. Pages: 1-1.\n"
                        "图片实际内容 该图展示模型结构、关键模块和注意力流向。"
                    ),
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertTrue(metrics["visual_evidence_hit"])
        self.assertTrue(metrics["visual_summary_hit"])
        self.assertEqual(metrics["score_breakdown"]["cited_visual_summary"], 1.0)

    def test_vision_expected_passes_with_prefixed_summary_heading(self) -> None:
        case = EvaluationCase(id="vision-with-prefixed-summary", question="What does the figure show?", expected_modalities=["vision"])
        response = make_response(
            evidence=[
                make_evidence(
                    chunk_type="figure_image",
                    image_id="image-1",
                    text=(
                        "CSF Functions Summary: ### 图片说明 1. 实际内容："
                        "这是环形功能示意图，展示 GOVERN、IDENTIFY、PROTECT、DETECT、RESPOND、RECOVER。"
                    ),
                )
            ]
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertTrue(metrics["visual_evidence_hit"])
        self.assertTrue(metrics["visual_summary_hit"])


if __name__ == "__main__":
    unittest.main()
