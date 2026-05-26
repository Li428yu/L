from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.agent_parts.answering import ANSWER_SYSTEM_PROMPT
from backend.app.evaluation import (
    JUDGE_SCORE_KEYS,
    JUDGE_SYSTEM_PROMPT,
    judge_evidence_payload,
    judge_trace_payload,
)
from backend.app.image_processing import ExtractedImage, _vision_prompt_for_image


class PromptCompactionTests(unittest.TestCase):
    def test_core_prompts_stay_compact(self) -> None:
        image = ExtractedImage(
            id="img-1",
            document_id="doc-1",
            image_hash="hash",
            page_start=1,
            page_end=1,
            bbox=(0.0, 0.0, 10.0, 10.0),
            image_path="",
            thumbnail_path="",
            width=10,
            height=10,
            kind="chart_image",
            ocr_text="accuracy 92%",
            vision_summary="",
            caption_text="Figure 1. Accuracy.",
            status="ready",
        )

        self.assertLessEqual(len(ANSWER_SYSTEM_PROMPT), 260)
        self.assertLessEqual(len(JUDGE_SYSTEM_PROMPT), 140)
        self.assertLessEqual(len(_vision_prompt_for_image(image)), 180)

    def test_judge_score_keys_are_stable(self) -> None:
        self.assertEqual(
            JUDGE_SCORE_KEYS,
            [
                "answer_relevance",
                "faithfulness",
                "citation_support",
                "context_usage",
                "multi_document_clarity",
                "visual_grounding",
                "claim_coverage",
                "refusal_correctness",
                "completeness",
                "no_hallucination",
            ],
        )

    def test_judge_evidence_payload_is_bounded(self) -> None:
        evidence = [
            SimpleNamespace(
                citation_id=f"E{index}",
                paper_name="paper",
                page=index,
                chunk_type="text",
                image_id="",
                quote="x" * 700,
                text="fallback text",
            )
            for index in range(8)
        ]

        payload = judge_evidence_payload(evidence)

        self.assertEqual(len(payload), 6)
        self.assertLessEqual(len(payload[0]["quote"]), 523)
        self.assertTrue(payload[0]["quote"].endswith("..."))

    def test_judge_trace_payload_is_bounded(self) -> None:
        trace = SimpleNamespace(
            retrieval_pipeline="hybrid",
            ranking_method="rrf",
            retrieved_count=20,
            final_prompt_evidence=[f"E{index}" for index in range(10)],
            evidence_coverage={"status": "ok"},
            evidence_quality_trace=[
                SimpleNamespace(
                    chunk_id=f"chunk-{index}",
                    selection_status="selected",
                    quality_label="strong",
                    candidate_rank=index,
                    rejection_reason="",
                    judge_reason="r" * 240,
                )
                for index in range(10)
            ],
            multi_document_coverage={"covered_document_count": 2},
            visual_ocr_warnings=[f"warning-{index}" for index in range(6)],
            verification={"status": "ok"},
        )

        payload = judge_trace_payload(trace)

        self.assertEqual(len(payload["final_prompt_evidence"]), 8)
        self.assertEqual(len(payload["evidence_quality_trace"]), 8)
        self.assertEqual(len(payload["visual_ocr_warnings"]), 4)
        self.assertLessEqual(len(payload["evidence_quality_trace"][0]["reason"]), 163)


if __name__ == "__main__":
    unittest.main()
