from __future__ import annotations

import unittest

from backend.app.eval_grading import apply_eval_result_grading, build_eval_run_grading_summary
from backend.app.models import EvaluationResult


def make_result(**overrides) -> EvaluationResult:
    data = {
        "case_id": "case-1",
        "question": "What does the evidence say?",
        "answer": "Grounded answer [E1].",
        "evidence": [{"citation_id": "E1", "chunk_id": "chunk-1"}],
        "trace_summary": {"case_metadata": {"expected_modalities": [], "expected_refusal": False}},
        "retrieval_hit": True,
        "citation_hit": True,
        "keyword_hit_rate": 1.0,
        "context_precision": 0.8,
        "context_recall": 0.8,
        "gold_chunk_count": 0,
        "gold_chunk_recall_at_k": 0.0,
        "gold_chunk_candidate_recall_at_k": 0.0,
        "document_coverage": 1.0,
        "citation_accuracy": 1.0,
        "answer_relevance": 1.0,
        "faithfulness_proxy": 0.9,
        "claim_hit_rate": 1.0,
        "forbidden_claim_rate": 0.0,
        "refusal_correctness": 1.0,
        "relation_hit": 1.0,
        "score": 0.9,
        "score_breakdown": {"claim_term_count": 0.0},
        "latency_ms": 12,
    }
    data.update(overrides)
    return EvaluationResult(**data)


class EvalGradingTests(unittest.TestCase):
    def test_clean_result_passes(self) -> None:
        result = apply_eval_result_grading(make_result())

        self.assertEqual(result.result_status, "pass")
        self.assertEqual(result.failure_categories, [])
        self.assertEqual(result.trace_summary["grading"]["status"], "pass")

    def test_execution_error_is_blocked(self) -> None:
        result = apply_eval_result_grading(
            make_result(
                error="model timeout",
                score=0.0,
                answer="",
                score_breakdown={"answer_generation_failed": 1.0},
            )
        )

        self.assertEqual(result.result_status, "blocked")
        self.assertIn("answer_generation_failure", result.failure_categories)

    def test_missing_candidate_gold_chunk_is_retrieval_failure(self) -> None:
        result = apply_eval_result_grading(
            make_result(
                gold_chunk_count=1,
                gold_chunk_recall_at_k=0.0,
                gold_chunk_candidate_recall_at_k=0.0,
                score=0.4,
            )
        )

        self.assertEqual(result.result_status, "fail")
        self.assertIn("retrieval_failure", result.failure_categories)

    def test_candidate_gold_chunk_dropped_is_filtering_failure(self) -> None:
        result = apply_eval_result_grading(
            make_result(
                gold_chunk_count=1,
                gold_chunk_recall_at_k=0.0,
                gold_chunk_candidate_recall_at_k=1.0,
                score=0.4,
            )
        )

        self.assertEqual(result.result_status, "fail")
        self.assertIn("evidence_filtering_failure", result.failure_categories)
        self.assertNotIn("retrieval_failure", result.failure_categories)

    def test_visual_citation_and_refusal_failures_are_classified(self) -> None:
        result = apply_eval_result_grading(
            make_result(
                trace_summary={"case_metadata": {"expected_modalities": ["vision"], "expected_refusal": True}},
                visual_evidence_hit=False,
                citation_accuracy=0.4,
                refusal_correctness=0.0,
                score=0.35,
            )
        )

        self.assertEqual(result.result_status, "fail")
        self.assertIn("visual_ocr_failure", result.failure_categories)
        self.assertIn("citation_failure", result.failure_categories)
        self.assertIn("refusal_failure", result.failure_categories)

    def test_embedding_fallback_blocks_case_trustworthiness(self) -> None:
        result = apply_eval_result_grading(
            make_result(
                embedding_used_fallback=True,
                score=0.9,
            )
        )

        self.assertEqual(result.result_status, "blocked")
        self.assertIn("embedding_fallback", result.failure_categories)
        self.assertEqual(result.trace_summary["grading"]["status"], "blocked")

    def test_run_summary_counts_statuses_and_categories(self) -> None:
        passed = apply_eval_result_grading(make_result(case_id="pass"))
        failed = apply_eval_result_grading(
            make_result(
                case_id="fail",
                gold_chunk_count=1,
                gold_chunk_recall_at_k=0.0,
                gold_chunk_candidate_recall_at_k=0.0,
                score=0.4,
            )
        )

        summary = build_eval_run_grading_summary([passed, failed])

        self.assertEqual(summary["status_counts"], {"pass": 1, "fail": 1})
        self.assertEqual(summary["failure_category_counts"]["retrieval_failure"], 1)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertTrue(summary["evaluation_trustworthy"])
        self.assertEqual(summary["trust_gate_status"], "passed")

    def test_run_summary_marks_embedding_fallback_not_comparable(self) -> None:
        fallback = apply_eval_result_grading(
            make_result(
                case_id="fallback",
                embedding_used_fallback=True,
                score=0.9,
            )
        )

        summary = build_eval_run_grading_summary([fallback])

        self.assertEqual(summary["status_counts"], {"blocked": 1})
        self.assertFalse(summary["evaluation_trustworthy"])
        self.assertEqual(summary["trust_gate_status"], "not_comparable")
        self.assertEqual(summary["trust_gate_failures"][0]["category"], "embedding_fallback")
        self.assertEqual(summary["trust_gate_failures"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
