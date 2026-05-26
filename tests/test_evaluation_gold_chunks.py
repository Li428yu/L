from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.evaluation import run_eval_suite, score_eval_case
from backend.app.models import EvaluationCase, EvidenceItem, RagTrace


def make_evidence(chunk_id: str, *, citation_id: str = "E1", text: str = "Relevant evidence.") -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=0.9,
        text=text,
        quote=text,
    )


def make_trace(*, top_k: int, evidence_quality_trace: list[dict] | None = None) -> RagTrace:
    return RagTrace(
        model_profile="test",
        vector_store="test",
        vector_record_count=10,
        top_k=top_k,
        filter_document_ids=["doc-1"],
        retrieved_count=len(evidence_quality_trace or []),
        final_prompt_evidence=[],
        evidence_quality_trace=evidence_quality_trace or [],
    )


def quality_row(chunk_id: str, candidate_rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "paper_name": "paper.pdf",
        "candidate_rank": candidate_rank,
        "selection_status": "selected_by_retrieval_filter",
        "quality_label": "strong",
        "score": 0.9,
        "relevance_score": 0.8,
        "readability_score": 0.8,
    }


def make_response(*, evidence: list[EvidenceItem], top_k: int, evidence_quality_trace: list[dict] | None = None):
    return SimpleNamespace(
        answer="The answer is grounded in the cited evidence. [E1]",
        evidence=evidence,
        rag_trace=make_trace(top_k=top_k, evidence_quality_trace=evidence_quality_trace),
    )


class EvaluationGoldChunkTests(unittest.TestCase):
    def test_gold_chunk_hit_and_miss_ids_are_reported(self) -> None:
        case = EvaluationCase(
            id="gold-hit",
            question="What does the evidence say?",
            expected_chunk_ids=["gold-1", "gold-3"],
        )
        response = make_response(
            top_k=3,
            evidence=[
                make_evidence("gold-1", citation_id="E1"),
                make_evidence("decoy", citation_id="E2"),
                make_evidence("gold-3", citation_id="E3"),
            ],
            evidence_quality_trace=[
                quality_row("gold-1", 1),
                quality_row("decoy", 2),
                quality_row("gold-3", 3),
            ],
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertEqual(metrics["gold_chunk_recall_at_1"], 0.5)
        self.assertEqual(metrics["gold_chunk_recall_at_k"], 1.0)
        self.assertEqual(metrics["gold_chunk_hit_ids"], ["gold-1", "gold-3"])
        self.assertEqual(metrics["gold_chunk_missed_ids"], [])
        self.assertEqual(metrics["score_breakdown"]["gold_chunk_answer_all_hit"], 1.0)

    def test_gold_chunk_miss_distinguishes_filtered_from_not_retrieved(self) -> None:
        case = EvaluationCase(
            id="gold-partial",
            question="What does the evidence say?",
            expected_chunk_ids=["gold-kept", "gold-dropped", "gold-never"],
        )
        response = make_response(
            top_k=3,
            evidence=[make_evidence("gold-kept", citation_id="E1")],
            evidence_quality_trace=[
                quality_row("gold-kept", 1),
                quality_row("gold-dropped", 2),
                quality_row("decoy", 3),
            ],
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertAlmostEqual(metrics["gold_chunk_recall_at_k"], 1 / 3)
        self.assertAlmostEqual(metrics["gold_chunk_candidate_recall_at_k"], 2 / 3)
        self.assertEqual(metrics["gold_chunk_hit_ids"], ["gold-kept"])
        self.assertEqual(metrics["gold_chunk_missed_ids"], ["gold-dropped", "gold-never"])
        self.assertEqual(metrics["gold_chunk_dropped_after_retrieval_ids"], ["gold-dropped"])
        self.assertEqual(metrics["gold_chunk_not_retrieved_ids"], ["gold-never"])
        self.assertLessEqual(metrics["score_cap"], 0.60)

    def test_run_eval_suite_applies_gold_cap_after_llm_judge(self) -> None:
        case = EvaluationCase(
            id="gold-miss-with-judge",
            question="What does the evidence say?",
            expected_chunk_ids=["missing-gold"],
        )
        response = make_response(
            top_k=3,
            evidence=[make_evidence("decoy", citation_id="E1")],
            evidence_quality_trace=[quality_row("decoy", 1)],
        )
        agent = SimpleNamespace(ask=lambda request: response, settings=SimpleNamespace(enable_llm_judge=True))

        with patch(
            "backend.app.evaluation.score_with_llm_judge",
            return_value={"used": True, "score": 1.0, "scores": {}, "reason": "perfect judge"},
        ):
            run = run_eval_suite(
                suite_name="test-suite",
                cases=[case],
                agent=agent,
                document_ids=["doc-1"],
                enable_judge=True,
            )

        result = run.results[0]
        self.assertEqual(result.gold_chunk_missed_ids, ["missing-gold"])
        self.assertEqual(result.score, 0.40)
        self.assertGreater(result.score_breakdown["score_after_judge_before_caps"], result.score)
        self.assertEqual(result.score_breakdown["final_score_capped"], 1.0)
        self.assertEqual(result.trace_summary["gold_evidence"]["status"], "not_retrieved")


if __name__ == "__main__":
    unittest.main()
