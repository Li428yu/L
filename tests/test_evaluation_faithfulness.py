from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.evaluation import score_eval_case
from backend.app.models import EvaluationCase, EvidenceItem


def make_evidence(text: str) -> EvidenceItem:
    return EvidenceItem(
        citation_id="E1",
        chunk_id="chunk-E1",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=0.8,
        text=text,
        quote=text,
    )


def make_response(*, answer: str, evidence_text: str):
    return SimpleNamespace(
        answer=answer,
        evidence=[make_evidence(evidence_text)],
        rag_trace=SimpleNamespace(
            top_k=1,
            verification={},
            document_relation_map=[],
            multi_document_coverage={},
            visual_ocr_warnings=[],
        ),
    )


class EvaluationFaithfulnessTests(unittest.TestCase):
    def test_supported_cited_sentence_scores_high(self) -> None:
        case = EvaluationCase(id="faithful", question="What does the model use?")
        response = make_response(
            answer="The model uses attention for sequence modeling [E1].",
            evidence_text="The paper proposes an attention mechanism for sequence modeling.",
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertGreaterEqual(metrics["faithfulness_proxy"], 0.7)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertEqual(metrics["score_breakdown"]["faithfulness_weak_sentence_count"], 0.0)

    def test_valid_citation_does_not_hide_unsupported_claim(self) -> None:
        case = EvaluationCase(id="unfaithful", question="What does the model use?")
        response = make_response(
            answer="The model uses convolution and reports 99 percent accuracy [E1].",
            evidence_text="The paper proposes an attention mechanism for sequence modeling.",
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertLess(metrics["faithfulness_proxy"], 0.45)
        self.assertEqual(metrics["score_breakdown"]["faithfulness_weak_sentence_count"], 1.0)

    def test_supported_but_uncited_sentence_is_capped(self) -> None:
        case = EvaluationCase(id="uncited", question="What does the model use?")
        response = make_response(
            answer="The model uses attention for sequence modeling.",
            evidence_text="The paper proposes an attention mechanism for sequence modeling.",
        )

        metrics = score_eval_case(case=case, response=response)

        self.assertEqual(metrics["citation_accuracy"], 0.0)
        self.assertGreater(metrics["faithfulness_proxy"], 0.0)
        self.assertLessEqual(metrics["faithfulness_proxy"], 0.65)
        self.assertEqual(metrics["score_breakdown"]["faithfulness_uncited_sentence_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
