from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.evaluation import citation_page_diagnostics_for_eval, score_eval_case
from backend.app.models import EvaluationCase, EvidenceItem


def make_evidence(
    *,
    citation_id: str = "E1",
    page: int = 10,
    page_start: int | None = None,
    page_end: int | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=page,
        page_start=page_start,
        page_end=page_end,
        source="paper.pdf",
        file_hash="hash",
        score=0.8,
        text="Evidence text.",
        quote="Evidence text.",
    )


def make_response(evidence: list[EvidenceItem]):
    return SimpleNamespace(
        answer="Answer grounded in evidence. [E1]",
        evidence=evidence,
        rag_trace=SimpleNamespace(
            top_k=len(evidence) or 1,
            verification={},
            document_relation_map=[],
            multi_document_coverage={},
            visual_ocr_warnings=[],
        ),
    )


class EvaluationPageDiagnosticsTests(unittest.TestCase):
    def test_page_diagnostics_reports_partial_match_and_nearest_evidence(self) -> None:
        evidence = [make_evidence(page=10, page_start=10, page_end=12)]

        diagnostics = citation_page_diagnostics_for_eval(expected_pages=[11, 15], evidence=evidence)

        self.assertEqual(diagnostics["status"], "partial_page_match")
        self.assertEqual(diagnostics["matched_pages"], [11])
        self.assertEqual(diagnostics["missed_pages"], [15])
        self.assertEqual(diagnostics["nearest_evidence"][0]["nearest_page"], 12)
        self.assertEqual(diagnostics["nearest_evidence"][0]["distance"], 3)

    def test_page_diagnostics_flags_large_page_mismatch(self) -> None:
        evidence = [make_evidence(page=15)]

        diagnostics = citation_page_diagnostics_for_eval(expected_pages=[5], evidence=evidence)

        self.assertEqual(diagnostics["status"], "page_mismatch")
        self.assertEqual(diagnostics["missed_pages"], [5])
        self.assertEqual(diagnostics["nearest_evidence"][0]["distance"], 10)
        self.assertIn("页码", diagnostics["hint"])

    def test_score_eval_case_exposes_page_diagnostics(self) -> None:
        case = EvaluationCase(
            id="page-mismatch",
            question="What does the page say?",
            expected_pages=[5],
        )
        response = make_response([make_evidence(page=15)])

        metrics = score_eval_case(case=case, response=response)

        self.assertFalse(metrics["citation_hit"])
        self.assertEqual(metrics["citation_page_diagnostics"]["status"], "page_mismatch")
        self.assertEqual(metrics["score_breakdown"]["citation_page_missed_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
