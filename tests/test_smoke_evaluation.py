from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.evaluation import answer_point_coverage, evidence_matches_gold, score_smoke_case
from backend.app.models import EvaluationCase, EvidenceItem


def make_evidence(*, citation_id: str = "E1", text: str = "Winsock connect flow", document_id: str = "doc-1") -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=document_id,
        paper_name="network.docx",
        page=1,
        source="network.docx",
        file_hash="hash",
        score=0.8,
        text=text,
        quote=text,
    )


def make_response(*, answer: str, evidence: list[EvidenceItem], embedding_fallback: bool = False):
    return SimpleNamespace(
        answer=answer,
        evidence=evidence,
        rag_trace=SimpleNamespace(
            retrieval_strategy="hybrid_soft",
            retrieval_pipeline="dense+bm25",
            ranking_method="rrf",
            retrieved_count=len(evidence),
            top_k=5,
            embedding_requested_model="doubao-embedding-vision-251215",
            embedding_provider="doubao-embedding-vision-251215",
            embedding_used_fallback=embedding_fallback,
            embedding_fallback_reason="",
            final_prompt_evidence=[item.citation_id for item in evidence],
            evidence_quality="strong",
            evidence_coverage={},
            multi_document_coverage={},
            verification={},
        ),
    )


class SmokeEvaluationTests(unittest.TestCase):
    def test_passes_when_answer_cites_matching_evidence(self) -> None:
        case = EvaluationCase(
            id="ok",
            question="What API is used?",
            expected_keywords=["Winsock"],
            expected_evidence_keywords=["connect"],
        )
        response = make_response(
            answer="The design uses Winsock connect flow [E1].",
            evidence=[make_evidence(text="The client calls connect through Winsock.")],
        )

        metrics = score_smoke_case(case=case, response=response)

        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["valid_citation_count"], 1)
        self.assertEqual(metrics["evidence_keyword_hit_rate"], 1.0)

    def test_fails_when_cited_evidence_misses_expected_keyword(self) -> None:
        case = EvaluationCase(
            id="bad-evidence",
            question="What API is used?",
            expected_evidence_keywords=["connect"],
        )
        response = make_response(
            answer="The design uses a network API [E1].",
            evidence=[make_evidence(text="This paragraph only discusses project background.")],
        )

        metrics = score_smoke_case(case=case, response=response)

        self.assertEqual(metrics["status"], "fail")
        self.assertIn("evidence_keyword_missing", metrics["failure_categories"])

    def test_embedding_fallback_blocks_trust(self) -> None:
        case = EvaluationCase(id="fallback", question="Summarize with evidence.")
        response = make_response(
            answer="The document describes the implementation [E1].",
            evidence=[make_evidence()],
            embedding_fallback=True,
        )

        metrics = score_smoke_case(case=case, response=response)

        self.assertEqual(metrics["status"], "fail")
        self.assertIn("embedding_fallback", metrics["failure_categories"])

    def test_gold_matching_normalizes_pdf_ligatures(self) -> None:
        evidence = make_evidence(
            text="The method uses a compound coef\ufb01cient to scale network width, depth, and resolution."
        )

        self.assertTrue(
            evidence_matches_gold(
                evidence,
                {
                    "document": "network.docx",
                    "text_contains": ["compound coefficient", "width, depth, and resolution"],
                },
            )
        )
        self.assertEqual(answer_point_coverage(["coefficient"], "compound coef\ufb01cient"), 1.0)

    def test_gold_matching_accepts_surface_variants_for_mechanism_phrases(self) -> None:
        efficient = make_evidence(
            text="The method uniformly scales all dimensions of depth/width/resolution using a compound coefficient."
        )
        lora = make_evidence(
            text=(
                "LoRA freezes the pretrained model weights and injects trainable "
                "rank decomposition matrices into each layer."
            )
        )

        self.assertTrue(
            evidence_matches_gold(
                efficient,
                {
                    "document": "network.docx",
                    "text_contains": ["compound coefficient", "width, depth, and resolution"],
                },
            )
        )
        self.assertTrue(
            evidence_matches_gold(
                lora,
                {
                    "document": "network.docx",
                    "text_contains": ["freezing the pre-trained model weights", "rank decomposition matrices"],
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
