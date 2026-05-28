from __future__ import annotations

import unittest
from typing import Any

from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin


class DummyVectorStore:
    def __init__(self, rows_by_document: dict[str, list[dict[str, Any]]]) -> None:
        self.rows_by_document = rows_by_document

    def get_document_chunks(self, document_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        return self.rows_by_document.get(document_id, [])[:limit]


class PaperStructureHarness(AgentRetrievalMixin, AgentTextUtilityMixin):
    def __init__(self, rows_by_document: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.vector_store = DummyVectorStore(rows_by_document or {})

    def _looks_like_reference_question(self, question: str) -> bool:
        return False

    def _looks_like_document_wide_question(self, question: str) -> bool:
        return False


def make_row(
    chunk_id: str,
    *,
    text: str,
    section: str = "",
    chunk_type: str = "text",
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "paper_name": "unfamiliar-paper.pdf",
            "page": 1,
            "source": "unfamiliar-paper.pdf",
            "file_hash": "hash",
            "section": section,
            "chunk_type": chunk_type,
        },
    }


class PaperStructureRetrievalTests(unittest.TestCase):
    def test_retrieval_queries_expand_by_generic_structure_roles(self) -> None:
        harness = PaperStructureHarness()
        question = "How does the method architecture work and what ablation results are reported?"

        queries = harness._build_retrieval_queries(
            question=question,
            soft_intent={"preferred_roles": ["approach", "result"]},
            document_ids=["doc-1"],
        )

        joined = " ".join(queries).lower()
        self.assertIn("method", joined)
        self.assertIn("architecture", joined)
        self.assertIn("experimental results", joined)
        self.assertIn("ablation", joined)

    def test_structure_evidence_promotes_matching_unfamiliar_sections(self) -> None:
        rows = [
            make_row(
                "method",
                section="Methods",
                text="We propose a calibration pipeline with a compact model and a simple training objective.",
            ),
            make_row(
                "results",
                section="Results",
                text=(
                    "Experimental results show that the method improves accuracy by 7 points. "
                    "The ablation study confirms that each module contributes to the final score."
                ),
            ),
            make_row(
                "dataset",
                section="Data",
                text="The dataset contains 18,000 labeled examples collected from three public sources.",
            ),
        ]
        harness = PaperStructureHarness({"doc-1": rows})
        question = "What experimental results and ablations are reported?"
        roles = harness._paper_structure_roles_for_question(question)

        evidence = harness._paper_structure_evidence(
            question=question,
            document_ids=["doc-1"],
            top_k=2,
            roles=roles,
        )

        self.assertEqual(evidence[0].chunk_id, "results")
        self.assertEqual(evidence[0].score_source, "paper_structure_claim")
        self.assertIn("Experimental results", evidence[0].quote)

    def test_dataset_questions_use_dataset_structure_terms(self) -> None:
        harness = PaperStructureHarness()

        roles = harness._paper_structure_roles_for_question("What dataset scale was used for training?")
        query_terms = harness._paper_structure_role_terms("dataset", kind="query")

        self.assertIn("dataset", roles)
        self.assertIn("training data", query_terms)
        self.assertIn("data collection", query_terms)

    def test_pdf_text_normalization_handles_ligatures_and_dehyphenation(self) -> None:
        harness = PaperStructureHarness()

        normalized = harness._sanitize_evidence_text(
            "The compound coefﬁcient works with pre-\ntrained weights and ﬂoating point values."
        )

        self.assertIn("coefficient", normalized)
        self.assertIn("pretrained", normalized)
        self.assertIn("floating", normalized)

    def test_salient_evidence_promotes_generic_abstract_efficiency_claims(self) -> None:
        rows = [
            make_row(
                "appendix-table",
                section="Appendix",
                text="Training Optimizer AdamW Batch Size 8 Learning Rate 0.0002 Validation score 71.0.",
            ),
            make_row(
                "abstract-claim",
                section="Abstract",
                text=(
                    "Abstract We introduce a generic adaptation method. "
                    "It is efficient because it uses 120 times fewer trainable parameters "
                    "and reduces GPU memory by 3 times while preserving task quality."
                ),
            ),
        ]
        harness = PaperStructureHarness({"doc-1": rows})
        question = "What efficiency benefits does the abstract report?"
        roles = harness._paper_structure_roles_for_question(question)

        evidence = harness._paper_salient_evidence(
            question=question,
            document_ids=["doc-1"],
            top_k=1,
            roles=roles,
        )

        self.assertEqual(evidence[0].chunk_id, "abstract-claim")
        self.assertEqual(evidence[0].score_source, "paper_salient")
        self.assertIn("trainable parameters", evidence[0].quote)


if __name__ == "__main__":
    unittest.main()
