from __future__ import annotations

import unittest
from typing import Any

from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class DummyVectorStore:
    def __init__(self, rows_by_document: dict[str, list[dict[str, Any]]]) -> None:
        self.rows_by_document = rows_by_document

    def get_document_chunks(self, document_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        return self.rows_by_document.get(document_id, [])[:limit]


class EvidenceExpansionHarness(AgentRetrievalMixin, AgentTextUtilityMixin):
    def __init__(self, rows_by_document: dict[str, list[dict[str, Any]]]) -> None:
        self.vector_store = DummyVectorStore(rows_by_document)

    def _looks_like_reference_question(self, question: str) -> bool:
        return False

    def _looks_like_document_wide_question(self, question: str) -> bool:
        return False


def make_row(
    chunk_id: str,
    *,
    text: str,
    page: int = 1,
    section: str = "Results",
    chunk_type: str = "text",
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "paper_name": "paper.pdf",
            "page": page,
            "page_start": page,
            "page_end": page,
            "section": section,
            "source": "paper.pdf",
            "file_hash": "hash",
            "chunk_type": chunk_type,
        },
    }


def make_anchor(chunk_id: str, *, score: float = 0.82, page: int = 1) -> EvidenceItem:
    return EvidenceItem(
        citation_id="",
        chunk_id=chunk_id,
        document_id="doc-1",
        paper_name="paper.pdf",
        page=page,
        page_start=page,
        page_end=page,
        section="Results",
        source="paper.pdf",
        file_hash="hash",
        score=score,
        final_score=score,
        score_source="rrf_fusion",
        text="The experiment section discusses nearby ImageNet results.",
        quote="The experiment section discusses nearby ImageNet results.",
    )


class EvidenceExpansionTests(unittest.TestCase):
    def test_neighbor_chunk_expands_to_sentence_evidence_with_target_phrase(self) -> None:
        rows = [
            make_row("chunk-0", text="Background text about scaling models.", page=1),
            make_row("chunk-1", text="The experiment section discusses nearby ImageNet results.", page=1),
            make_row(
                "chunk-2",
                text=(
                    "EfficientNet-B7 achieves 84.3% top-1 accuracy on ImageNet "
                    "while being 8.4x smaller and 6.1x faster than GPipe."
                ),
                page=1,
            ),
        ]
        harness = EvidenceExpansionHarness({"doc-1": rows})

        expanded = harness._expand_retrieval_candidates(
            question="What ImageNet result does EfficientNet-B7 report?",
            candidates=[make_anchor("chunk-1", page=1)],
            limit=8,
        )

        expanded_text = "\n".join(item.text for item in expanded)
        self.assertIn("84.3%", expanded_text)
        self.assertTrue(any(item.chunk_type in {"sentence", "phrase"} for item in expanded))
        self.assertTrue(any(item.score_source.startswith("expanded_") for item in expanded))
        self.assertTrue(expanded[0].score_source.startswith("expanded_"))

    def test_table_chunk_expands_to_table_row_evidence(self) -> None:
        rows = [
            make_row(
                "chunk-table",
                text=(
                    "Model | Accuracy | Params\n"
                    "Model A | 88.0% | 12M\n"
                    "Model B | 91.2% | 9M\n"
                    "Model C | 89.4% | 11M"
                ),
                page=3,
                section="Results",
                chunk_type="table",
            )
        ]
        harness = EvidenceExpansionHarness({"doc-1": rows})

        expanded = harness._expand_retrieval_candidates(
            question="What accuracy does Model B report?",
            candidates=[make_anchor("chunk-table", page=3)],
            limit=6,
        )

        table_rows = [item for item in expanded if item.chunk_type == "table_row"]
        self.assertTrue(table_rows)
        self.assertTrue(any("Model B" in item.text and "91.2%" in item.text for item in table_rows))

    def test_document_level_expansion_finds_distant_mechanism_phrase(self) -> None:
        rows = [
            make_row(
                "chunk-1",
                text="EfficientNet improves accuracy and efficiency in the abstract.",
                page=1,
                section="Abstract",
            ),
            make_row("chunk-2", text="Unrelated implementation detail.", page=2, section="Introduction"),
            make_row("chunk-3", text="More unrelated background.", page=3, section="Introduction"),
            make_row(
                "chunk-13",
                text=(
                    "In this paper, we propose a new compound scaling method, "
                    "which uses a compound coefficient to uniformly scale network "
                    "width, depth, and resolution in a principled way."
                ),
                page=4,
                section="Methods",
            ),
        ]
        harness = EvidenceExpansionHarness({"doc-1": rows})

        expanded = harness._expand_retrieval_candidates(
            question="How does EfficientNet pursue efficiency?",
            candidates=[make_anchor("chunk-1", page=1)],
            limit=8,
        )

        expanded_text = "\n".join(item.text for item in expanded)
        self.assertIn("compound coefficient", expanded_text)
        self.assertIn("width, depth, and resolution", expanded_text)
        self.assertTrue(any("expansion_source:document" in item.quality_reasons for item in expanded))


if __name__ == "__main__":
    unittest.main()
