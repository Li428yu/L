from __future__ import annotations

import unittest

from backend.app.agent_parts.citation_stability import AgentCitationStabilityMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.models import EvidenceItem


class CitationStabilityHarness(
    AgentCitationStabilityMixin,
    AgentRetrievalFilterMixin,
    AgentTextUtilityMixin,
):
    pass


def make_evidence(
    chunk_id: str,
    *,
    text: str,
    score: float = 0.7,
    page: int = 1,
    document_id: str = "doc-1",
    chunk_type: str = "text",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id="",
        chunk_id=chunk_id,
        document_id=document_id,
        paper_name="paper.pdf",
        page=page,
        page_start=page,
        page_end=page,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        final_score=score,
        text=text,
        quote=text,
        chunk_type=chunk_type,
    )


class CitationStabilityTests(unittest.TestCase):
    def test_table_1_candidate_replaces_table_2_for_complexity_question(self) -> None:
        harness = CitationStabilityHarness()
        question = "How does Table 1 compare self-attention, recurrent, and convolutional layers?"
        wrong_table = make_evidence(
            "table-2",
            page=8,
            chunk_type="table",
            text="Table 2 reports BLEU results on WMT 2014 with 28.4 and 41.8.",
        )
        direct_table = make_evidence(
            "table-1",
            page=6,
            chunk_type="table",
            text=(
                "Table 1 compares layer types. It lists Complexity per Layer, "
                "Sequential Operations, and Maximum Path Length for self-attention, "
                "recurrent, and convolutional layers."
            ),
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[wrong_table],
            candidates=[wrong_table, direct_table],
            limit=1,
        )

        self.assertEqual(selected[0].chunk_id, "table-1")
        self.assertEqual(selected[0].page, 6)
        self.assertIn("Maximum Path Length", selected[0].quote)

    def test_bleu_question_prefers_table_2_over_complexity_table(self) -> None:
        harness = CitationStabilityHarness()
        question = "What BLEU results does the Attention paper report on WMT 2014?"
        table_1 = make_evidence(
            "table-1",
            page=6,
            chunk_type="table",
            text="Table 1 lists Complexity per Layer, Sequential Operations, and Maximum Path Length.",
        )
        table_2 = make_evidence(
            "table-2",
            page=8,
            chunk_type="table",
            text="Table 2 reports BLEU on WMT 2014: English-to-German 28.4 and English-to-French 41.8.",
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[table_1],
            candidates=[table_1, table_2],
            limit=1,
        )

        self.assertEqual(selected[0].chunk_id, "table-2")
        self.assertEqual(selected[0].page, 8)
        self.assertIn("28.4", selected[0].quote)
        self.assertIn("41.8", selected[0].quote)

    def test_table_1_excerpt_keeps_position_and_gets_focused_quote(self) -> None:
        harness = CitationStabilityHarness()
        question = "How does Table 1 compare self-attention, recurrent, and convolutional layers?"
        selected_excerpt = make_evidence(
            "selected-table-1",
            page=6,
            text=(
                "As noted in Table 1, a self-attention layer connects all positions with a "
                "constant number of sequentially executed operations, whereas a recurrent "
                "layer requires O(n) sequential operations. Hence we also compare the maximum "
                "path length between any two input and output positions."
            ),
        )
        selected_excerpt.quote = "Table 2 reports BLEU results on WMT 2014 with 28.4 and 41.8."
        full_table = make_evidence(
            "full-table-1",
            page=6,
            chunk_type="table",
            text=(
                "Table 1: Maximum path lengths, per-layer complexity and minimum number of "
                "sequential operations for self-attention, recurrent, and convolutional layers."
            ),
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[selected_excerpt, full_table],
            candidates=[selected_excerpt, full_table],
            limit=2,
        )

        self.assertEqual(selected[0].chunk_id, "selected-table-1")
        self.assertIn("Table 1", selected[0].quote)
        self.assertIn("sequential", selected[0].quote)
        self.assertNotIn("Table 2", selected[0].quote)

    def test_gpt2_context_candidate_replaces_generic_selected_quote(self) -> None:
        harness = CitationStabilityHarness()
        question = "How does the GPT-2 paper describe vocabulary and context length settings?"
        generic = make_evidence(
            "generic",
            page=2,
            text="GPT-2 uses language modeling on WebText.",
        )
        direct = make_evidence(
            "bpe-context",
            page=4,
            text=(
                "GPT-2 uses Byte Pair Encoding (BPE). The vocabulary is expanded to 50,257. "
                "The context size is increased to 1024 tokens."
            ),
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[generic],
            candidates=[generic, direct],
            limit=1,
        )

        self.assertEqual(selected[0].chunk_id, "bpe-context")
        self.assertEqual(selected[0].page, 4)
        self.assertIn("50,257", selected[0].quote)
        self.assertIn("1024", selected[0].quote)

    def test_wmt_training_data_candidate_is_promoted_over_bleu_result(self) -> None:
        harness = CitationStabilityHarness()
        question = "What WMT 2014 training data sizes are reported in the Attention paper?"
        bleu_result = make_evidence(
            "bleu-result",
            page=8,
            text="Table 2 reports BLEU on WMT 2014: English-to-German 28.4 and English-to-French 41.8.",
        )
        training_data = make_evidence(
            "training-data",
            page=7,
            text=(
                "We trained on the standard WMT 2014 English-German dataset consisting of about "
                "4.5 million sentence pairs. For English-French, we used the WMT 2014 "
                "English-French dataset consisting of 36M sentences."
            ),
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[bleu_result, training_data],
            candidates=[bleu_result, training_data],
            limit=2,
        )

        self.assertEqual(selected[0].chunk_id, "training-data")
        self.assertEqual(selected[0].page, 7)
        self.assertIn("4.5 million", selected[0].quote)
        self.assertIn("36M", selected[0].quote)

    def test_replacement_preserves_other_document_coverage(self) -> None:
        harness = CitationStabilityHarness()
        question = "How does the GPT-2 paper describe vocabulary and context length settings?"
        generic_same_doc = make_evidence(
            "generic-gpt2",
            document_id="doc-gpt2",
            text="GPT-2 uses language modeling on WebText.",
        )
        other_doc = make_evidence(
            "other-doc",
            document_id="doc-clip",
            text="CLIP uses image-text pairs and contrastive learning.",
        )
        direct = make_evidence(
            "bpe-context",
            document_id="doc-gpt2",
            text="GPT-2 uses BPE with a 50,257 vocabulary and a 1024 token context length.",
        )

        selected = harness._stabilize_final_evidence_citations(
            question=question,
            selected=[generic_same_doc, other_doc],
            candidates=[generic_same_doc, other_doc, direct],
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in selected], ["bpe-context", "other-doc"])
        self.assertEqual({item.document_id for item in selected}, {"doc-gpt2", "doc-clip"})


if __name__ == "__main__":
    unittest.main()
