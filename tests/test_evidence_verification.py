from __future__ import annotations

import unittest

from backend.app.agent_parts.retrieval_evidence import AgentRetrievalEvidenceMixin
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.agent_parts.verification import AgentVerificationMixin
from backend.app.models import EvidenceItem


class EvidenceJudgeHarness(
    AgentVerificationMixin,
    AgentRetrievalEvidenceMixin,
    AgentTextUtilityMixin,
):
    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.statuses: list[str] = []

    def _emit_status(self, text: str) -> None:
        self.statuses.append(text)

    def _strict_evidence_judge_question(self, question: str) -> bool:
        return False

    def _looks_like_table_question(self, question: str) -> bool:
        return False

    def _judge_single_evidence(self, **kwargs):
        item = kwargs["item"]
        verdict = self.verdicts[item.chunk_id]
        return {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "verdict": verdict,
            "confidence": 0.9,
            "reason": f"forced {verdict}",
            "retrieval_strategy": kwargs.get("retrieval_strategy", ""),
        }


def make_evidence(
    chunk_id: str,
    *,
    citation_id: str,
    score: float = 0.4,
    text: str = "attention mechanism improves sequence modeling",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        text=text,
        quote=text,
    )


class EvidenceVerificationTests(unittest.TestCase):
    def test_direct_evidence_is_retained_and_renumbered(self) -> None:
        judge = EvidenceJudgeHarness({"chunk-9": "direct"})
        state = {
            "question": "How does attention improve sequence modeling?",
            "evidence": [
                make_evidence("chunk-9", citation_id="OLD", score=0.3),
            ],
            "runtime": [],
            "retrieval_strategy": "hybrid",
        }

        result = judge._judge_evidence(state)

        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0].chunk_id, "chunk-9")
        self.assertEqual(result["evidence"][0].citation_id, "E1")
        self.assertEqual(result["evidence_judgments"][0]["citation_id"], "E1")

    def test_rejected_candidate_is_removed_when_other_evidence_passes(self) -> None:
        judge = EvidenceJudgeHarness({"weak": "reject", "strong": "direct"})
        state = {
            "question": "How does attention improve sequence modeling?",
            "evidence": [
                make_evidence("weak", citation_id="E1", score=0.99, text="unrelated appendix text"),
                make_evidence("strong", citation_id="E2", score=0.5),
            ],
            "runtime": [],
            "retrieval_strategy": "hybrid",
        }

        result = judge._judge_evidence(state)

        self.assertEqual([item.chunk_id for item in result["evidence"]], ["strong"])
        self.assertEqual(result["evidence"][0].citation_id, "E1")
        self.assertEqual(
            [judgment["verdict"] for judgment in result["evidence_judgments"]],
            ["reject", "direct"],
        )

    def test_all_rejected_candidates_are_not_restored(self) -> None:
        judge = EvidenceJudgeHarness({"weak-1": "reject", "weak-2": "reject"})
        state = {
            "question": "How does attention improve sequence modeling?",
            "evidence": [
                make_evidence("weak-1", citation_id="E1", score=0.99, text="unrelated appendix text"),
                make_evidence("weak-2", citation_id="E2", score=0.95, text="another unrelated paragraph"),
            ],
            "runtime": [],
            "retrieval_strategy": "hybrid",
        }

        result = judge._judge_evidence(state)

        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["evidence_quality"], "none")
        self.assertEqual(
            [judgment["verdict"] for judgment in result["evidence_judgments"]],
            ["reject", "reject"],
        )


if __name__ == "__main__":
    unittest.main()
