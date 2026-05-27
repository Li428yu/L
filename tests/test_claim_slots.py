from __future__ import annotations

import unittest

from backend.app.agent_parts.claim_slots import AgentClaimSlotMixin
from backend.app.models import EvidenceItem


class ClaimSlotHarness(AgentClaimSlotMixin):
    pass


def make_evidence(
    text: str,
    *,
    citation_id: str = "E1",
    score: float = 0.8,
    quote: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=score,
        text=text,
        quote=quote or text,
    )


class ClaimSlotTests(unittest.TestCase):
    def test_adds_missing_gpt2_vocabulary_and_context_claims(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens. "
                "The model uses byte pair encoding (BPE)."
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="How does the GPT-2 paper describe vocabulary and context length settings?",
            answer="GPT-2 changes tokenization and context settings. [E1]",
            evidence=evidence,
        )

        self.assertIn("Byte Pair Encoding", answer)
        self.assertIn("50,257", answer)
        self.assertIn("1024", answer)
        self.assertIn("[E1]", answer)

    def test_replaces_wrong_refusal_when_sam_mask_claims_are_supported(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "Our dataset contains 1.1B masks. Of these masks, 99.1% were fully automatically generated."
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="What representative numbers does SAM report for automatically generated masks?",
            answer="没有找到自动生成掩码的代表性数字，无法回答。",
            evidence=evidence,
        )

        self.assertEqual("SAM reports 1.1B masks, with 99.1% fully automatically generated. [E1]", answer)

    def test_does_not_add_claims_without_direct_evidence_support(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [make_evidence("GPT-2 uses language modeling on WebText.")]

        answer = harness._ensure_claim_slots_in_answer(
            question="How does the GPT-2 paper describe vocabulary and context length settings?",
            answer="GPT-2 uses language modeling on WebText. [E1]",
            evidence=evidence,
        )

        self.assertEqual("GPT-2 uses language modeling on WebText. [E1]", answer)


if __name__ == "__main__":
    unittest.main()
