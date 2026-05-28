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
    def test_does_not_add_document_specific_claims_from_runtime_rules(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "The method expands its vocabulary to 50,257 entries and increases context size to 1024 tokens."
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="How does this paper describe vocabulary and context length settings?",
            answer="The paper changes tokenization and context settings. [E1]",
            evidence=evidence,
        )

        self.assertEqual("The paper changes tokenization and context settings. [E1]", answer)

    def test_does_not_replace_refusal_with_sample_specific_template(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "The dataset contains 1.1B annotations. Of these annotations, 99.1% were automatically generated."
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="What representative numbers does the paper report for automatically generated annotations?",
            answer="没有找到自动生成标注的代表性数字，无法回答。",
            evidence=evidence,
        )

        self.assertEqual("没有找到自动生成标注的代表性数字，无法回答。", answer)

    def test_does_not_add_claims_without_direct_evidence_support(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [make_evidence("The model uses language modeling on a web corpus.")]

        answer = harness._ensure_claim_slots_in_answer(
            question="How does the paper describe vocabulary and context length settings?",
            answer="The model uses language modeling on a web corpus. [E1]",
            evidence=evidence,
        )

        self.assertEqual("The model uses language modeling on a web corpus. [E1]", answer)

    def test_adds_missing_resource_numbers_from_direct_evidence(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "The method can reduce the number of trainable parameters by 10,000 times "
                "and the memory requirement by 3 times.",
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="What efficiency benefits does the abstract report?",
            answer="The method greatly reduces trainable parameters and memory use. [E1]",
            evidence=evidence,
        )

        self.assertIn("关键事实补充", answer)
        self.assertIn("10,000 times", answer)
        self.assertIn("3 times", answer)

    def test_resource_supplement_prefers_scaled_numbers_over_footnote_numbers(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "1 This background sentence mentions trainable parameters and latency but has no measured benefit.",
                citation_id="E1",
                score=1.0,
            ),
            make_evidence(
                "The method reduces trainable parameters by 10,000 times and memory requirement by 3 times.",
                citation_id="E2",
                score=0.8,
            ),
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="What efficiency benefits does the abstract report?",
            answer="The method improves efficiency. [E1]",
            evidence=evidence,
        )

        self.assertIn("[E2]", answer)
        self.assertIn("10,000 times", answer)
        self.assertIn("3 times", answer)

    def test_resource_supplement_does_not_match_resource_word_inside_method_name(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "The model reaches 84.3% accuracy while being 8.4x smaller and 6.1x faster.",
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="How does EfficientNet describe its compound scaling method?",
            answer="It uses a compound coefficient to scale width, depth, and resolution. [E1]",
            evidence=evidence,
        )

        self.assertNotIn("关键事实补充", answer)

    def test_adds_missing_factor_list_from_direct_evidence(self) -> None:
        harness = ClaimSlotHarness()
        evidence = [
            make_evidence(
                "We show that composition of data augmentations is critical, a learnable "
                "nonlinear transformation before the loss improves representations, and "
                "larger batch sizes with more training steps help."
            )
        ]

        answer = harness._ensure_claim_slots_in_answer(
            question="Which design or training factors are important for learned representations?",
            answer="Random cropping is important. [E1]",
            evidence=evidence,
        )

        self.assertIn("关键事实补充", answer)
        self.assertIn("composition of data augmentations", answer)
        self.assertIn("training steps", answer)


if __name__ == "__main__":
    unittest.main()
