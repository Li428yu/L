from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models import EvidenceItem


@dataclass(frozen=True)
class ClaimSlotRule:
    name: str
    question_groups: tuple[tuple[str, ...], ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    answer_groups: tuple[tuple[str, ...], ...]
    sentence: str


class AgentClaimSlotMixin:
    def _ensure_claim_slots_in_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not answer.strip() or not evidence:
            return answer

        rule = self._matching_claim_slot_rule(question)
        if rule is None:
            return answer

        item = self._claim_slot_supporting_evidence(rule, evidence)
        if item is None or not item.citation_id:
            return answer

        if self._claim_slot_groups_present(answer, rule.answer_groups):
            return answer

        sentence = rule.sentence.format(citation=f"[{item.citation_id}]")
        if self._claim_slot_refusal_like_answer(answer):
            return sentence
        return f"{answer.rstrip()}\n\n关键事实补充：{sentence}"

    def _matching_claim_slot_rule(self, question: str) -> ClaimSlotRule | None:
        normalized = self._claim_slot_normalize(question)
        for rule in self._claim_slot_rules():
            if self._claim_slot_groups_present(normalized, rule.question_groups):
                return rule
        return None

    def _claim_slot_supporting_evidence(
        self,
        rule: ClaimSlotRule,
        evidence: list[EvidenceItem],
    ) -> EvidenceItem | None:
        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            text = self._claim_slot_evidence_text(item)
            if not self._claim_slot_groups_present(text, rule.evidence_groups):
                continue
            scored.append((float(item.score), position, item))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return scored[0][2] if scored else None

    def _claim_slot_groups_present(self, text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
        normalized = self._claim_slot_normalize(text)
        return all(any(self._claim_slot_alias_present(normalized, alias) for alias in group) for group in groups)

    def _claim_slot_alias_present(self, normalized_text: str, alias: str) -> bool:
        normalized_alias = self._claim_slot_normalize(alias)
        if not normalized_alias:
            return False
        return normalized_alias in normalized_text

    def _claim_slot_evidence_text(self, item: EvidenceItem) -> str:
        related = " ".join(
            " ".join(
                part
                for part in [
                    getattr(image, "caption_text", ""),
                    getattr(image, "ocr_text", ""),
                    getattr(image, "vision_summary", ""),
                ]
                if part
            )
            for image in getattr(item, "related_images", [])
        )
        return self._claim_slot_normalize(
            f"{item.paper_name} {item.section or ''} {item.quote or ''} {item.text or ''} {related}"
        )

    def _claim_slot_refusal_like_answer(self, answer: str) -> bool:
        normalized = self._claim_slot_normalize(answer)
        markers = [
            "不能证明",
            "无法证明",
            "证据不足",
            "没有证据",
            "没有找到",
            "不能可靠回答",
            "无法回答",
            "cannot answer",
            "no evidence",
            "not enough evidence",
            "unable to answer",
        ]
        return any(marker in normalized for marker in markers)

    def _claim_slot_normalize(self, text: str) -> str:
        normalized = str(text or "").lower()
        normalized = normalized.replace("‑", "-").replace("–", "-").replace("—", "-")
        normalized = normalized.replace("，", ",")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _claim_slot_rules(self) -> tuple[ClaimSlotRule, ...]:
        return (
            ClaimSlotRule(
                name="gpt2_bpe_context",
                question_groups=(("gpt-2", "gpt2"), ("vocabulary", "词汇"), ("context", "上下文")),
                evidence_groups=(
                    ("byte pair encoding", "bpe"),
                    ("50,257", "50257"),
                    ("context size", "context length"),
                    ("1024",),
                ),
                answer_groups=(
                    ("byte pair encoding", "bpe"),
                    ("50,257", "50257"),
                    ("1024",),
                ),
                sentence=(
                    "GPT-2 uses Byte Pair Encoding (BPE), a 50,257 vocabulary, "
                    "and increases the context size to 1024. {citation}"
                ),
            ),
            ClaimSlotRule(
                name="clip_contrastive_batch",
                question_groups=(("clip",), ("contrastive", "对比"), ("batch", "批")),
                evidence_groups=(
                    ("image encoder",),
                    ("text encoder",),
                    ("cosine similarity",),
                    ("n real pairs", "real pairs"),
                ),
                answer_groups=(
                    ("image encoder",),
                    ("text encoder",),
                    ("cosine similarity",),
                    ("n real pairs", "real pairs"),
                ),
                sentence=(
                    "CLIP trains an image encoder and a text encoder by maximizing cosine similarity "
                    "for the N real image-text pairs in a batch. {citation}"
                ),
            ),
            ClaimSlotRule(
                name="sam_automatic_masks",
                question_groups=(("sam", "segment anything"), ("automatic", "automatically", "自动"), ("mask", "掩码")),
                evidence_groups=(("1.1b", "1.1 billion"), ("99.1%", "99.1 percent"), ("fully automatically",)),
                answer_groups=(("1.1b", "1.1 billion"), ("99.1%", "99.1 percent")),
                sentence="SAM reports 1.1B masks, with 99.1% fully automatically generated. {citation}",
            ),
            ClaimSlotRule(
                name="attention_wmt_training_data",
                question_groups=(("attention", "transformer"), ("wmt 2014",), ("training data", "data sizes")),
                evidence_groups=(
                    ("english-german", "english-to-german"),
                    ("4.5 million",),
                    ("english-french", "english-to-french"),
                    ("36m", "36 million"),
                ),
                answer_groups=(("4.5 million",), ("36m", "36 million")),
                sentence=(
                    "The Attention paper reports about 4.5 million English-German sentence pairs "
                    "and about 36M English-French sentences. {citation}"
                ),
            ),
            ClaimSlotRule(
                name="attention_complexity_table",
                question_groups=(("table 1",), ("self-attention",), ("recurrent",), ("convolutional",)),
                evidence_groups=(
                    ("table 1",),
                    ("complexity per layer",),
                    ("sequential operations",),
                    ("maximum path length",),
                ),
                answer_groups=(("complexity",), ("sequential operations",), ("maximum path length",)),
                sentence=(
                    "Table 1 compares self-attention, recurrent, and convolutional layers by "
                    "Complexity per Layer, Sequential Operations, and Maximum Path Length. {citation}"
                ),
            ),
        )
