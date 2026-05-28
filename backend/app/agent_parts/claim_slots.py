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
            answer = self._ensure_generic_resource_numbers_in_answer(
                question=question,
                answer=answer,
                evidence=evidence,
            )
            return self._ensure_generic_factor_list_in_answer(
                question=question,
                answer=answer,
                evidence=evidence,
            )

        item = self._claim_slot_supporting_evidence(rule, evidence)
        if item is None or not item.citation_id:
            return answer

        if self._claim_slot_groups_present(answer, rule.answer_groups):
            return answer

        sentence = rule.sentence.format(citation=f"[{item.citation_id}]")
        if self._claim_slot_refusal_like_answer(answer):
            return sentence
        return f"{answer.rstrip()}\n\n关键事实补充：{sentence}"

    def _ensure_generic_resource_numbers_in_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not self._claim_slot_resource_question(question):
            return answer
        answer_normalized = self._claim_slot_normalize(answer)
        best: tuple[tuple[int, int, float, int], EvidenceItem, str] | None = None
        for item, sentence in self._claim_slot_ranked_sentences(evidence):
            sentence_normalized = self._claim_slot_normalize(sentence)
            if not self._claim_slot_resource_sentence(sentence_normalized):
                continue
            numbers = self._claim_slot_numbers(sentence)
            missing_numbers = [
                number
                for number in numbers
                if self._claim_slot_normalize(number) not in answer_normalized
            ]
            if not missing_numbers or not item.citation_id:
                continue
            resource_hits = sum(1 for term in self._claim_slot_resource_terms() if term in sentence_normalized)
            score = (len(set(missing_numbers)), resource_hits, float(item.score or 0.0), len(sentence))
            if best is None or score > best[0]:
                best = (score, item, sentence)
        if best is not None:
            _, item, sentence = best
            return self._claim_slot_append_key_fact(answer, sentence, item)
        return answer

    def _ensure_generic_factor_list_in_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not self._claim_slot_factor_question(question):
            return answer
        answer_normalized = self._claim_slot_normalize(answer)
        best: tuple[tuple[int, int, float, int], EvidenceItem, str] | None = None
        for item, sentence in self._claim_slot_ranked_sentences(evidence):
            sentence_normalized = self._claim_slot_normalize(sentence)
            factor_hits = [
                term
                for term in self._claim_slot_factor_terms()
                if term in sentence_normalized
            ]
            missing_hits = [term for term in factor_hits if term not in answer_normalized]
            if len(set(factor_hits)) < 4 or len(set(missing_hits)) < 2 or not item.citation_id:
                continue
            score = (len(set(missing_hits)), len(set(factor_hits)), float(item.score or 0.0), len(sentence))
            if best is None or score > best[0]:
                best = (score, item, sentence)
        if best is not None:
            _, item, sentence = best
            return self._claim_slot_append_key_fact(answer, sentence, item)
        return answer

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

    def _claim_slot_ranked_sentences(self, evidence: list[EvidenceItem]) -> list[tuple[EvidenceItem, str]]:
        rows: list[tuple[float, int, EvidenceItem, str]] = []
        for position, item in enumerate(evidence):
            text = " ".join(part for part in [item.quote, item.text] if part)
            for sentence in self._claim_slot_sentences(text):
                rows.append((float(item.score or 0.0), position, item, sentence))
        rows.sort(key=lambda row: (row[0], -row[1], len(row[3])), reverse=True)
        return [(item, sentence) for _, _, item, sentence in rows]

    def _claim_slot_sentences(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?;])\s+|\s+[•-]\s+", normalized)
        return [self._claim_slot_trim_sentence(part) for part in parts if part.strip()]

    def _claim_slot_trim_sentence(self, sentence: str, limit: int = 420) -> str:
        sentence = re.sub(r"\s+", " ", str(sentence or "")).strip(" -;")
        if len(sentence) <= limit:
            return sentence
        boundary = sentence.rfind(" ", 0, limit)
        if boundary < int(limit * 0.6):
            boundary = limit
        return sentence[:boundary].rstrip(" ,;") + "..."

    def _claim_slot_append_key_fact(self, answer: str, sentence: str, item: EvidenceItem) -> str:
        citation = f"[{item.citation_id}]"
        supplement = sentence if citation in sentence else f"{sentence} {citation}"
        return f"{answer.rstrip()}\n\n关键事实补充：{supplement}"

    def _claim_slot_resource_question(self, question: str) -> bool:
        normalized = self._claim_slot_normalize(question)
        terms = [
            "efficiency",
            "efficient",
            "resource",
            "memory",
            "parameter",
            "parameters",
            "compute",
            "storage",
            "throughput",
            "latency",
            "speedup",
            "faster",
            "smaller",
            "cost",
        ]
        return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)

    def _claim_slot_resource_sentence(self, normalized_sentence: str) -> bool:
        return any(term in normalized_sentence for term in self._claim_slot_resource_terms()) and bool(
            self._claim_slot_numbers(normalized_sentence)
        )

    def _claim_slot_resource_terms(self) -> tuple[str, ...]:
        return (
            "trainable parameter",
            "trainable parameters",
            "memory requirement",
            "memory",
            "storage",
            "compute",
            "throughput",
            "latency",
            "speedup",
            "faster",
            "smaller",
            "efficient",
            "efficiency",
        )

    def _claim_slot_numbers(self, text: str) -> list[str]:
        return re.findall(
            r"\b(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:x|times|%|m|b|k|million|billion))\b",
            str(text or "").lower(),
        )

    def _claim_slot_factor_question(self, question: str) -> bool:
        normalized = self._claim_slot_normalize(question)
        topic_markers = ["design", "training", "factor", "factors", "ablation", "component", "components"]
        ask_markers = ["important", "identify", "affect", "effect", "learned representation", "representation"]
        return any(marker in normalized for marker in topic_markers) and any(
            marker in normalized for marker in ask_markers
        )

    def _claim_slot_factor_terms(self) -> tuple[str, ...]:
        return (
            "data augmentation",
            "augmentations",
            "composition",
            "critical",
            "important",
            "learnable",
            "nonlinear",
            "transformation",
            "loss",
            "batch",
            "batch size",
            "training steps",
            "temperature",
            "component",
            "components",
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
        return ()
