from __future__ import annotations

from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalScoringMixin:
    def _candidate_has_embedded_noise(self, item: EvidenceItem) -> bool:
        text = self._sanitize_evidence_text(item.text)
        return (
            self._looks_like_front_matter_noise(text)
            or self._looks_like_field_or_metadata_sentence(text)
            or self._looks_like_submission_or_assignment_noise(text)
        )

    def _evidence_noise_penalty_for_soft_intent(
        self,
        *,
        question: str,
        text: str,
        section: str,
        soft_intent: dict[str, Any],
    ) -> float:
        intent = str(soft_intent.get("intent") or "")
        operation = str(soft_intent.get("operation") or "")
        roles = set(str(role) for role in soft_intent.get("preferred_roles", []))
        excluded = set(str(role) for role in soft_intent.get("exclude_roles", []))
        penalty = 0.0

        metadata_requested = intent == "field_lookup_question" or operation == "extract"
        reference_requested = intent == "reference_question" or "reference" in roles
        if self._looks_like_front_matter_noise(text):
            penalty += 0.3 if metadata_requested else 1.8
        if self._looks_like_field_or_metadata_sentence(text) and not metadata_requested:
            penalty += 1.0
        if section == "References" or self._looks_like_reference_section_text(text):
            penalty += 0.0 if reference_requested else 1.45
        if self._looks_like_submission_or_assignment_noise(text) and not self._question_allows_submission_details(question):
            penalty += 1.45 if "submission" in excluded or soft_intent.get("scope") == "whole_document" else 0.7
        if self._looks_like_code_heavy_text(text) and not self._question_asks_for_code_details(question):
            penalty += 1.0 if "code" in excluded else 0.55
        if self._is_table_like_text(text) and not self._looks_like_table_question(question):
            penalty += 0.95 if "table" in excluded else 0.45
        return penalty

    def _soft_focus_score(self, soft_intent: dict[str, Any], text: str) -> float:
        terms = [str(item).strip() for item in soft_intent.get("focus", []) if str(item).strip()]
        if not terms:
            return 0.0
        normalized = text.lower()
        hits = 0
        for term in terms:
            lowered = term.lower()
            if term in text or lowered in normalized:
                hits += 1
        return min(1.0, hits / max(len(terms), 1))

    def _soft_role_score(
        self,
        *,
        soft_intent: dict[str, Any],
        text: str,
        section: str,
        index: int,
        total: int,
    ) -> float:
        roles = self._normalized_preferred_roles(soft_intent)
        if not roles:
            return 0.0
        role_scores = self._semantic_role_scores(text=text, section=section, index=index, total=total)
        return max((role_scores.get(role, 0.0) for role in roles), default=0.0)

    def _normalized_preferred_roles(self, soft_intent: dict[str, Any]) -> list[str]:
        aliases = {
            "result": "claim",
            "results": "claim",
            "finding": "claim",
            "findings": "claim",
            "method": "approach",
            "methods": "approach",
            "requirement": "approach",
            "step": "approach",
            "implementation": "approach",
            "risk": "caveat",
            "limitation": "caveat",
            "bibliography": "reference",
            "metadata": "field",
        }
        roles: list[str] = []
        for raw in soft_intent.get("preferred_roles", []):
            role = aliases.get(str(raw).strip(), str(raw).strip())
            if role and role not in roles:
                roles.append(role)
        return roles

    def _score_rows_by_semantic_role(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, list[tuple[float, dict[str, Any]]]]:
        scored: dict[str, list[tuple[float, dict[str, Any]]]] = {
            "purpose": [],
            "approach": [],
            "claim": [],
            "conclusion": [],
            "caveat": [],
            "example": [],
            "informative": [],
        }
        total = max(len(rows), 1)
        for index, row in enumerate(rows):
            text = str(row.get("text", ""))
            if not text.strip() or self._looks_like_front_matter_noise(text):
                continue
            if self._looks_like_code_heavy_text(text):
                continue
            if self._looks_like_submission_or_assignment_noise(text):
                continue
            if self._looks_like_reference_section_text(text) and self._reference_marker_count(text) >= 2:
                continue
            metadata = row.get("metadata") or {}
            section = str(metadata.get("section") or "")
            role_scores = self._semantic_role_scores(text=text, section=section, index=index, total=total)
            informative_score = self._readable_text_score(text) + self._overview_structure_score(text)
            if index <= 2:
                informative_score += 0.16
            scored["informative"].append((informative_score, row))
            for role, score in role_scores.items():
                if score > 0:
                    scored.setdefault(role, []).append((score, row))

        for role in scored:
            scored[role].sort(key=lambda item: item[0], reverse=True)
        return scored

    def _semantic_role_scores(
        self,
        *,
        text: str,
        section: str,
        index: int,
        total: int,
    ) -> dict[str, float]:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        role_keywords = {
            "purpose": [
                "摘要",
                "abstract",
                "本文",
                "本研究",
                "本文围绕",
                "本文旨在",
                "研究目的",
                "主要讨论",
                "主要介绍",
                "目的",
                "aim",
                "purpose",
                "in this paper",
                "in this work",
            ],
            "approach": [
                "采用",
                "方法",
                "通过",
                "基于",
                "设计",
                "构建",
                "实验",
                "分析",
                "模型",
                "框架",
                "流程",
                "method",
                "approach",
                "experiment",
                "model",
                "framework",
            ],
            "claim": [
                "认为",
                "提出",
                "指出",
                "发现",
                "表明",
                "结果",
                "显示",
                "证明",
                "suggest",
                "show",
                "find",
                "propose",
                "result",
            ],
            "conclusion": [
                "结论",
                "总结",
                "总体而言",
                "综上",
                "启示",
                "建议",
                "展望",
                "conclusion",
                "discussion",
                "future work",
            ],
            "caveat": [
                "局限",
                "不足",
                "风险",
                "挑战",
                "限制",
                "问题",
                "需要注意",
                "隐私",
                "偏差",
                "偏见",
                "诚信",
                "责任",
                "依赖",
                "边界",
                "代价",
                "limitation",
                "risk",
                "challenge",
            ],
            "example": [
                "例如",
                "案例",
                "场景",
                "应用",
                "实践",
                "sample",
                "case",
                "scenario",
                "application",
            ],
        }
        section_bonus = {
            "Abstract": {"purpose": 1.0, "claim": 0.35},
            "Introduction": {"purpose": 0.35, "claim": 0.15},
            "Methods": {"approach": 0.9},
            "Results": {"claim": 0.9},
            "Discussion": {"claim": 0.4, "caveat": 0.35},
            "Conclusion": {"conclusion": 1.0, "claim": 0.45},
            "Limitations": {"caveat": 1.0},
            "FutureWork": {"conclusion": 0.45, "caveat": 0.45},
        }.get(section, {})
        position_bonus = 0.35 if index <= max(2, int(total * 0.08)) else 0.0
        scores: dict[str, float] = {}
        for role, keywords in role_keywords.items():
            hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
            score = hits * 0.35 + section_bonus.get(role, 0.0)
            if role == "purpose":
                score += position_bonus
            if role in {"conclusion", "caveat"} and index >= int(total * 0.65):
                score += 0.2
            if score > 0:
                scores[role] = score
        return scores

