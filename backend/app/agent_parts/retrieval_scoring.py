from __future__ import annotations

import re
from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalScoringMixin:
    def _paper_structure_signal_definitions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        return {
            "purpose": {
                "query": (
                    "abstract",
                    "introduction",
                    "motivation",
                    "contribution",
                    "main contribution",
                    "proposed approach",
                    "in this paper",
                    "in this work",
                ),
                "evidence": (
                    "abstract",
                    "introduction",
                    "motivation",
                    "contribution",
                    "we propose",
                    "we present",
                    "we introduce",
                    "we describe",
                    "we demonstrate",
                    "in this paper",
                    "in this work",
                ),
                "triggers": (
                    "abstract",
                    "summary",
                    "overview",
                    "main idea",
                    "contribution",
                    "what is this paper",
                    "summarize",
                    "概括",
                    "总结",
                    "主要贡献",
                    "核心贡献",
                    "摘要",
                ),
            },
            "approach": {
                "query": (
                    "method",
                    "approach",
                    "architecture",
                    "algorithm",
                    "framework",
                    "model",
                    "pipeline",
                    "design",
                    "objective",
                    "definition",
                    "mechanism",
                    "implementation",
                ),
                "evidence": (
                    "method",
                    "approach",
                    "architecture",
                    "algorithm",
                    "framework",
                    "model",
                    "pipeline",
                    "design",
                    "objective",
                    "definition",
                    "defined as",
                    "is designed to",
                    "consists of",
                    "allows us",
                    "we propose",
                    "we introduce",
                    "we use",
                ),
                "triggers": (
                    "main idea",
                    "idea",
                    "method",
                    "approach",
                    "architecture",
                    "algorithm",
                    "framework",
                    "model",
                    "pipeline",
                    "design",
                    "objective",
                    "mechanism",
                    "adaptation",
                    "how does",
                    "方法",
                    "机制",
                    "架构",
                    "结构",
                    "模型",
                    "流程",
                    "如何",
                ),
            },
            "claim": {
                "query": (
                    "experimental results",
                    "evaluation",
                    "benchmark",
                    "result",
                    "performance",
                    "metric",
                    "score",
                    "ablation",
                    "comparison",
                    "finding",
                    "efficiency",
                    "benefit",
                    "memory",
                    "parameters",
                    "speed",
                ),
                "evidence": (
                    "result",
                    "results",
                    "experimental results",
                    "evaluation",
                    "benchmark",
                    "performance",
                    "metric",
                    "score",
                    "accuracy",
                    "error",
                    "ablation",
                    "comparison",
                    "achieves",
                    "outperforms",
                    "state-of-the-art",
                    "faster",
                    "smaller",
                    "efficient",
                    "efficiency",
                    "memory",
                    "parameters",
                    "we find",
                    "we show",
                ),
                "triggers": (
                    "result",
                    "results",
                    "performance",
                    "benchmark",
                    "report",
                    "reported",
                    "benefit",
                    "benefits",
                    "efficiency",
                    "efficient",
                    "faster",
                    "smaller",
                    "memory",
                    "parameter",
                    "parameters",
                    "metric",
                    "score",
                    "accuracy",
                    "error",
                    "experiment",
                    "evaluation",
                    "ablation",
                    "结果",
                    "指标",
                    "表现",
                    "实验",
                    "评测",
                    "消融",
                ),
            },
            "dataset": {
                "query": (
                    "dataset",
                    "training data",
                    "corpus",
                    "data collection",
                    "data source",
                    "samples",
                    "annotations",
                    "benchmark dataset",
                    "split",
                    "preprocessing",
                ),
                "evidence": (
                    "dataset",
                    "training data",
                    "corpus",
                    "data collection",
                    "data source",
                    "samples",
                    "annotations",
                    "train",
                    "validation",
                    "test set",
                    "preprocessing",
                ),
                "triggers": (
                    "dataset",
                    "training data",
                    "corpus",
                    "data",
                    "sample",
                    "annotation",
                    "split",
                    "数据集",
                    "训练数据",
                    "语料",
                    "样本",
                    "标注",
                    "规模",
                ),
            },
            "visual": {
                "query": (
                    "figure",
                    "fig.",
                    "caption",
                    "diagram",
                    "overview",
                    "architecture diagram",
                    "workflow",
                    "image",
                    "chart",
                ),
                "evidence": (
                    "figure",
                    "fig.",
                    "caption",
                    "diagram",
                    "overview",
                    "architecture",
                    "workflow",
                    "image",
                    "chart",
                ),
                "triggers": (
                    "figure",
                    "fig.",
                    "image",
                    "diagram",
                    "chart",
                    "visual",
                    "caption",
                    "图",
                    "图片",
                    "图表",
                    "示意图",
                ),
            },
            "table": {
                "query": (
                    "table",
                    "rows",
                    "columns",
                    "metric",
                    "score",
                    "benchmark",
                    "comparison",
                    "ablation",
                ),
                "evidence": (
                    "table",
                    "metric",
                    "score",
                    "benchmark",
                    "comparison",
                    "ablation",
                    "accuracy",
                    "error",
                ),
                "triggers": (
                    "table",
                    "row",
                    "column",
                    "metric",
                    "score",
                    "表",
                    "表格",
                    "指标",
                ),
            },
            "caveat": {
                "query": (
                    "limitation",
                    "limitations",
                    "drawback",
                    "challenge",
                    "risk",
                    "failure case",
                    "future work",
                    "constraint",
                ),
                "evidence": (
                    "limitation",
                    "limitations",
                    "drawback",
                    "challenge",
                    "risk",
                    "failure",
                    "constraint",
                    "however",
                    "future work",
                ),
                "triggers": (
                    "limitation",
                    "drawback",
                    "challenge",
                    "risk",
                    "failure",
                    "future work",
                    "局限",
                    "不足",
                    "限制",
                    "风险",
                    "挑战",
                    "未来",
                ),
            },
            "conclusion": {
                "query": (
                    "conclusion",
                    "discussion",
                    "summary",
                    "future work",
                    "implication",
                ),
                "evidence": (
                    "conclusion",
                    "discussion",
                    "summary",
                    "future work",
                    "implication",
                    "we conclude",
                ),
                "triggers": (
                    "conclusion",
                    "discussion",
                    "future work",
                    "takeaway",
                    "结论",
                    "讨论",
                    "启示",
                    "未来",
                ),
            },
            "reference": {
                "query": ("references", "bibliography", "cited work", "citation"),
                "evidence": ("references", "bibliography", "citation"),
                "triggers": ("reference", "references", "bibliography", "引用", "参考文献"),
            },
            "field": {
                "query": ("title", "authors", "date", "keywords", "metadata"),
                "evidence": ("title", "authors", "date", "keywords"),
                "triggers": ("title", "author", "date", "keyword", "标题", "作者", "日期", "关键词"),
            },
        }

    def _paper_structure_role_alias(self, role: str) -> str:
        normalized = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "result": "claim",
            "results": "claim",
            "finding": "claim",
            "findings": "claim",
            "performance": "claim",
            "metric": "claim",
            "evaluation": "claim",
            "experiment": "claim",
            "method": "approach",
            "methods": "approach",
            "architecture": "approach",
            "requirement": "approach",
            "step": "approach",
            "implementation": "approach",
            "data": "dataset",
            "training_data": "dataset",
            "corpus": "dataset",
            "image": "visual",
            "figure": "visual",
            "fig": "visual",
            "caption": "visual",
            "bibliography": "reference",
            "metadata": "field",
        }
        return aliases.get(normalized, normalized)

    def _paper_structure_role_terms(self, role: str, *, kind: str = "query") -> list[str]:
        canonical = self._paper_structure_role_alias(role)
        definition = self._paper_structure_signal_definitions().get(canonical, {})
        terms = definition.get(kind) or definition.get("query") or ()
        return [term for term in terms if term]

    def _paper_structure_roles_for_question(
        self,
        question: str,
        soft_intent: dict[str, Any] | None = None,
    ) -> list[str]:
        soft_intent = soft_intent or {}
        normalized = " ".join(str(question or "").lower().split())
        roles: list[str] = []

        def add(role: str) -> None:
            canonical = self._paper_structure_role_alias(role)
            if canonical and canonical not in roles and canonical in self._paper_structure_signal_definitions():
                roles.append(canonical)

        for role in soft_intent.get("preferred_roles", []):
            add(str(role))

        intent = str(soft_intent.get("intent") or "")
        scope = str(soft_intent.get("scope") or "")
        if intent in {"document_wide_question", "compare_question"} or scope in {"whole_document", "multi_document"}:
            for role in ["purpose", "approach", "claim", "conclusion"]:
                add(role)

        for role, definition in self._paper_structure_signal_definitions().items():
            if any(trigger.lower() in normalized for trigger in definition.get("triggers", ())):
                add(role)

        for method_name, role in [
            ("_looks_like_broad_overview_question", "purpose"),
            ("_looks_like_metric_result_question", "claim"),
            ("_looks_like_visual_retrieval_question", "visual"),
            ("_looks_like_table_question", "table"),
            ("_looks_like_dataset_or_scale_question", "dataset"),
        ]:
            checker = getattr(self, method_name, None)
            if callable(checker) and checker(question):
                add(role)

        if "claim" in roles and "table" not in roles and any(term in normalized for term in ["table", "表格", "表"]):
            add("table")
        if "visual" in roles and "approach" not in roles and any(term in normalized for term in ["architecture", "workflow", "diagram", "架构", "流程"]):
            add("approach")

        return roles[:5]

    def _paper_structure_section_role_bonus(self, section: str) -> dict[str, float]:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(section or "").lower()).strip()
        if not normalized:
            return {}
        rules: list[tuple[tuple[str, ...], dict[str, float]]] = [
            (("abstract", "摘要"), {"purpose": 1.0, "claim": 0.35}),
            (("introduction", "intro", "引言", "绪论"), {"purpose": 0.45, "claim": 0.15}),
            (("method", "methods", "approach", "model", "architecture", "方法", "模型", "架构"), {"approach": 0.9}),
            (("data", "dataset", "materials", "corpus", "数据", "语料"), {"dataset": 0.85, "approach": 0.15}),
            (("experiment", "experiments", "evaluation", "result", "results", "实验", "结果", "评测"), {"claim": 0.9, "table": 0.2}),
            (("discussion", "讨论"), {"claim": 0.35, "caveat": 0.35, "conclusion": 0.2}),
            (("conclusion", "summary", "结论", "总结"), {"conclusion": 1.0, "claim": 0.35}),
            (("limitation", "limitations", "future", "局限", "限制", "未来"), {"caveat": 1.0, "conclusion": 0.35}),
            (("reference", "references", "bibliography", "参考文献"), {"reference": 1.0}),
        ]
        bonus: dict[str, float] = {}
        for markers, role_bonus in rules:
            if any(marker in normalized for marker in markers):
                for role, value in role_bonus.items():
                    bonus[role] = max(bonus.get(role, 0.0), value)
        return bonus

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
        roles: list[str] = []
        for raw in soft_intent.get("preferred_roles", []):
            role = self._paper_structure_role_alias(str(raw).strip())
            if role and role not in roles and role in self._paper_structure_signal_definitions():
                roles.append(role)
        return roles

    def _score_rows_by_semantic_role(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, list[tuple[float, dict[str, Any]]]]:
        scored: dict[str, list[tuple[float, dict[str, Any]]]] = {
            role: [] for role in self._paper_structure_signal_definitions()
        }
        scored["example"] = []
        scored["informative"] = []
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
            role: list(definition.get("evidence", ()))
            for role, definition in self._paper_structure_signal_definitions().items()
        }
        role_keywords["example"] = [
            "例如",
            "案例",
            "场景",
            "应用",
            "实践",
            "sample",
            "case",
            "scenario",
            "application",
        ]
        section_bonus = self._paper_structure_section_role_bonus(section)
        position_bonus = 0.35 if index <= max(2, int(total * 0.08)) else 0.0
        scores: dict[str, float] = {}
        for role, keywords in role_keywords.items():
            hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
            score = hits * 0.35 + section_bonus.get(role, 0.0)
            if role == "purpose":
                score += position_bonus
            if role in {"conclusion", "caveat"} and index >= int(total * 0.65):
                score += 0.2
            if role in {"claim", "table"} and self._is_table_like_text(text):
                score += 0.25
            if role == "visual" and any(marker in normalized for marker in ["figure", "fig.", "caption", "diagram", "图"]):
                score += 0.2
            if score > 0:
                scores[role] = score
        return scores
