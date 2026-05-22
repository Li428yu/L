from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent_parts.state import (
    DocumentProfile,
    PaperAgentState,
    ParsedTask,
    ReadingTaskContract,
)
from backend.app.models import EvidenceItem, RetrievalDebugItem, RuntimeStep


class AgentPlanningMixin:
    def _plan(self, state: PaperAgentState) -> PaperAgentState:
        self._emit_status("正在理解你的问题...")
        question = state["question"].strip()
        has_documents = bool(self._resolve_document_ids(state.get("document_ids")))
        needs_retrieval = has_documents and not self._looks_like_meta_question(question)
        local_intent = self._classify_question_intent(question)
        local_strategy = self._retrieval_strategy_for_question(question) if needs_retrieval else "no_retrieval"
        soft_intent = self._build_soft_intent_for_question(
            question=question,
            local_intent=local_intent,
            local_strategy=local_strategy,
            needs_retrieval=needs_retrieval,
        )
        intent = str(soft_intent.get("intent") or local_intent)
        retrieval_strategy = self._retrieval_strategy_from_soft_intent(
            soft_intent=soft_intent,
            local_strategy=local_strategy,
            needs_retrieval=needs_retrieval,
        )
        task_parse_reason = ""
        source = "模型软判断" if soft_intent.get("source") == "model" else "本地候选"
        reason = str(soft_intent.get("reason") or "").strip()
        focus = "、".join(str(item) for item in soft_intent.get("focus", [])[:4] if str(item).strip())
        soft_detail = f"{source}：{reason or '按用户原话生成可修正的意图候选。'}"
        if focus:
            soft_detail += f" 关注点：{focus}。"
        detail = (
            f"识别为「{self._friendly_intent(intent)}」，检索策略为「{self._friendly_retrieval_strategy(retrieval_strategy)}」。"
            f"{soft_detail}{task_parse_reason}"
            if needs_retrieval
            else f"识别为「{self._friendly_intent(intent)}」，这轮不需要检索文档。{soft_detail}"
        )
        return {
            **state,
            "intent": intent,
            "retrieval_strategy": retrieval_strategy,
            "soft_intent": soft_intent,
            "compound_tasks": [],
            "task_parse_reason": task_parse_reason,
            "needs_retrieval": needs_retrieval,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(node="planner", title="判断路径", detail=detail),
            ],
        }

    def _route_after_planner(self, state: PaperAgentState) -> str:
        return "retrieve" if state.get("needs_retrieval") else "answer"

    def _build_soft_intent_for_question(
        self,
        *,
        question: str,
        local_intent: str,
        local_strategy: str,
        needs_retrieval: bool,
    ) -> dict[str, Any]:
        fallback = self._local_soft_intent(
            question=question,
            local_intent=local_intent,
            local_strategy=local_strategy,
            needs_retrieval=needs_retrieval,
        )
        if not needs_retrieval:
            return fallback

        system_prompt = (
            "你是文档问答系统里的“意图理解助手”。你的任务不是回答用户，"
            "而是把用户的口语问题理解成可修正的阅读意图，用于后续检索证据。"
            "不要把本地候选当成最终结论；如果用户说“这文章写了啥/讲了什么”，"
            "通常是在问主要内容，而不是作者、单位、日期等首页信息。"
            "只返回 JSON，不要输出解释文字。"
        )
        user_prompt = f"""
用户问题：{question}

本地候选意图：{local_intent}
本地候选检索：{local_strategy}

请返回 JSON，字段如下：
{{
  "intent": "reference_question/field_lookup_question/compare_question/document_wide_question/meta_question/specific_question",
  "operation": "extract/summarize/analyze/compare/judge/answer",
  "scope": "field/section/whole_document/multi_document/specific_point",
  "focus": ["用户真正关心的1-4个内容点"],
  "preferred_roles": ["purpose/approach/claim/result/conclusion/caveat/example/field/reference"],
  "exclude_roles": ["front_matter/metadata/bibliography/submission/code/table"],
  "answer_style": "short/plain/analytical/list",
  "confidence": 0.0,
  "reason": "一句话说明为什么这样理解"
}}

要求：
- JSON 必须能被解析。
- focus、preferred_roles、exclude_roles 都用短词。
- 不确定时保守写成 answer + specific_point，并在 reason 说明。
""".strip()

        try:
            raw = self.model_clients.chat_text(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                model=self.settings.default_chat_model,
            )
            parsed = self._extract_json_object(raw)
            if not parsed:
                return fallback
            return self._normalize_soft_intent(
                parsed,
                fallback=fallback,
                source="model",
            )
        except RuntimeError:
            return fallback

    def _local_soft_intent(
        self,
        *,
        question: str,
        local_intent: str,
        local_strategy: str,
        needs_retrieval: bool,
    ) -> dict[str, Any]:
        contract = self._build_reading_task_contract(question)
        focus = self._overview_focus_keywords(question)[:4] or self._question_keywords(question)[:4]
        preferred_roles: list[str] = []
        exclude_roles = ["front_matter", "metadata"]

        if local_intent == "field_lookup_question":
            preferred_roles = ["field"]
            focus = [self._field_label(target) for target in self._field_lookup_targets(question)] or focus
        elif local_intent == "reference_question":
            preferred_roles = ["reference"]
            exclude_roles = ["front_matter", "metadata", "submission"]
        elif contract.operation == "summarize":
            preferred_roles = list(contract.role_hints)
            exclude_roles = list(contract.exclude_roles)
        else:
            preferred_roles = ["purpose", "approach", "claim", "conclusion"]

        operation = {
            "field_lookup_question": "extract",
            "reference_question": "extract",
            "compare_question": "compare",
            "document_wide_question": "summarize",
        }.get(local_intent, "answer")
        scope = {
            "field_lookup_question": "field",
            "reference_question": "section",
            "compare_question": "multi_document",
            "document_wide_question": "whole_document",
        }.get(local_intent, "specific_point")
        if contract.scope == "whole_document":
            scope = "whole_document"

        return {
            "intent": local_intent,
            "operation": operation,
            "scope": scope,
            "focus": focus,
            "preferred_roles": preferred_roles,
            "exclude_roles": exclude_roles,
            "answer_style": contract.style if contract.style != "plain_colloquial" else "plain",
            "confidence": 0.52 if needs_retrieval else 0.4,
            "reason": "本地只提供候选理解，最终回答前仍交给模型按用户原话修正。",
            "source": "local",
            "local_intent": local_intent,
            "local_strategy": local_strategy,
        }

    def _extract_json_object(self, raw: str) -> dict[str, Any] | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

    def _normalize_soft_intent(
        self,
        parsed: dict[str, Any],
        *,
        fallback: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        allowed_intents = {
            "reference_question",
            "field_lookup_question",
            "compare_question",
            "document_wide_question",
            "meta_question",
            "specific_question",
        }
        allowed_operations = {"extract", "summarize", "analyze", "compare", "judge", "answer"}
        allowed_scopes = {"field", "section", "whole_document", "multi_document", "specific_point"}

        def clean_list(value: Any, fallback_value: list[str], limit: int = 6) -> list[str]:
            if not isinstance(value, list):
                return fallback_value[:limit]
            cleaned: list[str] = []
            for item in value:
                text = str(item).strip()
                if not text or text in cleaned:
                    continue
                cleaned.append(text[:40])
                if len(cleaned) >= limit:
                    break
            return cleaned or fallback_value[:limit]

        intent = str(parsed.get("intent") or "")
        operation = str(parsed.get("operation") or "")
        scope = str(parsed.get("scope") or "")
        confidence = parsed.get("confidence", fallback.get("confidence", 0.5))
        try:
            confidence_value = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            confidence_value = float(fallback.get("confidence", 0.5) or 0.5)

        return {
            **fallback,
            "intent": intent if intent in allowed_intents else fallback.get("intent", "specific_question"),
            "operation": operation if operation in allowed_operations else fallback.get("operation", "answer"),
            "scope": scope if scope in allowed_scopes else fallback.get("scope", "specific_point"),
            "focus": clean_list(parsed.get("focus"), list(fallback.get("focus", [])), limit=4),
            "preferred_roles": clean_list(
                parsed.get("preferred_roles"),
                list(fallback.get("preferred_roles", [])),
                limit=6,
            ),
            "exclude_roles": clean_list(parsed.get("exclude_roles"), list(fallback.get("exclude_roles", [])), limit=6),
            "answer_style": str(parsed.get("answer_style") or fallback.get("answer_style") or "plain")[:20],
            "confidence": confidence_value,
            "reason": str(parsed.get("reason") or fallback.get("reason") or "").strip()[:180],
            "source": source,
        }

    def _retrieval_strategy_from_soft_intent(
        self,
        *,
        soft_intent: dict[str, Any],
        local_strategy: str,
        needs_retrieval: bool,
    ) -> str:
        if not needs_retrieval:
            return "no_retrieval"
        intent = str(soft_intent.get("intent") or "")
        operation = str(soft_intent.get("operation") or "")
        scope = str(soft_intent.get("scope") or "")
        if intent == "reference_question":
            return "hybrid_reference"
        if intent == "field_lookup_question" and operation == "extract":
            return "hybrid_field_lookup"
        if intent == "compare_question" or scope == "multi_document":
            return "hybrid_comparison"
        if operation == "judge":
            return "hybrid_soft"
        if intent == "document_wide_question" or scope == "whole_document":
            return "hybrid_overview"
        return "hybrid_soft" if local_strategy != "no_retrieval" else local_strategy

    def _format_question_understanding_for_model(
        self,
        *,
        question: str,
        retrieval_strategy: str,
        evidence: list[EvidenceItem],
        soft_intent: dict[str, Any] | None = None,
    ) -> str:
        candidates: list[str] = []
        soft_intent = soft_intent or {}
        soft_focus = "、".join(str(item) for item in soft_intent.get("focus", []) if str(item).strip())
        soft_roles = "、".join(str(item) for item in soft_intent.get("preferred_roles", []) if str(item).strip())
        soft_exclusions = "、".join(str(item) for item in soft_intent.get("exclude_roles", []) if str(item).strip())
        if soft_intent:
            source = "模型软判断" if soft_intent.get("source") == "model" else "本地候选"
            candidates.append(
                f"{source}认为本轮更像“{soft_intent.get('operation', 'answer')} / {soft_intent.get('scope', 'specific_point')}”；"
                f"关注点：{soft_focus or '由证据判断'}；优先证据角色：{soft_roles or '不限'}；"
                f"应排除：{soft_exclusions or '明显无关噪声'}。这仍只是候选，请你按用户原话和证据自行修正。"
            )
        field_targets = self._field_lookup_targets(question)
        field_labels = "、".join(self._field_label(target) for target in field_targets)
        analytical_markers = [
            "为什么",
            "原因",
            "意义",
            "作用",
            "影响",
            "评价",
            "判断",
            "合理",
            "可靠",
            "对应",
            "匹配",
            "解释",
            "分析",
            "概括",
            "总结",
        ]

        if field_targets:
            if any(marker in question for marker in analytical_markers):
                candidates.append(
                    f"用户提到了“{field_labels}”，但问法带有解释/分析/评价色彩；可以利用字段证据，但不要机械地只摘字段。"
                )
            else:
                candidates.append(
                    f"用户可能是在索取原文里的“{field_labels}”字段；若证据已给出字段边界，应只回答目标字段，不扩展相邻作者、单位、日期等信息。"
                )

        if self._looks_like_content_overview_request(question):
            candidates.append(
                "用户可能是在用口语问文档主要内容；回答应先抓核心主题和主要信息，不要把首页元信息或提交说明当正文。"
            )

        if self._looks_like_experiment_content_overview_question(question):
            candidates.append(
                "用户可能是在问实验做什么、训练什么能力或实验内容怎么展开；优先使用实验目的、实验内容、实验要求/步骤附近证据，忽略提交说明和选做题，除非用户明确问提交或选做。"
            )

        if self._looks_like_reference_question(question):
            candidates.append("用户可能是在查参考文献或引用来源；此时应优先列出文献条目，不要概括正文。")
        if self._looks_like_compare_question(question):
            candidates.append("用户可能是在比较多个对象；应按对象分别说明，再给出差异或共同点。")

        if not candidates:
            candidates.append("用户可能是在问一个具体内容点；请先判断目标概念或对象，再基于最相关证据直接回答。")

        evidence_note = (
            f"本轮本地检索策略是“{self._friendly_retrieval_strategy(retrieval_strategy)}”，"
            f"交给模型的证据有 {len(evidence)} 条。"
        )
        return "\n".join(
            [
                evidence_note,
                "以下只是候选理解，允许你根据用户原话和证据修正：",
                *[f"- {candidate}" for candidate in candidates],
            ]
        )

    def _classify_question_intent(self, question: str) -> str:
        if self._looks_like_reference_question(question):
            return "reference_question"
        if self._looks_like_field_lookup_question(question):
            return "field_lookup_question"
        if self._looks_like_compare_question(question):
            return "compare_question"
        if self._looks_like_document_wide_question(question):
            return "document_wide_question"
        if self._looks_like_meta_question(question):
            return "meta_question"
        return "specific_question"

    def _parse_compound_tasks(self, question: str) -> list[ParsedTask]:
        return []

    def _task_parse_reason(self, tasks: list[ParsedTask]) -> str:
        return ""

    def _retrieval_strategy_for_question(self, question: str) -> str:
        if self._looks_like_reference_question(question):
            return "reference_section"
        if self._looks_like_field_lookup_question(question):
            return "field_lookup"
        if self._looks_like_compare_question(question):
            return "comparison_overview"
        if (
            self._looks_like_document_wide_question(question)
        ):
            return "document_overview"
        if self._looks_like_meta_question(question):
            return "no_retrieval"
        return "vector_similarity"

    def _friendly_intent(self, intent: str) -> str:
        labels = {
            "reference_question": "参考文献问题",
            "field_lookup_question": "字段提取问题",
            "compare_question": "多文档对比问题",
            "document_wide_question": "整篇概括/分析问题",
            "meta_question": "使用说明问题",
            "specific_question": "具体内容问答",
        }
        return labels.get(intent, intent)

    def _friendly_retrieval_strategy(self, strategy: str) -> str:
        labels = {
            "reference_section": "参考文献区检索",
            "field_lookup": "字段精确提取",
            "comparison_overview": "多文档概览检索",
            "document_overview": "整篇文档重点检索",
            "vector_similarity": "向量相似度检索",
            "hybrid_soft": "Dense + BM25 + RRF",
            "hybrid_reference": "参考文献候选 + 双路召回 + RRF",
            "hybrid_field_lookup": "字段候选 + 双路召回 + RRF",
            "hybrid_comparison": "对比候选 + 双路召回 + RRF",
            "hybrid_overview": "全文概括双路召回 + RRF",
            "hybrid_retry": "扩大范围后的双路召回 + RRF",
            "no_retrieval": "不检索文档",
        }
        return labels.get(strategy, strategy)

    def _looks_like_meta_question(self, question: str) -> bool:
        keywords = ["你是谁", "怎么使用", "有哪些功能", "如何上传", "支持什么模型"]
        return any(keyword in question for keyword in keywords)

    def _looks_like_overview_question(self, question: str) -> bool:
        keywords = ["概括", "总结", "讲了什么", "讲啥", "写了啥", "写的啥", "说了啥", "主要内容", "通俗语言", "大意", "一句话"]
        return any(keyword in question for keyword in keywords) or self._looks_like_content_overview_request(question)

    def _looks_like_content_overview_request(self, question: str) -> bool:
        normalized = " ".join(question.split())
        if not normalized:
            return False
        if self._looks_like_field_lookup_question(normalized):
            return False

        overview_actions = [
            "介绍",
            "说说",
            "讲讲",
            "讲一下",
            "说一下",
            "看一下",
            "简单说",
            "简单讲",
            "大概",
            "概括",
            "总结",
            "讲了什么",
            "写了什么",
            "写了啥",
            "是干嘛",
            "做什么",
            "干什么",
        ]
        content_scopes = [
            "文章",
            "论文",
            "文档",
            "报告",
            "材料",
            "这篇",
            "这份",
            "全文",
            "内容",
            "实验",
            "章节",
            "部分",
        ]
        has_action = any(action in normalized for action in overview_actions)
        has_scope = any(scope in normalized for scope in content_scopes)
        return has_action and has_scope

    def _looks_like_experiment_content_overview_question(self, question: str) -> bool:
        if "实验" not in question:
            return False
        cues = [
            "内容",
            "目的",
            "要求",
            "步骤",
            "过程",
            "任务",
            "介绍",
            "说说",
            "讲讲",
            "概括",
            "总结",
            "做什么",
            "干什么",
            "是干嘛",
        ]
        return any(cue in question for cue in cues) and not self._question_allows_submission_details(question)

    def _looks_like_document_wide_question(self, question: str) -> bool:
        keywords = [
            "概括",
            "总结",
            "讲了什么",
            "讲啥",
            "讲了啥",
            "写了啥",
            "写的啥",
            "说了啥",
            "大概讲",
            "大概说",
            "主要内容",
            "主要写",
            "通俗语言",
            "大意",
            "一句话",
            "发现",
            "重点",
            "核心",
            "结论",
            "贡献",
            "方法",
            "局限",
            "不足",
            "目的",
            "主题",
            "能学到",
            "学到什么",
            "收获",
            "启发",
        ]
        return any(keyword in question for keyword in keywords) or self._looks_like_content_overview_request(question)

    def _looks_like_broad_overview_question(self, question: str) -> bool:
        contract = self._build_reading_task_contract(question)
        return (
            contract.operation == "summarize"
            and contract.scope == "whole_document"
            and contract.target in {"main_content", "experiment_content"}
        )

    def _build_reading_task_contract(self, question: str) -> ReadingTaskContract:
        broad_markers = ["概括", "总结", "讲了什么", "讲什么", "主要内容", "主要讲", "大意", "一句话", "介绍", "做什么", "干什么", "是干嘛"]
        colloquial_markers = ["啥", "咋", "大概", "看一下", "说一下", "讲讲", "写了啥", "讲了啥"]
        style = "plain_colloquial" if any(marker in question for marker in colloquial_markers) else "plain"
        is_content_overview = self._looks_like_content_overview_request(question)
        is_experiment_content = self._looks_like_experiment_content_overview_question(question)

        if not any(marker in question for marker in broad_markers + ["写了啥", "写的啥", "说了啥"]) and not is_content_overview and not is_experiment_content:
            return ReadingTaskContract(
                operation="answer",
                scope="focused",
                depth="normal",
                style=style,
                target="specific_content",
            )
        narrow_markers = [
            "方法",
            "怎么做",
            "如何研究",
            "研究设计",
            "局限",
            "不足",
            "问题",
            "风险",
            "结论",
            "发现",
            "核心",
            "重点",
            "贡献",
            "能学到",
            "学到什么",
            "收获",
            "启发",
            "参考文献",
            "作者",
            "单位",
            "日期",
            "关键词",
            "摘要是什么",
        ]
        if any(marker in question for marker in narrow_markers):
            return ReadingTaskContract(
                operation="answer",
                scope="focused",
                depth="normal",
                style=style,
                target="specific_content",
            )
        depth = "one_sentence" if "一句话" in question else "brief"
        if is_experiment_content:
            return ReadingTaskContract(
                operation="summarize",
                scope="whole_document",
                depth=depth,
                style=style,
                target="experiment_content",
                exclude_roles=("front_matter", "bibliography", "metadata", "submission"),
                role_hints=("purpose", "approach", "requirement", "step", "implementation"),
            )
        return ReadingTaskContract(
            operation="summarize",
            scope="whole_document",
            depth=depth,
            style=style,
            target="main_content",
            exclude_roles=("front_matter", "bibliography", "metadata"),
            role_hints=("purpose", "approach", "claim", "conclusion"),
        )

    def _overview_focus_keywords(self, question: str) -> list[str]:
        keywords: list[str] = []
        if any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"]):
            keywords.extend([
                "研究方法",
                "方法",
                "采用",
                "样本",
                "数据",
                "问卷",
                "访谈",
                "实验",
                "模型",
                "机制建构",
                "method",
                "model",
                "architecture",
                "training",
                "experiment",
                "evaluation",
            ])
        if any(word in question for word in ["结论", "发现", "核心", "重点", "贡献"]):
            keywords.extend([
                "结论",
                "结论与展望",
                "总体而言",
                "研究认为",
                "发现",
                "表明",
                "贡献",
                "提出",
                "未来研究",
                "results",
                "achieves",
                "state-of-the-art",
                "conclusion",
                "we show",
                "we propose",
            ])
        if any(word in question for word in ["局限", "不足"]):
            keywords.extend(["未来研究", "实证数据", "数据来源", "样本", "验证", "检验", "缺少", "缺乏", "尚未", "不足"])
        if any(word in question for word in ["问题", "风险"]):
            keywords.extend(["风险", "挑战", "问题", "隐私", "偏差", "诚信", "责任", "不足"])
        if any(word in question for word in ["目的", "主题", "讲了什么", "主要内容", "概括", "总结", "大意", "一句话"]):
            keywords.extend([
                "摘要",
                "本文围绕",
                "研究目的",
                "旨在",
                "主要讨论",
                "研究认为",
                "结论",
                "abstract",
                "introduction",
                "in this paper",
                "in this work",
                "we propose",
                "we present",
                "we introduce",
                "we show",
                "transformer",
                "attention",
                "sequence transduction",
                "machine translation",
            ])
        if any(word in question for word in ["能学到", "学到什么", "收获", "启发"]):
            keywords.extend(["机制", "应用场景", "学习支持", "风险", "治理", "人机协同", "数据", "隐私", "算法", "价值"])
        if self._looks_like_experiment_content_overview_question(question):
            keywords.extend([
                "实验目的",
                "实验内容",
                "实验任务",
                "实验要求",
                "实验步骤",
                "实验过程",
                "实验原理",
                "操作步骤",
                "掌握",
                "学会",
                "设计",
                "编写",
                "实现",
                "应用",
            ])
        return list(dict.fromkeys(keywords or ["摘要", "本文", "研究", "结论", "方法"]))

    def _needs_opening_context(self, question: str) -> bool:
        return any(word in question for word in ["概括", "总结", "讲了什么", "讲啥", "讲了啥", "写了啥", "写的啥", "说了啥", "主要内容", "大意", "一句话", "目的", "主题", "能学到", "学到什么", "收获", "启发"])

    def _looks_like_compare_question(self, question: str) -> bool:
        keywords = ["对比", "比较", "不同点", "差异", "区别", "有什么不同", "哪里不同"]
        return any(keyword in question for keyword in keywords)

    def _looks_like_reference_question(self, question: str) -> bool:
        normalized = question.lower().strip()
        keywords = [
            "参考文献",
            "引用文献",
            "引用了哪些",
            "参考了哪些",
            "用了哪些文献",
            "用了哪些参考",
            "文献列表",
            "参考列表",
            "references",
            "bibliography",
        ]
        if any(keyword in normalized for keyword in keywords):
            return True
        return "文献" in question and any(
            verb in question
            for verb in ["哪些", "列出", "引用", "参考", "出处", "来源"]
        )

    def _looks_like_field_lookup_question(self, question: str) -> bool:
        targets = self._field_lookup_targets(question)
        if not targets:
            return False

        broad_markers = [
            "分析",
            "解释",
            "意思",
            "含义",
            "评价",
            "判断",
            "为什么",
            "有必要",
            "必要吗",
            "需要吗",
            "该不该",
            "干扰",
            "怎么",
            "如何",
            "可靠",
            "局限",
            "不足",
            "贡献",
            "方法",
            "结论",
            "风险",
            "治理",
            "支撑",
            "匹配",
            "相符",
            "跑题",
            "概括",
            "总结",
            "摘要一下",
        ]
        if any(marker in question for marker in broad_markers):
            return False

        lookup_markers = [
            "是什么",
            "有哪些",
            "哪些",
            "列出",
            "提取",
            "写出",
            "给我",
            "告诉我",
            "是多少",
            "是谁",
            "字段",
            "部分",
            "内容",
            "keyword",
        ]
        normalized = question.lower().strip()
        return any(marker in normalized for marker in lookup_markers) or len(normalized) <= 28

    def _field_lookup_targets(self, question: str) -> list[str]:
        normalized = question.lower()
        aliases = [
            ("keywords", ["关键词", "关键字", "keywords", "keyword", "key words"]),
            ("abstract", ["摘要", "abstract"]),
            ("authors", ["作者"]),
            ("affiliation", ["作者单位", "单位", "机构"]),
            ("date", ["完成日期", "日期", "时间"]),
            ("title", ["标题", "题目"]),
        ]
        targets: list[str] = []
        for field, field_aliases in aliases:
            if any(alias in normalized for alias in field_aliases):
                targets.append(field)
        return targets

    def _field_lookup_debug_keywords(self, question: str) -> list[str]:
        keywords: list[str] = []
        for target in self._field_lookup_targets(question):
            keywords.extend([self._field_label(target), *self._field_aliases(target)])
        return list(dict.fromkeys(keywords))

    def _field_aliases(self, field: str) -> list[str]:
        return {
            "keywords": ["关键词", "关键字", "Keywords", "Key words"],
            "abstract": ["摘要", "摘 要", "Abstract"],
            "authors": ["作者"],
            "affiliation": ["作者单位", "单位", "机构"],
            "date": ["日期", "完成日期", "时间"],
            "title": ["标题", "题目"],
        }.get(field, [])

    def _field_label(self, field: str) -> str:
        return {
            "keywords": "关键词",
            "abstract": "摘要",
            "authors": "作者",
            "affiliation": "单位",
            "date": "日期",
            "title": "标题",
        }.get(field, field)

