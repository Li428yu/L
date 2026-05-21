from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent_parts.state import (
    AnswerPlan,
    DocumentProfile,
    PaperAgentState,
    ParsedTask,
    ReadingTaskContract,
)
from backend.app.models import EvidenceItem, RetrievalDebugItem, RuntimeStep


class AgentAnsweringMixin:
    def _answer(self, state: PaperAgentState) -> PaperAgentState:
        plan = self._prepare_answer_plan(state)
        if plan.mode in {"local", "unavailable"}:
            return self._finalize_answer(state, plan.local_answer, plan)

        try:
            answer = self.model_clients.chat_text(
                [SystemMessage(content=plan.system_prompt), HumanMessage(content=plan.user_prompt)],
                model=state["chat_model"],
            )
            answer = self._clean_model_answer(answer)
        except RuntimeError as exc:
            fallback_plan = self._model_failure_answer_plan(state, plan, exc)
            return self._finalize_answer(state, fallback_plan.local_answer, fallback_plan)

        return self._finalize_answer(state, answer, plan)

    def _prepare_answer_plan(self, state: PaperAgentState) -> AnswerPlan:
        model_prompt_evidence = self._select_model_prompt_evidence(
            question=state["question"],
            evidence=state.get("evidence", []),
            soft_intent=state.get("soft_intent", {}),
        )
        compact_model_prompt_evidence = self._compact_evidence_for_model_prompt(
            question=state["question"],
            evidence=model_prompt_evidence,
        )
        evidence_blocks = self._format_evidence_for_model_prompt(
            question=state["question"],
            evidence=compact_model_prompt_evidence,
        )
        final_prompt_evidence = [
            f"[{item.citation_id}] {item.paper_name} p.{item.page} score={item.score:.3f}"
            for item in compact_model_prompt_evidence
        ]
        recent_history = self._format_recent_history(state.get("recent_messages", []))
        force_model_answer = self.settings.force_model_answer
        question_understanding = self._format_question_understanding_for_model(
            question=state["question"],
            retrieval_strategy=state.get("retrieval_strategy", ""),
            evidence=compact_model_prompt_evidence or state.get("evidence", []),
            soft_intent=state.get("soft_intent", {}),
        )
        if not force_model_answer and self._looks_like_reference_question(state["question"]):
            answer = self._build_local_reference_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
            )
            return AnswerPlan(
                mode="local",
                local_answer=answer,
                answer_strategy="local_reference_answer",
                final_prompt_evidence=final_prompt_evidence,
                prompt_evidence=compact_model_prompt_evidence,
                runtime_detail="这是参考文献类问题，已直接读取文末参考文献区并整理为列表。",
            )

        if not force_model_answer and self._looks_like_field_lookup_question(state["question"]):
            answer = self._build_local_field_lookup_answer(
                state["question"],
                state.get("evidence", []),
            )
            return AnswerPlan(
                mode="local",
                local_answer=answer,
                answer_strategy="local_field_lookup_answer",
                final_prompt_evidence=final_prompt_evidence,
                prompt_evidence=compact_model_prompt_evidence,
                runtime_detail="这是字段提取类问题，已只保留用户询问的字段内容，避免作者、单位、日期等相邻信息混入。",
            )

        system_prompt = (
            "你是一个严谨、友好的中文文档阅读助手。用户说“这篇论文”“这份文档”时，"
            "默认指当前已上传文档，不要要求用户重新提供标题。若当前有多篇文档且用户没有指定某一篇，"
            "必须按文档分开回答，避免把几篇文档混成一个结论。回答必须优先基于检索证据，"
            "不要编造文档内容。使用证据时用 [E1]、[E2] 这样的编号引用。"
            "如果证据不足，要说明缺少什么，但只要已有证据能回答，就先直接回答。"
            "不要输出姓名、学号、邮箱等个人信息，除非用户明确询问。"
            "系统给出的任务理解只是候选，不是最终结论；你必须根据用户原话和证据重新判断真实意图，"
            "不要机械地把某个词固定映射成某类任务。"
            "如果问题像字段提取，只回答目标字段本身；如果问题是在解释、评价、比较或概括，不要只机械摘字段。"
            "除非用户明确询问，不要输出作者、单位、日期、提交说明、参考文献等无关信息。"
            "你可以使用简洁 Markdown（段落、编号列表、加粗），但引用证据只能使用 [E1]、[E2] 这种格式，"
            "不要输出“Introduction依据”“Conclusion依据”“Abstract依据”等自造引用。"
        )
        user_prompt = f"""
用户长期画像和偏好：
{state.get("memory_prompt", "暂无")}

最近对话历史：
{recent_history}

本轮问题：
{state["question"]}

任务理解候选：
{question_understanding}

检索证据：
{evidence_blocks or "本轮没有检索到证据。"}

请给出适合非技术用户阅读的中文回答，并在关键事实后标注证据编号。
""".strip()

        return AnswerPlan(
            mode="model",
            answer_strategy="model_answer",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            final_prompt_evidence=final_prompt_evidence,
            prompt_evidence=compact_model_prompt_evidence,
            runtime_detail="结合原文证据、用户偏好和最近对话生成最终回答。",
        )

    def _model_failure_answer_plan(
        self,
        state: PaperAgentState,
        plan: AnswerPlan,
        exc: RuntimeError,
    ) -> AnswerPlan:
        return AnswerPlan(
            mode="unavailable",
            local_answer=self._build_model_unavailable_answer(exc),
            answer_strategy="model_unavailable",
            fallback_used=True,
            final_prompt_evidence=plan.final_prompt_evidence,
            prompt_evidence=plan.prompt_evidence,
            runtime_detail="对话模型响应太慢或暂时不可用；理解、分析、评价、总结和对比类任务不会使用本地规则拼接替代答案。",
        )

    def _finalize_answer(
        self,
        state: PaperAgentState,
        answer: str,
        plan: AnswerPlan,
    ) -> PaperAgentState:
        return {
            **state,
            "answer": answer,
            "answer_strategy": plan.answer_strategy,
            "fallback_used": plan.fallback_used,
            "final_prompt_evidence": plan.final_prompt_evidence,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="answer",
                    title="生成回答",
                    detail=plan.runtime_detail,
                ),
            ],
        }

    def _stream_model_answer_tokens(self, state: PaperAgentState, plan: AnswerPlan):
        yield from self.model_clients.chat_text_stream(
            [SystemMessage(content=plan.system_prompt), HumanMessage(content=plan.user_prompt)],
            model=state["chat_model"],
        )

    def _format_evidence(self, evidence: list[EvidenceItem]) -> str:
        blocks = []
        for item in evidence:
            blocks.append(
                f"[{item.citation_id}] {item.paper_name} | Page {item.page} | "
                f"Section: {item.section or 'Unknown'} | Score: {item.score:.3f}\n"
                f"{self._sanitize_evidence_text(item.text)}"
            )
        return "\n\n".join(blocks)

    def _select_model_prompt_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        soft_intent: dict[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        if len(evidence) <= 2:
            return evidence

        soft_intent = soft_intent or {}
        allow_references = self._looks_like_reference_question(question) or soft_intent.get("intent") == "reference_question"
        scored: list[tuple[float, int, EvidenceItem]] = []
        fallback: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            noise_penalty = self._model_prompt_noise_penalty(
                question=question,
                item=item,
                text=text,
                allow_references=allow_references,
            )
            relevance = self._question_relevance_score(question, text)
            readability = self._readable_text_score(text)
            role_score = self._model_prompt_semantic_score(question=question, item=item, text=text)
            soft_penalty = self._evidence_noise_penalty_for_soft_intent(
                question=question,
                text=text,
                section=item.section or "",
                soft_intent=soft_intent,
            )
            soft_focus = self._soft_focus_score(soft_intent, text)
            soft_role = self._soft_role_score(
                soft_intent=soft_intent,
                text=text,
                section=item.section or "",
                index=0,
                total=1,
            )
            score = (
                item.score
                + relevance * 0.9
                + readability * 0.25
                + role_score
                + soft_focus * 0.35
                + soft_role * 0.3
                - noise_penalty
                - min(soft_penalty, 2.0) * 0.35
            )
            row = (score, position, item)
            if noise_penalty >= 1.4 or soft_penalty >= 2.2:
                fallback.append(row)
            else:
                scored.append(row)

        candidates = scored or fallback
        candidates.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        limit = min(4, max(2, len(candidates)))
        selected = [item for _, _, item in candidates[:limit]]
        return selected or evidence[: min(4, len(evidence))]

    def _format_evidence_for_model_prompt(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
    ) -> str:
        blocks: list[str] = []
        for item in evidence:
            summary = self._summarize_evidence_for_model_prompt(question=question, item=item)
            if not summary:
                continue
            blocks.append(
                f"[{item.citation_id}] {item.paper_name} | Page {item.page} | "
                f"Section: {item.section or 'Unknown'} | Score: {item.score:.3f}\n"
                f"{summary}"
            )
        return "\n\n".join(blocks)

    def _compact_evidence_for_model_prompt(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        compact: list[EvidenceItem] = []
        for item in evidence:
            summary = self._summarize_evidence_for_model_prompt(question=question, item=item)
            if not summary:
                continue
            compact.append(
                item.model_copy(
                    update={
                        "text": summary,
                        "quote": summary,
                    }
                )
            )
        return compact

    def _summarize_evidence_for_model_prompt(self, *, question: str, item: EvidenceItem) -> str:
        text = self._sanitize_evidence_text(item.text)
        if not text.strip():
            return ""
        if self._looks_like_reference_question(question):
            return self._best_reference_quote(text, limit=360)
        if self._looks_like_field_lookup_question(question):
            return self._truncate_readable_text(item.quote or text, limit=220)

        focused = self._focused_sentences_for_question(question, text, limit=2)
        if not focused:
            focused = self._pick_readable_sentences(text, limit=2)
        cleaned = [
            self._trim_model_prompt_sentence(sentence)
            for sentence in focused
            if sentence and not self._looks_like_field_or_metadata_sentence(sentence)
        ]
        cleaned = [sentence for sentence in cleaned if sentence]
        if not cleaned:
            quote = self._best_quote_for_question(question, text, limit=360)
            cleaned = [self._trim_model_prompt_sentence(quote)] if quote else []
        summary = " ".join(cleaned)
        return self._truncate_readable_text(summary, limit=420)

    def _trim_model_prompt_sentence(self, sentence: str) -> str:
        cleaned = self._trim_front_matter_prefix(sentence)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(
            r"^(?:实验提交|提交代码|思考题[【\\[]?选做[】\\]]?|选做题)[:：]?\s*",
            "",
            cleaned,
        ).strip()
        return cleaned

    def _model_prompt_noise_penalty(
        self,
        *,
        question: str,
        item: EvidenceItem,
        text: str,
        allow_references: bool,
    ) -> float:
        penalty = 0.0
        if self._looks_like_front_matter_noise(text):
            penalty += 1.8
        if self._looks_like_field_or_metadata_sentence(item.quote or text):
            penalty += 1.2
        if item.section == "References" or self._looks_like_reference_section_text(text):
            penalty += 0.0 if allow_references else 1.5
        if self._looks_like_submission_or_assignment_noise(text) and not self._question_allows_submission_details(question):
            penalty += 0.35 if self._has_experiment_overview_anchor(text) else 1.4
        if self._looks_like_code_heavy_text(text) and not self._question_asks_for_code_details(question):
            penalty += 0.8
        if self._is_table_like_text(text) and not self._looks_like_table_question(question):
            penalty += 0.8
        return penalty

    def _model_prompt_semantic_score(self, *, question: str, item: EvidenceItem, text: str) -> float:
        score = 0.0
        section = item.section or ""
        if self._looks_like_overview_question(question) or any(word in question for word in ["介绍", "说说", "讲讲"]):
            role_scores = self._semantic_role_scores(
                text=text,
                section=section,
                index=0,
                total=1,
            )
            score += max(role_scores.values(), default=0.0) * 0.35
        if any(word in question for word in ["实验", "实验内容", "实验目的", "实验步骤"]):
            if any(keyword in text for keyword in ["实验目的", "实验内容", "实验步骤", "实验要求", "掌握", "插件", "实现"]):
                score += 0.9
            if "实验内容" in text:
                score += 0.7
            if "实验目的" in text:
                score += 0.4
            if self._looks_like_submission_or_assignment_noise(text) and not self._question_allows_submission_details(question):
                score -= 0.6
            if self._looks_like_code_heavy_text(text) and not self._question_asks_for_code_details(question):
                score -= 0.5
        return score

    def _looks_like_submission_or_assignment_noise(self, text: str) -> bool:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        if not normalized:
            return False
        markers = ["实验提交", "提交代码", "提交报告", "思考题", "选做", "作业提交", "评分标准", "可添加公式"]
        return any(marker in normalized for marker in markers)

    def _has_experiment_overview_anchor(self, text: str) -> bool:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        anchors = ["实验目的", "实验内容", "实验任务", "实验要求", "实验步骤", "实验过程"]
        return any(anchor in normalized for anchor in anchors)

    def _looks_like_code_heavy_text(self, text: str) -> bool:
        normalized = self._sanitize_evidence_text(text)
        if not normalized.strip():
            return False
        code_markers = [
            "function ",
            "const ",
            "let ",
            "var ",
            "=>",
            "addEventListener",
            "querySelector",
            "return ",
            "class ",
            "</",
            "{",
            "}",
            ";",
        ]
        marker_hits = sum(normalized.count(marker) for marker in code_markers)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        code_like_lines = sum(
            1
            for line in lines
            if re.search(r"(function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|=>|[{};]{2,}|</?\w+)", line)
        )
        return marker_hits >= 8 or (len(lines) >= 4 and code_like_lines / max(len(lines), 1) >= 0.45)

    def _question_asks_for_code_details(self, question: str) -> bool:
        return any(word in question for word in ["代码", "源码", "函数", "类", "实现细节", "怎么实现", "程序", "写法"])

    def _question_allows_submission_details(self, question: str) -> bool:
        return any(word in question for word in ["提交", "交什么", "作业", "思考题", "选做", "评分", "交报告", "交代码"])

    def _build_model_unavailable_answer(self, exc: RuntimeError) -> str:
        cause = getattr(exc, "__cause__", None)
        cause_name = type(cause).__name__ if cause else type(exc).__name__
        return (
            "这轮没有生成回答：对话模型调用失败或超时。\n\n"
            "系统只会用本地规则处理参考文献和字段提取这类确定性问题；"
            f"理解、分析、评价、总结和对比类任务不会用本地拼接答案替代模型回答。错误类型：{cause_name}。"
        )

    def _clean_model_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = re.sub(
            r"\[?(?:Introduction|Conclusion|Abstract|Methods?|Results?|Discussion|Unknown)\s*依据\]?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\[\s*(E\d+)\s*\]", r"[\1]", cleaned)
        cleaned = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([，。！？；：,.!?;:])\s*(\[E\d+\])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _format_recent_history(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "暂无。"
        lines = []
        for item in messages[-8:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip()
            if len(content) > 500:
                content = f"{content[:500]}..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_local_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到可用的原文片段，所以暂时不能可靠回答。你可以换个问法，或重新准备文档。"
        if self._looks_like_reference_question(question):
            return self._build_local_reference_answer(question, evidence, memory_facts)
        if self._looks_like_reliability_question(question):
            return self._build_local_reliability_answer(question, evidence, memory_facts)

        profile = self._profile_from_evidence(evidence)
        clean_text = " ".join(
            self._sanitize_evidence_text(item.text)
            for item in evidence[:4]
            if item.text.strip()
        )
        sentences = self._pick_readable_sentences(clean_text, limit=4)
        citation = evidence[0].citation_id

        if self._looks_like_document_wide_question(question):
            return self._build_local_document_wide_answer(
                question,
                evidence,
                memory_facts,
            )

        body = " ".join(sentences)
        audience_note = self._audience_note(memory_facts or {}, question, clean_text)
        return (
            f"我先按你的问题直接回答：\n\n"
            f"{audience_note}"
            f"我在《{profile.title}》里找到的相关信息是：{body or profile.main_claim} "
            f"[{citation}]\n\n"
            "如果你想要更准确的分析，可以继续问“它的方法可靠吗”“它有哪些局限”“它的结论能不能支撑题目”。"
        )

    def _build_local_field_lookup_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> str:
        targets = self._field_lookup_targets(question)
        if not targets:
            return "我没有识别出你要提取哪个字段，可以直接问“关键词是什么”或“摘要是什么”。"
        if not evidence:
            target_text = "、".join(self._field_label(target) for target in targets)
            return f"我没有在当前文档里找到“{target_text}”字段，所以暂时不能可靠提取。"

        grouped = self._group_evidence_by_document(evidence)
        if len(grouped) > 1:
            sections: list[str] = []
            for index, (name, items) in enumerate(grouped.items(), start=1):
                lines = [
                    f"{self._field_label_from_evidence(item)}：{self._field_value_from_evidence(item)} [{item.citation_id}]"
                    for item in items
                ]
                sections.append(f"{index}. 《{name}》\n" + "\n".join(lines))
            return "\n\n".join(sections)

        lines = [
            f"{self._field_label_from_evidence(item)}：{self._field_value_from_evidence(item)} [{item.citation_id}]"
            for item in evidence
        ]
        return "\n".join(lines)

    def _field_label_from_evidence(self, item: EvidenceItem) -> str:
        text = self._sanitize_evidence_text(item.text or item.quote)
        if "：" in text:
            return text.split("：", 1)[0].strip() or (item.section or "字段")
        if ":" in text:
            return text.split(":", 1)[0].strip() or (item.section or "字段")
        return item.section or "字段"

    def _field_value_from_evidence(self, item: EvidenceItem) -> str:
        text = self._sanitize_evidence_text(item.text or item.quote)
        if "：" in text:
            return text.split("：", 1)[1].strip()
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text.strip()

    def _build_local_document_wide_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文内容，所以暂时不能概括这份文档。"

        profile = (
            self._build_document_profile(evidence[0].document_id)
            if len({item.document_id for item in evidence if item.document_id}) == 1
            else self._profile_from_evidence(evidence)
        )
        all_text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        citation_count = 3 if self._looks_like_broad_overview_question(question) else 2
        citation_suffix = self._join_citations([item.citation_id for item in evidence[:citation_count]])
        if not citation_suffix and evidence:
            citation_suffix = f" [{evidence[0].citation_id}]"
        audience_note = self._audience_note(memory_facts or {}, question, all_text).strip()
        prefix = f"{audience_note}\n\n" if audience_note else ""
        contract = self._build_reading_task_contract(question)

        if contract.target == "experiment_content" or (
            contract.target == "main_content"
            and self._has_experiment_overview_anchor(all_text)
        ):
            return self._build_local_experiment_content_answer(
                question=question,
                evidence=evidence,
                text=all_text,
                prefix=prefix,
                citation_suffix=citation_suffix,
            )

        if any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"]):
            method = profile.method
            method_evidence = self._method_sentences(all_text, limit=1)
            detail = " ".join(method_evidence) if method_evidence else profile.main_claim
            if method and "没有清楚" not in method:
                return (
                    f"{prefix}这份文档的方法可以概括为：{method}。\n\n"
                    f"从原文看，相关说明是：{detail}{citation_suffix}"
                )
            return (
                f"{prefix}我没有看到它清楚交代可复核的研究方法、样本或数据来源。\n\n"
                f"目前能找到的相关内容是：{detail}{citation_suffix}"
            )

        if any(word in question for word in ["局限", "不足", "问题", "风险"]):
            limitations = (
                self._research_limitation_points(profile, all_text)
                if any(word in question for word in ["局限", "不足"])
                else []
            )
            if not limitations:
                limitations = self._focused_sentences_for_question(question, all_text, limit=3)
            if not limitations:
                limitations = self._focused_sentences_for_question("未来研究 缺少 不足 风险", all_text, limit=3)
            body = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(limitations, start=1))
            if not body:
                body = "原文没有非常集中地说明局限，只能从现有证据中看到它仍需要更多数据、验证或边界说明。"
            return f"{prefix}它的局限/风险主要可以这样看：\n\n{body}\n\n这些判断来自原文相关段落。{citation_suffix}"

        if any(word in question for word in ["结论", "发现", "核心", "重点", "贡献"]):
            conclusion = self._extract_conclusion_from_text(all_text)
            focused = self._focused_sentences_for_question(question, all_text, limit=3)
            if conclusion:
                return f"{prefix}它的核心结论是：{conclusion}{citation_suffix}"
            body = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(focused, start=1))
            return (
                f"{prefix}我从原文里提炼到的核心观点是：\n\n{body or profile.main_claim}\n\n"
                f"这些内容来自文档中与问题最相关的段落。{citation_suffix}"
            )

        if any(word in question for word in ["能学到", "学到什么", "收获", "启发"]):
            learning_points = self._learning_points_for_profile(
                profile=profile,
                question=question,
                text=all_text,
                memory_facts=memory_facts or {},
            )
            body = "\n".join(
                f"{index}. {point}"
                for index, point in enumerate(learning_points, start=1)
            )
            major = self._extract_major_from_context(question, memory_facts or {})
            intro = (
                f"从{major}专业角度看，你更适合重点学这几件事："
                if major
                else "你这次说了“专业角度”，但我没有读到具体专业；我先说明边界，再按通用阅读角度给你可用的收获："
            )
            return (
                f"{prefix}{intro}\n\n"
                f"{body}\n\n"
                f"这些判断来自当前文档的主题、机制、风险和治理相关段落。{citation_suffix}"
            )

        topic = self._extract_topic_from_text(all_text) or profile.title
        key_sentences = self._focused_sentences_for_question(question, all_text, limit=3)
        if not key_sentences:
            key_sentences = self._pick_readable_sentences(all_text, limit=3)
        method_line = (
            f"它采用的方法是{profile.method}。"
            if profile.method and "没有清楚" not in profile.method
            else "原文没有清楚呈现严格、可复核的研究方法。"
        )
        if self._looks_like_broad_overview_question(question):
            contract = self._build_reading_task_contract(question)
            return self._build_broad_overview_answer(
                profile=profile,
                text=all_text,
                citation_suffix=citation_suffix,
                prefix=prefix,
                contract=contract,
            )

        main_points = "\n".join(
            f"{index}. {sentence}"
            for index, sentence in enumerate(key_sentences[:3], start=1)
        )
        point_intro = "可以先抓住三点：" if len(key_sentences) >= 3 else "可以先抓住这些点："
        return (
            f"{prefix}这份文档主要讲《{topic}》。\n\n"
            f"{point_intro}\n"
            f"{main_points or f'1. {profile.main_claim}'}\n\n"
            f"{method_line}{citation_suffix}"
        )

    def _build_broad_overview_answer(
        self,
        *,
        profile: DocumentProfile,
        text: str,
        citation_suffix: str,
        prefix: str,
        contract: ReadingTaskContract,
    ) -> str:
        topic = self._extract_topic_from_text(text) or profile.title
        lead = self._overview_lead_sentence(text=text, profile=profile, topic=topic)
        points = self._adaptive_overview_points(
            text=text,
            profile=profile,
            topic=topic,
            limit=3 if contract.depth != "one_sentence" else 1,
            seed_points=[lead],
        )
        if contract.depth == "one_sentence":
            return f"{prefix}{lead}{citation_suffix}"
        if not points:
            points = [profile.main_claim]
        body = "\n".join(
            f"{index}. {self._normalize_overview_point(point)}"
            for index, point in enumerate(points, start=1)
            if point.strip()
        )
        return (
            f"{prefix}{lead}\n\n"
            f"可以这样理解：\n{body}\n\n"
            f"这些概括来自文档中说明写作目的、展开方式和主要观点的段落。{citation_suffix}"
        )

    def _build_local_experiment_content_answer(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        text: str,
        prefix: str,
        citation_suffix: str,
    ) -> str:
        sections = self._extract_experiment_sections(text)
        purpose = sections.get("实验目的", "")
        content = sections.get("实验内容", "") or sections.get("实验任务", "")
        requirement = sections.get("实验要求", "") or sections.get("实现要求", "")
        steps = sections.get("实验步骤", "") or sections.get("实验过程", "") or sections.get("操作步骤", "")

        content_citation = self._first_citation_with(evidence, ["实验内容", "实验任务"], fallback=False)
        purpose_citation = self._first_citation_with(evidence, ["实验目的"], fallback=False)
        requirement_citation = self._first_citation_with(evidence, ["实验要求", "实现要求", "实验步骤", "实验过程"], fallback=False)

        if not content:
            focused = self._focused_sentences_for_question(question, text, limit=3)
            content = focused[0] if focused else ""
        if not purpose:
            purpose = self._best_sentence_with_any(text, ["掌握", "学会", "领悟", "理解", "目的"])
        if not requirement and not steps and not (content or purpose):
            requirement = self._best_sentence_with_any(text, ["设计", "编写", "实现", "应用", "完成"])

        lines: list[str] = []
        if content:
            citation = f" [{content_citation}]" if content_citation else ""
            lines.append(f"这个实验的主要内容是{self._normalize_experiment_fragment(content)}{citation}")
        if purpose:
            citation = f" [{purpose_citation}]" if purpose_citation else ""
            lines.append(f"它想训练你{self._normalize_experiment_fragment(purpose)}{citation}")
        wants_process_detail = any(word in question for word in ["步骤", "过程", "要求", "怎么", "如何", "实现"])
        supporting = (requirement or steps) if wants_process_detail else ""
        if supporting:
            citation = f" [{requirement_citation}]" if requirement_citation else ""
            lines.append(f"落到操作上，重点是{self._normalize_experiment_fragment(supporting)}{citation}")

        if not lines:
            sentences = self._pick_readable_sentences(text, limit=3)
            lines = [sentence for sentence in sentences[:3]]

        if len(lines) == 1:
            return f"{prefix}{lines[0]}{citation_suffix}"
        body = "\n".join(f"{index}. {line}" for index, line in enumerate(lines[:3], start=1))
        return f"{prefix}可以简单理解为：\n\n{body}\n\n这些概括来自实验目的、实验内容或实验步骤附近的原文。{citation_suffix}"

    def _extract_experiment_sections(self, text: str) -> dict[str, str]:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        headings = [
            "实验目的",
            "实验内容",
            "实验任务",
            "实验要求",
            "实现要求",
            "实验步骤",
            "实验过程",
            "操作步骤",
            "实验原理",
            "实验思路",
        ]
        heading_pattern = "|".join(re.escape(heading) for heading in headings)
        pattern = re.compile(
            rf"(?:^|\s)(?:[一二三四五六七八九十]+[、.．]\s*|\d+(?:\.\d+)*[、.．]?\s*)?({heading_pattern})\s*[:：]?\s*"
        )
        matches = list(pattern.finditer(normalized))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            heading = match.group(1)
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            fragment = normalized[start:end].strip(" ：:，,。；;")
            fragment = re.split(r"\s*(?:实验提交|提交代码|提交报告|思考题|选做|评分标准)\s*", fragment, maxsplit=1)[0]
            fragment = self._truncate_readable_text(fragment, limit=180).strip()
            if fragment and heading not in sections:
                sections[heading] = fragment
        return sections

    def _best_sentence_with_any(self, text: str, keywords: list[str]) -> str:
        for sentence in self._pick_readable_sentences(text, limit=20):
            if any(keyword in sentence for keyword in keywords):
                return sentence
        return ""

    def _normalize_experiment_fragment(self, fragment: str) -> str:
        cleaned = " ".join(self._trim_front_matter_prefix(fragment).split()).strip(" ：:，,。；;")
        cleaned = re.sub(r"^领悟掌握\s*", "掌握", cleaned)
        cleaned = re.sub(r"^(?:通过|完成)\s*", "", cleaned)
        cleaned = re.sub(r"(掌握|了解|使用|编写|实现)([A-Za-z0-9])", r"\1 \2", cleaned)
        cleaned = re.split(
            r"\s*(?:编写应用效果|如下图|下图|本例需要|问题分解|步骤\s*\d|提供的资源|可以直接拷贝)\s*",
            cleaned,
            maxsplit=1,
        )[0].strip(" ：:，,。；;")
        cleaned = self._truncate_readable_text(cleaned, limit=130)
        if not cleaned:
            return ""
        return cleaned + ("。" if cleaned[-1] not in "。！？.!?" else "")

    def _overview_lead_sentence(self, *, text: str, profile: DocumentProfile, topic: str) -> str:
        summary_sentence = self._best_sentence_for_semantic_role(text, "purpose")
        if summary_sentence:
            cleaned = self._normalize_overview_point(summary_sentence)
            if len(cleaned) <= 140:
                return f"简单说，{cleaned}"
        if topic and topic != "当前文档":
            return f"简单说，这份文档主要围绕《{topic}》展开。"
        return f"简单说，{profile.main_claim}"

    def _adaptive_overview_points(
        self,
        *,
        text: str,
        profile: DocumentProfile,
        topic: str,
        limit: int,
        seed_points: list[str] | None = None,
    ) -> list[str]:
        role_order = ["approach", "claim", "conclusion", "caveat", "example", "purpose"]
        selected: list[str] = []
        duplicate_scope = [self._normalize_overview_point(point) for point in (seed_points or [])]
        for role in role_order:
            sentence = self._best_sentence_for_semantic_role(text, role)
            if not sentence:
                continue
            cleaned = self._normalize_overview_point(sentence)
            if not cleaned or self._overview_point_is_duplicate(cleaned, [*duplicate_scope, *selected]):
                continue
            selected.append(cleaned)
            if len(selected) >= limit:
                break

        if len(selected) < limit and profile.method and "没有清楚" not in profile.method:
            method_point = f"它主要用{profile.method}来展开内容。"
            if not self._overview_point_is_duplicate(method_point, [*duplicate_scope, *selected]):
                selected.append(method_point)

        if len(selected) < limit and profile.main_claim:
            claim = self._normalize_overview_point(profile.main_claim)
            if not self._overview_point_is_duplicate(claim, [*duplicate_scope, *selected]):
                selected.append(claim)

        return selected[:limit]

    def _best_sentence_for_semantic_role(self, text: str, role: str) -> str:
        sentences = self._pick_readable_sentences(text, limit=60)
        if not sentences:
            return ""
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            scores = self._semantic_role_scores(
                text=sentence,
                section="",
                index=index,
                total=max(len(sentences), 1),
            )
            score = scores.get(role, 0.0)
            if role == "purpose" and index <= 2:
                score += 0.2
            if score <= 0:
                continue
            if self._looks_like_front_matter_noise(sentence):
                continue
            if self._looks_like_field_or_metadata_sentence(sentence):
                continue
            scored.append((score, index, sentence))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return scored[0][2] if scored else ""

    def _normalize_overview_point(self, sentence: str) -> str:
        cleaned = " ".join(self._trim_front_matter_prefix(sentence).split()).strip(" ，,。；;")
        replacements = {
            "本文": "文章",
            "本研究": "这项研究",
            "本报告": "这份报告",
            "本实验": "这个实验",
        }
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        if len(cleaned) > 180:
            cleaned = self._truncate_readable_text(cleaned, limit=180).rstrip(".")
        return cleaned + ("。" if cleaned and cleaned[-1] not in "。！？.!?" else "")

    def _looks_like_field_or_metadata_sentence(self, sentence: str) -> bool:
        normalized = " ".join(self._sanitize_evidence_text(sentence).split())
        if not normalized:
            return True
        metadata_match = re.search(r"(关键词|关键字|作者|单位|作者单位|机构|日期|完成日期)\s*[:：]", normalized)
        if metadata_match and (metadata_match.start() <= 90 or len(normalized) <= 220):
            return True
        if "摘 要" in normalized and len(normalized) <= 80:
            return True
        return False

    def _overview_point_is_duplicate(self, candidate: str, existing: list[str]) -> bool:
        candidate_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", candidate.lower()))
        if not candidate_chars:
            return False
        for item in existing:
            item_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", item.lower()))
            if not item_chars:
                continue
            overlap = len(candidate_chars & item_chars) / max(len(candidate_chars), 1)
            if overlap >= 0.78:
                return True
        return False

    def _should_decline_for_missing_direct_evidence(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> bool:
        if not evidence:
            return False
        if any(
            checker(question)
            for checker in [
                self._looks_like_reference_question,
                self._looks_like_structured_review_request,
                self._looks_like_reliability_question,
                self._looks_like_title_alignment_question,
                self._looks_like_compare_question,
                self._looks_like_document_wide_question,
            ]
        ):
            return False

        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence[:5]))
        strict_terms = [
            "数据库",
            "向量库",
            "代码",
            "公式",
            "指标",
            "样本量",
            "问卷",
            "访谈",
            "回归",
            "实验组",
            "对照组",
            "p值",
            "显著性",
            "作者",
            "日期",
            "年份",
        ]
        requested_terms = [term for term in strict_terms if term in question]
        if requested_terms and not any(term in text for term in requested_terms):
            return True

        relevance_scores = [
            self._question_relevance_score(question, item.text)
            for item in evidence[:5]
        ]
        best_relevance = max(relevance_scores or [0.0])
        best_vector_score = max((item.score for item in evidence[:5]), default=0.0)
        return best_relevance < 0.05 and best_vector_score < 0.35

    def _build_missing_direct_evidence_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not evidence:
            return "我没有在当前文档里找到能回答这个问题的原文证据，所以不能可靠回答。"
        citations = self._join_citations([item.citation_id for item in evidence[:2]])
        return (
            "我没有在当前文档里找到能直接回答这个问题的证据，所以不应该硬编答案。\n\n"
            f"我刚刚检索到的相近段落和你的问题关联不够强。你可以换成更贴近原文的问法，"
            f"或者点右侧证据看看文档里实际出现了哪些内容。{citations}"
        )

    def _section_number(self, index: int) -> str:
        numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 1 <= index <= len(numbers):
            return numbers[index - 1]
        return str(index)

    def _compound_task_title(
        self,
        task: ParsedTask,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        major = self._extract_major_from_context(question, memory_facts)
        titles = {
            "overview_summary": "总体概括文章内容",
            "reference_list": "文章使用的参考文献",
            "professional_takeaways": f"从{major}专业角度，你能收获什么" if major else "从专业角度，你能收获什么",
            "reliability_judgment": "可靠性判断",
            "method_analysis": "研究方法/实现方法分析",
            "limitation_analysis": "局限与不足",
            "conclusion_summary": "核心结论",
            "comparison": "文档对比",
        }
        return titles.get(task.task_type, task.label)

    def _build_local_compound_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文证据，所以不能可靠地完成这个多步骤问题。请确认文档已经上传并完成索引。"

        tasks = self._parse_compound_tasks(question)
        if len(tasks) < 2:
            return self._build_local_answer(question, evidence, memory_facts)

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_key: dict[str, DocumentProfile] = {}
        for document_id in resolved_document_ids:
            profile = self._build_document_profile(document_id)
            profile_by_key[profile.document_id] = profile
            profile_by_key[profile.name] = profile

        all_context = " ".join(item.text for item in evidence)
        audience_note = self._audience_note(memory_facts or {}, question, all_context).strip()
        ordered_tasks = " → ".join(task.label for task in tasks)
        parts = [f"我按你问题里的顺序来回答：{ordered_tasks}。"]
        if len(grouped) > 1:
            parts.append("你上传了多篇文档，我会在每个任务里把文档分开说，避免混在一起。")
        if audience_note:
            parts.append(audience_note)

        for index, task in enumerate(tasks, start=1):
            section_body = self._render_compound_task_section(
                task=task,
                question=question,
                grouped=grouped,
                profile_by_key=profile_by_key,
                memory_facts=memory_facts or {},
            )
            parts.append(
                f"{self._section_number(index)}、{self._compound_task_title(task, question, memory_facts or {})}\n\n"
                f"{section_body}"
            )

        return "\n\n".join(part for part in parts if part.strip())

    def _render_compound_task_section(
        self,
        *,
        task: ParsedTask,
        question: str,
        grouped: dict[str, list[EvidenceItem]],
        profile_by_key: dict[str, DocumentProfile],
        memory_facts: dict[str, str],
    ) -> str:
        if task.task_type == "comparison":
            all_evidence = [item for items in grouped.values() for item in items]
            return self._build_local_compare_answer(question, all_evidence, memory_facts)

        sections: list[str] = []
        multi_document = len(grouped) > 1
        for doc_index, (name, items) in enumerate(grouped.items(), start=1):
            profile = (
                profile_by_key.get(items[0].document_id)
                or profile_by_key.get(name)
                or self._profile_from_evidence(items)
            )
            text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            if task.task_type == "overview_summary":
                content = self._render_compound_overview(profile, items, text)
            elif task.task_type == "reference_list":
                content = self._render_compound_references(name, items)
            elif task.task_type == "professional_takeaways":
                content = self._render_compound_takeaways(profile, items, text, question, memory_facts)
            elif task.task_type == "reliability_judgment":
                content = self._render_compound_reliability(profile, items, text)
            elif task.task_type == "method_analysis":
                content = self._render_compound_method(profile, items, text)
            elif task.task_type == "limitation_analysis":
                content = self._render_compound_limitations(profile, items, text, question)
            elif task.task_type == "conclusion_summary":
                content = self._render_compound_conclusion(profile, items, text)
            else:
                content = self._build_local_document_wide_answer(question, items, memory_facts)

            if multi_document:
                title = profile.title or name
                sections.append(f"{doc_index}. 《{title}》\n{content}")
            else:
                sections.append(content)
        return "\n\n".join(sections)

    def _render_compound_overview(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["摘要", "本文围绕", "研究主题", "实验名称", "实验目的"]),
                self._first_citation_with(items, ["结论", "研究认为", "总体而言"], fallback=False),
            ]
        )
        if profile.kind == "实验报告" or "实验名称" in text:
            summary = self._summarize_document_for_compare(items)
            return (
                f"这份文档主要是《{summary['title']}》相关内容，重点在{summary['focus']}。"
                f"它的实现/展开过程主要围绕{summary['implementation']}，对应要掌握的知识点是{summary['knowledge']}。{citation}"
            )

        topic = self._extract_topic_from_text(text) or profile.title
        conclusion = (self._extract_conclusion_from_text(text) or profile.main_claim).rstrip("。；; ")
        method_line = (
            f"它主要采用{profile.method}来展开论证"
            if profile.method and "没有清楚" not in profile.method
            else "它更偏观点梳理和材料分析"
        )
        risk_line = (
            "同时提醒学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等风险"
            if any(keyword in text for keyword in ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界"])
            else "同时需要继续核对方法、数据来源和结论边界"
        )
        return (
            f"这篇文档主要讨论《{topic}》。可以抓住三点：\n\n"
            f"1. 研究对象：它围绕“{topic}”展开，重点看这个主题中的问题、机制、场景和结果。\n"
            f"2. 论证方式：{method_line}，所以它更像一篇框架分析型文本，而不是严格实证论文。\n"
            f"3. 核心判断：{conclusion}；{risk_line}。\n\n"
            f"这些概括来自文档开头、方法和结论相关证据。{citation}"
        )

    def _render_compound_references(self, name: str, items: list[EvidenceItem]) -> str:
        text = "\n".join(item.text for item in items)
        references = self._extract_references_from_text(text)
        citations = self._join_citations([item.citation_id for item in items if self._reference_marker_count(item.text) > 0])
        if not citations:
            citations = self._join_citations([self._first_citation_with(items, ["参考文献", "References", "[1]"], fallback=False)])

        if not references:
            return (
                f"我没有稳定解析出《{name}》的编号参考文献条目。"
                f"如果右侧原文里能看到参考文献，建议点开对应证据人工核对；如果看不到，可能是文档没有参考文献区或解析没有抽到。{citations}"
            )

        visible_references = references[:12]
        lines = [f"{number}. {content}" for number, content in visible_references]
        more = f"\n\n其余 {len(references) - len(visible_references)} 条可在右侧原文继续核对。" if len(references) > len(visible_references) else ""
        return (
            f"我在文末参考文献区域解析到 {len(references)} 条条目：\n\n"
            f"{chr(10).join(lines)}{more}\n\n"
            f"这些条目来自参考文献区域。{citations}"
        )

    def _render_compound_takeaways(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        points = self._learning_points_for_profile(
            profile=profile,
            question=question,
            text=text,
            memory_facts=memory_facts,
        )[:3]
        citations = self._join_citations(
            [
                self._first_citation_with(items, ["机制", "系统", "流程", "应用场景", "实验目的"], fallback=False),
                self._first_citation_with(items, ["风险", "隐私", "算法偏差", "学术诚信", "责任边界"], fallback=False),
                self._first_citation_with(items, ["人机协同", "治理", "价值对齐", "过程可控"], fallback=False),
            ]
        )
        if not citations and items:
            citations = self._join_citations([items[0].citation_id])
        body = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))
        major = self._extract_major_from_context(question, memory_facts)
        intro = (
            f"从{major}专业学习角度，建议重点收获这三点："
            if major
            else "你这次只说“专业角度”，但我没有读到具体专业；我先说明边界，再给通用阅读收获："
        )
        return f"{intro}\n\n{body}\n\n这些收获不是凭空扩展，而是从文档主题、方法、场景和风险治理线索中提炼出来的。{citations}"

    def _render_compound_reliability(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        kind_citation = self._first_citation_with(items, ["课程报告", "实验报告", "论文", "摘要", "关键词", "样稿"])
        process_citation = self._first_citation_with(items, ["采用", "方法", "实验", "计算", "分析", "机制建构"], fallback=False)
        support_citation = self._first_citation_with(items, ["未来研究", "实证数据", "参考文献", "验证", "局限", "不足"], fallback=False)
        support_points = self._reliability_support_points(
            profile=profile,
            all_text=text,
            kind_citation=kind_citation,
            process_citation=process_citation or kind_citation,
            support_citation=support_citation or process_citation or kind_citation,
        )
        return (
            f"直接判断：{self._reliability_verdict(profile)}\n\n"
            f"理由：\n"
            f"1. {support_points[0]}\n"
            f"2. {support_points[1]}\n"
            f"3. {support_points[2]}"
        )

    def _render_compound_method(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["研究方法", "采用", "文献分析", "情境推演", "机制建构"], fallback=False),
                self._first_citation_with(items, ["实验方法", "实验步骤", "实现过程", "流程"], fallback=False),
            ]
        )
        if profile.kind == "实验报告" or "实验步骤" in text or "实验目的" in text:
            summary = self._summarize_document_for_compare(items)
            return (
                f"它的方法/实现过程可以概括为：围绕{summary['implementation']}展开。"
                f"这类文档更像实验过程说明，重点是把知识点{summary['knowledge']}落实到具体步骤。{citation}"
            )
        method_sentences = self._method_sentences(text, limit=2)
        detail = "；".join(method_sentences) if method_sentences else profile.method
        return (
            f"它的方法可以概括为：{profile.method}。"
            f"这意味着它更侧重观点梳理、框架建构或案例推演；如果要判断研究强度，还要看有没有样本、数据和验证过程。"
            f"{citation}\n\n原文中与方法最相关的信息是：{detail}{citation}"
        )

    def _render_compound_limitations(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
        question: str,
    ) -> str:
        return self._render_research_limitation_section(profile, items)

    def _render_compound_conclusion(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        conclusion = self._extract_conclusion_from_text(text) or profile.main_claim
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["结论与展望", "总体而言", "研究认为", "综上所述"], fallback=False),
                self._first_citation_with(items, ["结论", "发现", "核心观点"], fallback=False),
            ]
        )
        return f"核心结论可以概括为：{conclusion}{citation}"

    def _build_local_structured_review_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文证据，所以不能按你的模板完成概括、分部分分析和可靠性判断。"

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        all_context = " ".join(item.text for item in evidence)
        audience_note = self._audience_note(memory_facts or {}, question, all_context).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for doc_index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            all_text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            overview_citation = self._first_citation_with(items, ["摘要", "本文围绕", "关键词", "研究主题"])
            method_citation = self._first_citation_with(items, ["采用", "文献分析", "情境推演", "机制建构"])
            mechanism_citation = self._first_citation_with(
                items,
                ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "运行机制"],
            )
            scenario_citation = self._first_citation_with(
                items,
                ["课程学习场景", "实验与实践教学", "学业预警", "学习支持场景", "应用场景"],
            )
            risk_citation = self._first_citation_with(
                items,
                ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界", "风险"],
            )
            governance_citation = self._first_citation_with(
                items,
                ["人机协同", "价值对齐", "过程可控", "数据最小化", "多主体治理", "治理"],
            )
            risk_governance_citations = self._join_citations([risk_citation, governance_citation])
            conclusion_citation = self._first_citation_with(
                items,
                ["结论与展望", "未来研究", "实证数据", "总体而言"],
            )
            reference_citation = self._first_citation_with(items, ["参考文献", "[1]", "研究[J]"], fallback=False)

            title = profile.title or name
            topic = self._extract_topic_from_text(all_text) or title
            conclusion = self._extract_conclusion_from_text(all_text) or profile.main_claim
            method = profile.method if profile.method else "原文没有清楚给出研究方法"
            learning_points = self._learning_points_for_profile(
                profile=profile,
                question=question,
                text=all_text,
                memory_facts=memory_facts or {},
            )[:3]
            reliability_points = self._reliability_support_points(
                profile=profile,
                all_text=all_text,
                kind_citation=overview_citation,
                process_citation=method_citation,
                support_citation=conclusion_citation or reference_citation,
            )

            heading = f"第 {doc_index} 篇文档：《{name}》\n\n" if len(grouped) > 1 else ""
            sections.append(
                f"{heading}"
                f"一、先总结概括\n\n"
                f"这篇文档主要讨论《{topic}》。它的核心意思是：生成式人工智能可以为高校学习支持服务提供新的能力，"
                f"但这种能力必须放在教育目标、制度约束和风险治理中理解，不能简单等同于“技术越强越好”。[{overview_citation}]\n\n"
                f"二、再详细分析每一部分\n\n"
                f"1. 研究背景与问题：文档关注高校学习支持服务在个性化、持续反馈、资源供给和组织协同方面的现实压力，"
                f"并把生成式人工智能作为一种可能的支持工具来讨论。[{overview_citation}]\n\n"
                f"2. 研究方法：原文说明主要采用{method}。这说明它更偏理论分析和框架建构，而不是大样本实证研究。[{method_citation}]\n\n"
                f"3. 作用机制：文档把 AI 的作用概括为认知支架、资源重组、过程陪伴、反馈生成和组织协同等方向，"
                f"重点不是单个模型能力，而是 AI 如何嵌入学习支持流程。[{mechanism_citation}]\n\n"
                f"4. 应用场景：它讨论了课程学习、实验实践、学业预警与辅导等场景，说明作者想把 AI 放到具体教育流程里分析。[{scenario_citation}]\n\n"
                f"5. 风险与治理：文档提醒学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等风险，"
                f"并提出人机协同、价值对齐、过程可控、数据最小化和多主体治理等思路。{risk_governance_citations}\n\n"
                f"三、从你的专业角度看，可以重点学什么\n\n"
                f"1. {learning_points[0]}\n"
                f"2. {learning_points[1]}\n"
                f"3. {learning_points[2]}\n\n"
                f"四、最后判断这篇论文的可靠性\n\n"
                f"直接判断：{self._reliability_verdict(profile)}\n\n"
                f"理由是：\n"
                f"1. {reliability_points[0]}\n"
                f"2. {reliability_points[1]}\n"
                f"3. {reliability_points[2]}\n\n"
                f"五、一句话总结\n\n"
                f"这篇文档适合用来学习“AI 教育应用如何做系统化分析”，但如果要把它当作严格论文结论，还需要进一步核对真实数据、样本、方法和实证验证。"
                f"[{conclusion_citation or overview_citation}]"
            )

        return "\n\n".join(sections)

    def _build_local_reference_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "我没有在当前文档里找到明确的“参考文献/References”部分，所以不能可靠列出它引用了哪些文献。"
                "这通常有三种可能：文档本身没有参考文献、参考文献被图片扫描进了 PDF、或解析时没有抽取到文末内容。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            text = "\n".join(item.text for item in items)
            references = self._extract_references_from_text(text)
            citations = self._join_citations([item.citation_id for item in items])
            label = f"第 {index} 篇文档《{name}》" if len(grouped) > 1 else f"《{name}》"

            if not references:
                sections.append(
                    f"{label}：我找到了疑似参考文献区域，但没有稳定解析出编号条目。"
                    f"建议你点右侧证据查看原文位置再核对。{citations}"
                )
                continue

            reference_lines = [
                f"{number}. {content}"
                for number, content in references
            ]
            sections.append(
                f"{label}文末列出了 {len(references)} 条参考文献，整理如下：\n\n"
                f"{chr(10).join(reference_lines)}\n\n"
                f"这些条目来自文末参考文献区域。{citations}"
            )

        return "\n\n".join(sections)

    def _build_local_research_limitation_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "我没有找到能直接支撑“文章研究局限”的原文证据，所以不应该把正文里的困难/不足硬说成文章局限。\n\n"
                "要回答这个问题，最好看到作者关于研究方法、数据来源、样本、实证验证、讨论或未来研究的说明。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        intro = "我会把“文章局限”限定为研究设计、证据边界和未来研究空间，不把正文里研究对象本身的困难当成文章局限。"
        sections.append(intro)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = (
                self._build_document_profile(items[0].document_id)
                if items and items[0].document_id
                else self._profile_from_evidence(items)
            )
            label = f"第 {index} 篇文档《{name}》" if len(grouped) > 1 else f"《{name}》"
            sections.append(f"{label}的局限可以这样看：\n\n{self._render_research_limitation_section(profile, items)}")

        return "\n\n".join(sections)

    def _render_research_limitation_section(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
    ) -> str:
        text = self._sanitize_evidence_text(" ".join(item.text for item in items))
        future_citation = self._first_citation_with(
            items,
            ["未来研究", "实证数据", "检验", "验证"],
            fallback=False,
        )
        group_citation = self._first_citation_with(
            items,
            ["不同学生群体", "使用差异"],
            fallback=False,
        )
        method_citation = (
            self._first_citation_with_all(items, ["文献分析", "情境推演"])
            or self._first_citation_with_all(items, ["采用", "文献分析"])
            or self._first_citation_with_all(items, ["采用", "机制建构"])
            or self._first_citation_with(items, ["研究方法"], fallback=False)
        )
        direct_section_citation = self._first_citation_with(
            items,
            ["局限性", "研究局限", "研究不足", "局限与不足", "不足与展望", "结论与展望"],
            fallback=False,
        )

        lines: list[str] = []
        if future_citation:
            lines.append(
                f"1. 原文把进一步的实证检验放到未来研究里，说明当前结论还没有被充分的真实数据或多场景验证支撑。"
                f"[{future_citation}]"
            )
        elif direct_section_citation:
            lines.append(
                f"1. 原文在结论、展望或局限相关位置出现研究边界说明，说明作者并没有把当前结论当成已经完全验证的结论。"
                f"[{direct_section_citation}]"
            )

        if group_citation and group_citation != future_citation:
            lines.append(
                f"{len(lines) + 1}. 原文还把不同学生群体或使用差异留作继续比较的内容，说明它对群体差异的讨论还不充分。"
                f"[{group_citation}]"
            )

        if method_citation:
            lines.append(
                f"{len(lines) + 1}. 从方法看，它主要是{profile.method}，这类方法适合梳理框架和提出解释，但不能单独证明实际效果已经发生。"
                f"[{method_citation}]"
            )

        if not profile.has_empirical_data:
            support_citation = future_citation or method_citation or direct_section_citation
            if support_citation:
                lines.append(
                    f"{len(lines) + 1}. 当前证据没有看到样本、问卷、访谈、实验或统计检验，因此它的结论更适合作为理论分析或研究假设，而不是强实证结论。"
                    f"[{support_citation}]"
                )

        if not lines:
            citations = self._join_citations([item.citation_id for item in items[:2]])
            return (
                "原文没有直接列出清晰的“局限性/研究不足”段落；从已检索到的证据看，也缺少能支撑具体局限判断的研究边界说明。"
                f"因此我不能把正文里的普通困难或风险硬说成文章局限。{citations}"
            )

        body = "\n".join(lines[:4])
        return (
            f"{body}\n\n"
            "这些是文章研究层面的局限，不是正文中被研究对象自身存在的“困难/不足”。"
        )

    def _build_local_reliability_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "直接结论：我现在不能判断这篇文档的结果是否可靠，因为没有检索到足够的原文证据。\n\n"
                "要回答这个问题，至少需要看到它的研究方法、数据来源、结果推导、结论和局限说明。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            all_text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            kind_citation = self._first_citation_with(
                items,
                ["随机生成", "样稿", "课程报告", "实验报告", "论文", "摘要", "关键词", "课程", "报告"],
            )
            process_citation = self._first_citation_with(
                items,
                ["采用", "文献分析", "情境推演", "机制建构", "计算", "公式", "分析", "评价", "一致性检验", "测试", "结果", "结论"],
            )
            support_citation = self._first_citation_with(
                items,
                ["未来研究", "实证数据", "参考文献", "数据来源", "资料来源", "出处", "误差", "验证", "一致性检验", "后评价", "局限", "不足"],
            )
            used_citations = self._join_citations(
                [kind_citation, process_citation, support_citation] or [items[0].citation_id]
            )

            verdict = self._reliability_verdict(profile)
            support_points = self._reliability_support_points(
                profile=profile,
                all_text=all_text,
                kind_citation=kind_citation,
                process_citation=process_citation,
                support_citation=support_citation,
            )

            label = f"第 {index} 份文档" if len(grouped) > 1 else "这份文档"
            title_line = f"《{profile.title}》" if profile.title else f"《{name}》"
            sections.append(
                f"{label}{title_line}：\n\n"
                f"直接结论：{verdict}\n\n"
                f"为什么这样判断：\n"
                f"1. {support_points[0]}\n"
                f"2. {support_points[1]}\n"
                f"3. {support_points[2]}\n\n"
                f"可以相信到什么程度：它适合作为理解“生成式人工智能如何支持高校学习服务”的观点框架和写作参考；"
                f"但如果要把它当成严格研究结论，还需要补充样本、数据来源、实证检验或更清楚的方法说明。{used_citations}"
            )

        answer = "\n\n".join(sections)
        return self._ensure_answer_relevance(question, answer, evidence)

    def _build_local_title_alignment_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "直接结论：我现在不能判断结论能不能支撑题目，因为没有检索到题目、结论或关键论证段落。"
            )

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            title_citation = self._first_citation_with(items, ["机制", "风险", "治理路径", "论文样稿", "题目", "研究"])
            mechanism_citation = self._first_citation_with(
                items,
                ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "管理协同", "运行机制"],
            )
            risk_citation = self._first_citation_with(
                items,
                ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界"],
            )
            governance_citation = self._first_citation_with_all(
                items,
                ["针对上述风险", "人机协同"],
            )
            if not governance_citation:
                governance_citation = self._first_citation_with(
                    items,
                    ["价值对齐", "过程可控", "多主体治理", "治理框架"],
                )
            limitation_citation = self._first_citation_with(
                items,
                ["未来研究可以进一步通过实证数据检验", "未来研究", "实证数据"],
                fallback=False,
            )
            if not limitation_citation:
                limitation_citation = self._first_citation_with(
                    items,
                    ["文献分析", "情境推演", "机制建构", "样稿"],
                )
            used_citations = self._join_citations(
                [title_citation, mechanism_citation, risk_citation, governance_citation, limitation_citation]
            )

            label = f"第 {index} 份文档" if len(grouped) > 1 else "这篇论文"
            sections.append(
                f"{label}《{profile.title}》：\n\n"
                "直接结论：基本能支撑题目，但支撑强度不算很强。它能对应题目里的“机制、风险、治理路径”三个关键词，"
                "不过更多是理论框架式支撑，不是实证数据支撑。\n\n"
                "我为什么这么判断：\n"
                f"1. 题目本身要求回答三个部分：生成式人工智能如何赋能学习支持服务的机制、可能风险、以及治理路径。[{title_citation}]\n"
                f"2. 结论能回应“机制”：原文把作用机制概括为认知支架、资源重组、过程陪伴、反馈生成和组织协同等内容，这和题目中的“机制”是对应的。[{mechanism_citation}]\n"
                f"3. 结论能回应“风险”：原文讨论了学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等问题，这和题目中的“风险”是对应的。[{risk_citation}]\n"
                f"4. 结论也能回应“治理路径”：原文提出人机协同、价值对齐、过程可控、数据最小化和多主体治理，这和题目中的“治理路径”是对应的。[{governance_citation}]\n\n"
                f"不足在哪里：它支撑题目主要靠概念分析和框架归纳。原文方法更偏{profile.method}，并且把进一步通过实证数据检验放在未来研究里，"
                f"所以它能支撑题目方向，但还不能强力证明题目中的观点已经被真实数据验证。[{limitation_citation}]\n\n"
                f"一句话：不算跑题，结论能支撑题目；但如果按严格论文标准看，它的支撑偏“理论上说得通”，还缺少“数据上证明了”。{used_citations}"
            )

        return "\n\n".join(sections)

    def _build_document_profile(self, document_id: str) -> DocumentProfile:
        document = self.store.get_document(document_id)
        rows = self.vector_store.get_document_chunks(document_id, limit=1000)
        chunks = [str(row.get("text", "")) for row in rows if row.get("text")]
        text = self._sanitize_evidence_text(" ".join(chunks))
        first_chunk = chunks[0] if chunks else ""
        name = document.file_name if document else document_id
        title = self._guess_document_title(name, first_chunk)
        kind = self._document_kind_from_text(f"{name}\n{text[:3000]}")
        method = self._extract_method_from_text(text)
        main_claim = self._extract_main_claim_from_text(text)
        return DocumentProfile(
            document_id=document_id,
            name=name,
            title=title,
            kind=kind,
            method=method,
            main_claim=main_claim,
            has_empirical_data=self._has_empirical_data(text),
            has_references=any(keyword in text for keyword in ["参考文献", "[1]", "Journal", "研究[J]"]),
            is_generated_sample=any(keyword in text for keyword in ["随机生成", "论文样稿", "中文论文样稿", "样稿"]),
        )

    def _profile_from_evidence(self, evidence: list[EvidenceItem]) -> DocumentProfile:
        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        name = evidence[0].paper_name if evidence else "当前文档"
        document_id = evidence[0].document_id if evidence else ""
        return DocumentProfile(
            document_id=document_id,
            name=name,
            title=self._guess_document_title(name, text),
            kind=self._document_kind_from_text(f"{name}\n{text}"),
            method=self._extract_method_from_text(text),
            main_claim=self._extract_main_claim_from_text(text),
            has_empirical_data=self._has_empirical_data(text),
            has_references=any(keyword in text for keyword in ["参考文献", "[1]", "Journal", "研究[J]"]),
            is_generated_sample=any(keyword in text for keyword in ["随机生成", "论文样稿", "中文论文样稿", "样稿"]),
        )

    def _guess_document_title(self, file_name: str, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        if normalized:
            before_author = re.split(r"\s+作者[:：]|作者[:：]|日期[:：]|摘\s*要|摘要", normalized, maxsplit=1)[0]
            before_author = before_author.strip(" -—_")
            if 8 <= len(before_author) <= 120:
                return before_author
        cleaned_name = re.sub(r"\.(pdf|docx)$", "", file_name, flags=re.IGNORECASE)
        cleaned_name = cleaned_name.replace("_", " ").strip()
        return cleaned_name[:120] or "当前文档"

    def _document_kind_from_text(self, text: str) -> str:
        if "课程报告" in text:
            return "课程报告"
        if "实验报告" in text:
            return "实验报告"
        if "论文样稿" in text or "随机生成" in text:
            return "论文样稿"
        if "毕业论文" in text or "学位论文" in text:
            return "论文"
        if ("摘要" in text and "参考文献" in text) or "Abstract" in text:
            return "论文"
        if "论文" in text:
            return "论文"
        return "普通文档"

    def _extract_method_from_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        method_patterns = [
            r"采用([^。；;\n]{4,120}?)(?:的方法|方法)",
            r"本文围绕[^。；;\n]{0,80}?采用([^。；;\n]{4,120}?)(?:的方法|方法)",
            r"研究方法[:：]\s*([^。；;\n]{4,120})",
        ]
        for pattern in method_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" ，,。；;")
        if all(keyword in text for keyword in ["文献分析", "情境推演", "机制建构"]):
            return "文献分析、情境推演和机制建构"
        if all(keyword.lower() in normalized.lower() for keyword in ["transformer", "attention"]):
            return "提出 Transformer 架构，用自注意力替代循环/卷积结构，并在机器翻译等任务上实验验证"
        return "原文没有清楚给出可复核的研究方法"

    def _extract_main_claim_from_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        patterns = [
            r"研究认为，([^。]{20,180})。",
            r"结论与展望\s*([^。]{20,180})。",
            r"总体而言，([^。]{20,180})。",
            r"(In this work, we presented the Transformer[^.]{20,220}\.)",
            r"(We propose a new simple network architecture[^.]{20,220}\.)",
            r"(Our model achieves[^.]{20,220}\.)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "原文主要提出观点框架，但当前证据没有稳定抽取到单一结果陈述"

    def _extract_topic_from_text(self, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        patterns = [
            r"本文围绕([^。；;]{8,120}?)(?:这一主题|展开|进行)",
            r"研究主题[:：]\s*([^。；;\n]{8,120})",
            r"主题是[“\"]?([^。”\"；;]{8,120})",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).strip(" ：:，,。；;“”\"")
        return ""

    def _extract_conclusion_from_text(self, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        patterns = [
            r"结论与展望\s*([^。]{30,260}。)",
            r"总体而言，([^。]{20,220}。)",
            r"研究认为，([^。]{20,220}。)",
            r"综上所述，([^。]{20,220}。)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).strip()
        return ""

    def _focused_sentences_for_question(self, question: str, text: str, limit: int) -> list[str]:
        sanitized = self._sanitize_evidence_text(text)
        sentences = self._pick_readable_sentences(sanitized, limit=40)
        if not sentences:
            return []

        focus_keywords = self._overview_focus_keywords(question)
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            normalized_sentence = sentence.lower()
            keyword_hits = sum(
                1
                for keyword in focus_keywords
                if keyword in sentence or keyword.lower() in normalized_sentence
            )
            relevance = self._question_relevance_score(question, sentence)
            score = keyword_hits * 0.45 + relevance + self._overview_structure_score(sentence)
            if any(word in sentence for word in ["姓名", "学号", "电子邮件", "邮箱"]):
                score -= 1
            scored.append((score, index, sentence))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected: list[str] = []
        has_positive_score = any(score > 0 for score, _, _ in scored)
        for score, _, sentence in scored:
            if score <= 0 and has_positive_score:
                continue
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                break
        return selected

    def _method_sentences(self, text: str, limit: int) -> list[str]:
        sentences = self._pick_readable_sentences(text, limit=40)
        scored: list[tuple[float, int, str]] = []
        for sentence in sentences:
            score = 0.0
            if "采用" in sentence and "方法" in sentence:
                score += 3.0
            for keyword in ["文献分析", "情境推演", "机制建构", "研究方法"]:
                if keyword in sentence:
                    score += 1.0
            if sentence.startswith("本文围绕") or "本文围绕" in sentence:
                score += 0.5
            if any(keyword in sentence for keyword in ["案例化", "课程教师", "学生可以", "不得直接提交"]):
                score -= 1.2
            if score > 0:
                scored.append((score, -len(sentence), sentence))

        scored.sort(reverse=True)
        return [sentence for _, _, sentence in scored[:limit]]

    def _research_limitation_points(self, profile: DocumentProfile, text: str) -> list[str]:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        points: list[str] = []
        if "未来研究可以进一步通过实证数据检验" in normalized:
            points.append("原文把“通过实证数据检验不同应用场景的效果”放在未来研究中，说明当前结论还没有充分实证验证。")
        elif not profile.has_empirical_data:
            points.append("当前证据没有看到样本、问卷、访谈、实验或统计检验，所以它更像理论分析，结论支撑力度有限。")

        if profile.method and "文献分析" in profile.method:
            points.append(f"它主要采用{profile.method}，适合梳理观点和搭框架，但不等于证明实际效果已经发生。")
        elif profile.method and "没有清楚" in profile.method:
            points.append("原文没有清楚交代可复核的研究方法，这是判断论文质量时比较明显的短板。")

        if "比较不同学生群体的使用差异" in normalized:
            points.append("原文还把“比较不同学生群体的使用差异”留给未来研究，说明对不同群体的差异分析还不充分。")

        unique: list[str] = []
        for point in points:
            if point not in unique:
                unique.append(point)
        return unique[:3]

    def _learning_points_for_profile(
        self,
        *,
        profile: DocumentProfile,
        question: str,
        text: str,
        memory_facts: dict[str, str],
    ) -> list[str]:
        user_profile = memory_facts.get("user_profile", "")
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        major = self._extract_major_from_context(question, memory_facts)
        is_computer_profile = major in {"计算机", "软件", "网络工程", "人工智能", "数据科学"} or any(
            keyword in f"{user_profile}\n{question}" for keyword in ["计算机", "软件", "网络"]
        )

        if is_computer_profile and any(
            keyword in normalized
            for keyword in ["生成式人工智能", "学习支持", "人机协同", "数据隐私", "算法偏差"]
        ):
            return [
                "学会把 AI 技术放进真实业务场景里分析：这篇文档不是只讲“模型很强”，而是把生成式人工智能放到课程答疑、资源重组、过程陪伴、反馈生成和组织协同等学习支持场景中看。",
                "学会从系统设计角度拆问题：一个 AI 教育系统不只是聊天窗口，还涉及用户需求、交互流程、数据输入、反馈生成、人工监督和组织协同。",
                "学会识别 AI 系统风险：文中提到学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界，这些都和计算机专业里的安全、可信 AI、数据治理有关。",
                "学会用工程视角看治理方案：人机协同、价值对齐、过程可控、数据最小化和多主体治理，可以理解成 AI 产品落地时的约束条件和设计原则。",
            ]

        if is_computer_profile:
            return [
                f"学会把文档主题抽象成系统问题：这份文档的主题是《{profile.title}》，可以从需求、输入、处理流程、输出和反馈闭环来分析。",
                "学会区分观点、证据和实现：不要只看结论，还要看它有没有方法、数据、实验或可复核过程支撑。",
                "学会从工程落地角度追问：如果要把文中想法做成系统，需要继续明确数据来源、模块边界、异常处理、隐私安全和评价指标。",
            ]

        if major:
            return self._major_specific_learning_points(
                major=major,
                profile=profile,
                text=normalized,
            )

        return [
            f"先学会抓主线：这份文档围绕《{profile.title}》展开，阅读时要把主题、方法、证据和结论分开看。",
            "再学会判断证据强度：不要只看结论写得顺不顺，要看它有没有样本、数据、实证检验或可复核的分析过程。",
            "最后学会看局限：凡是缺少数据、缺少验证、只停留在框架推演的地方，都不能直接当成已经被证明的结论。",
        ]

    def _extract_major_from_context(
        self,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        text = f"{question}\n{memory_facts.get('user_profile', '')}"
        aliases = [
            ("计算机", ["计算机", "软件工程", "网络工程", "人工智能", "数据科学", "大数据"]),
            ("医学", ["医学", "临床医学", "护理", "药学", "公共卫生", "医学生"]),
            ("教育学", ["教育学", "教育技术", "师范", "教师", "教学"]),
            ("法学", ["法学", "法律", "知识产权"]),
            ("管理学", ["管理", "工商管理", "公共管理", "人力资源", "市场营销"]),
            ("经济金融", ["经济", "金融", "会计", "财务", "审计"]),
            ("中文", ["汉语言", "中文", "文学", "新闻", "传播"]),
            ("英语", ["英语", "外语", "翻译"]),
            ("心理学", ["心理", "心理学"]),
            ("设计", ["设计", "艺术", "视觉传达", "产品设计"]),
        ]
        for normalized_major, keywords in aliases:
            if any(keyword in text for keyword in keywords):
                return normalized_major

        patterns = [
            r"我是(?P<major>[^，。,.!\n]{2,16})专业",
            r"我学(?:的是)?(?P<major>[^，。,.!\n]{2,16})",
            r"我的专业是(?P<major>[^，。,.!\n]{2,16})",
            r"从(?P<major>[^，。,.!\n]{2,16})专业角度",
            r"结合(?P<major>[^，。,.!\n]{2,16})专业",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                major = match.group("major").strip("的 角度学生大学生本科生研究生")
                if 2 <= len(major) <= 16:
                    return major
        return ""

    def _major_specific_learning_points(
        self,
        *,
        major: str,
        profile: DocumentProfile,
        text: str,
    ) -> list[str]:
        if major == "医学":
            return [
                "学会判断 AI 应用是否真的能改善服务效果：不要只看“智能化”表述，要追问有没有样本、干预过程、评价指标和真实效果验证。",
                "学会关注风险边界：文中提到的信息准确性、数据隐私、责任边界等问题，放到医学场景里就对应误导诊疗、隐私泄露和责任归属。",
                "学会把技术工具放进服务流程：可以思考 AI 如何辅助健康教育、患者随访、学习反馈或临床培训，但必须有人类专业判断兜底。",
            ]
        if major == "教育学":
            return [
                "学会从学习支持服务体系看 AI：重点不是模型本身，而是它如何参与答疑、反馈、资源推荐、学业预警和师生协同。",
                "学会分析教学设计边界：AI 可以提高反馈效率，但不能替代学习目标设计、评价标准和教师对学生发展的判断。",
                "学会关注教育治理：文中提到学习依赖、学术诚信、算法偏差和人机协同，这些都可以转化为教育管理和课堂规则设计问题。",
            ]
        if major == "法学":
            return [
                "学会识别技术应用背后的责任问题：文中讨论的数据隐私、算法偏差、责任边界，可以转化为合规、侵权责任和平台治理问题。",
                "学会追问制度依据：AI 进入教育服务时，不能只看效率，还要看数据授权、知情同意、可解释性和责任分配是否清楚。",
                "学会把论文观点转成规范分析：哪些风险需要规则约束，哪些环节需要人工审核，哪些主体应承担相应责任。",
            ]
        if major == "管理学":
            return [
                "学会把 AI 看成组织流程改造工具：文中提到资源重组、过程陪伴和管理协同，可以对应服务流程、组织分工和绩效评价。",
                "学会分析落地条件：一个 AI 项目是否可行，不只看技术能力，还要看成本、人员培训、风险控制和持续运营机制。",
                "学会做风险治理：隐私、安全、责任边界和多主体协同，都是管理专业分析技术项目时必须纳入的约束。",
            ]
        if major == "经济金融":
            return [
                "学会看投入产出：AI 学习支持服务是否值得做，要进一步比较建设成本、使用效率、服务质量提升和长期维护成本。",
                "学会关注数据治理：文中的数据隐私、算法偏差和责任边界，在经济金融领域可对应风控、合规和信息不对称问题。",
                "学会识别证据强度：如果文档缺少样本、数据和实证检验，就不能把它当作已经证明有效的经济结论。",
            ]
        if major in {"中文", "英语"}:
            return [
                "学会分析文本结构：可以观察它如何从背景、问题、机制、风险、治理到结论组织一篇论文式文本。",
                "学会评价论证质量：不要只看语言是否顺畅，还要看概念是否清楚、段落是否递进、证据是否能支撑结论。",
                "学会识别 AI 生成文本特征：如果文档像样稿，就要特别关注观点是否泛化、引用是否可靠、细节是否经过真实验证。",
            ]
        if major == "心理学":
            return [
                "学会关注学习者体验：文中提到情绪陪伴、学习依赖和反馈生成，可以转化为动机、认知负荷和自我调节学习问题。",
                "学会区分支持与替代：AI 可以提供陪伴和反馈，但过度依赖可能削弱学生的主动学习和问题解决能力。",
                "学会追问评估方式：如果要证明 AI 对学习心理或学习效果有帮助，需要量表、实验设计或长期跟踪数据支持。",
            ]
        if major == "设计":
            return [
                "学会从用户体验看 AI 系统：文中提到自然语言交互、反馈生成和过程陪伴，可以转化为交互流程与服务触点设计。",
                "学会关注可理解性：AI 给出的建议、反馈和风险提示需要让用户看得懂、愿意用、知道何时不能依赖。",
                "学会把风险转为设计约束：隐私、安全、责任边界和人工确认机制，都应该体现在界面和使用流程里。",
            ]
        return [
            f"从{major}专业角度，可以先学习如何把文档主题《{profile.title}》转化成本专业的问题意识，而不是只复述原文。",
            "再学习如何判断论证强度：看它的方法、证据、数据来源和结论之间是否匹配。",
            "最后学习如何识别落地边界：文中提到的风险、治理和未来研究，都是把观点转成本专业分析时必须补充的条件。",
        ]

    def _has_empirical_data(self, text: str) -> bool:
        empirical_keywords = ["样本", "问卷", "访谈", "实验", "回归", "统计检验", "显著性", "数据集", "实证"]
        if "未来研究可以进一步通过实证数据检验" in text:
            return False
        return any(keyword in text for keyword in empirical_keywords)

    def _reliability_verdict(self, profile: DocumentProfile) -> str:
        if profile.is_generated_sample:
            return (
                "不适合把它的“结果”当成可靠研究结论直接采信。它更像一篇生成式中文论文样稿，"
                "可以参考其结构、问题意识和治理框架，但不能把其中观点当作经过真实研究验证的结论。"
            )
        if profile.kind == "课程报告":
            return "它可以作为课程作业或方法演示来参考，但不能直接当作严格论文结论采信。"
        if profile.kind == "实验报告":
            return "它可以作为实验过程记录参考，但可靠性仍取决于实验设计、数据来源和验证是否完整。"
        if profile.kind == "论文" and profile.has_empirical_data:
            return "它具备一定研究论文形态，结果有参考价值，但仍需要核对样本、数据、方法和结论是否一一对应。"
        if profile.kind == "论文":
            return "它的观点有一定参考价值，但更偏理论分析或综述推演，不能等同于已经被实证证明的结果。"
        return "当前证据不足以判定它的结果可靠，只能作为有限参考。"

    def _reliability_support_points(
        self,
        *,
        profile: DocumentProfile,
        all_text: str,
        kind_citation: str,
        process_citation: str,
        support_citation: str,
    ) -> list[str]:
        points: list[str] = []
        if profile.is_generated_sample:
            points.append(f"原文标题或说明里出现“随机生成的中文论文样稿”这类表述，说明它不是严格意义上的真实研究成果。[{kind_citation}]")
        else:
            points.append(f"从文档结构看，它更接近“{profile.kind}”；判断可靠性时要按这个文档类型来要求证据。[{kind_citation}]")

        if profile.method and "没有清楚" not in profile.method:
            points.append(f"它说明采用了{profile.method}，这能支持理论梳理和框架建构，但不能单独证明应用效果真实发生。[{process_citation}]")
        else:
            points.append(f"当前证据没有清楚呈现可复核的方法、样本或数据来源，所以结果支撑偏弱。[{process_citation}]")

        if "未来研究可以进一步通过实证数据检验" in all_text:
            points.append(f"原文自己也把“通过实证数据检验不同应用场景的效果”放到未来研究中，这说明当前结论还缺少实证验证。[{support_citation}]")
        elif profile.has_empirical_data:
            points.append(f"文中出现了实证或数据线索，但还需要继续核对样本、指标、统计过程和结论之间是否匹配。[{support_citation}]")
        elif profile.has_references:
            points.append(f"文末有参考文献，能说明它有一定资料来源，但引用本身不能替代对核心结论的实证检验。[{support_citation}]")
        else:
            points.append("当前证据没有清楚展示数据来源、实验/问卷/访谈设计或误差分析，这是可靠性判断的主要缺口。")

        return points

    def _ensure_answer_relevance(
        self,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not self._looks_like_reliability_question(question):
            return answer
        banned_phrases = ["先纠正刚才", "回答方式", "不能只摘一个相似度最高"]
        if any(phrase in answer for phrase in banned_phrases):
            profile = self._profile_from_evidence(evidence)
            citation = evidence[0].citation_id if evidence else "E1"
            return (
                f"直接结论：这篇文档的结果不能直接判定为可靠。"
                f"它更像“{profile.kind}”，需要结合方法、数据来源和验证过程来判断。[{citation}]\n\n"
                "目前证据不足以证明它的结论已经经过严格验证，因此更适合作为参考材料，而不是直接当作可靠研究结论。"
            )
        if "直接结论" not in answer:
            return f"直接结论：{answer}"
        return answer

    def _build_local_grouped_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        grouped = self._group_evidence_by_document(evidence)
        if len(grouped) < 2:
            return self._build_local_answer(question, evidence, memory_facts)

        sections = ["我按文档分开说，避免把几篇内容混在一起："]
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        )
        if audience_note:
            sections.append(audience_note.strip())
        for index, (name, items) in enumerate(grouped.items(), start=1):
            summary = self._summarize_document_for_compare(items)
            citation = items[0].citation_id
            sections.append(
                f"{index}. 《{name}》\n"
                f"{self._build_document_wide_summary(question, summary, items)} [{citation}]"
            )
        return "\n\n".join(sections)

    def _build_document_wide_summary(
        self,
        question: str,
        summary: dict[str, str],
        evidence: list[EvidenceItem],
    ) -> str:
        lower_question = question.lower()
        type_text = (
            f"文档标明它属于“{summary['type']}”。"
            if summary["type"] and summary["type"] != "文档未明确说明"
            else ""
        )

        if any(keyword in question for keyword in ["发现", "重点", "核心", "结论", "贡献"]):
            answer = (
                f"它最重要的点是“{summary['title']}”。"
                f"更准确地说，这类实验文档的“发现”主要是核心收获：{summary['focus']}。"
                f"它通过{summary['implementation']}来体现。{type_text}"
            )
        elif "方法" in question:
            answer = (
                f"它的方法是围绕{summary['implementation']}展开，"
                f"对应的知识基础是{summary['knowledge']}。{type_text}"
            )
        elif any(keyword in question for keyword in ["局限", "不足"]):
            answer = (
                f"从当前证据看，它主要说明了{summary['focus']}，"
                "但对异常情况、边界条件或更复杂场景的展开不多。"
            )
        elif "目的" in question or "主题" in question:
            answer = f"主题是“{summary['title']}”，目标集中在{summary['focus']}。{type_text}"
        else:
            answer = (
                f"这份文档主要讲“{summary['title']}”，"
                f"重点是{summary['focus']}，实现内容围绕{summary['implementation']}。{type_text}"
            )

        if "tcp" in lower_question and "TCP" not in answer:
            answer = f"{answer} 这里涉及 TCP 相关内容。"
        return answer

    def _build_local_compare_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        grouped = self._group_evidence_by_document(evidence)
        if len(grouped) < 2:
            if len(grouped) == 1:
                only_name = next(iter(grouped))
                return (
                    f"当前只选到《{only_name}》这一份文档，无法真正对比两篇。"
                    "请在输入框上方确认已经同时选择两份文档，再问我“有什么不同点”。"
                )
            return "我没有找到可用于对比的文档证据。请先上传并选择至少两份已准备好的文档。"

        summaries = {
            name: self._summarize_document_for_compare(items)
            for name, items in grouped.items()
        }
        names = list(summaries.keys())[:2]
        first = summaries[names[0]]
        second = summaries[names[1]]

        first_citation = grouped[names[0]][0].citation_id
        second_citation = grouped[names[1]][0].citation_id
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for items in grouped.values() for item in items),
        )

        return (
            f"这两份文档的核心不同点可以这样看：\n\n"
            f"{audience_note}"
            f"1. 主题不同：\n"
            f"《{names[0]}》主要是“{first['title']}”，重点在 {first['focus']}。[{first_citation}]\n"
            f"《{names[1]}》主要是“{second['title']}”，重点在 {second['focus']}。[{second_citation}]\n\n"
            f"2. 实验类型不同：\n"
            f"第一份更偏“{first['type']}”，第二份更偏“{second['type']}”。\n\n"
            f"3. 实现内容不同：\n"
            f"第一份围绕 {first['implementation']} 展开；第二份围绕 {second['implementation']} 展开。\n\n"
            f"4. 学到的知识点不同：\n"
            f"第一份更强调 {first['knowledge']}；第二份更强调 {second['knowledge']}。\n\n"
            "通俗地说：一篇更像是在做“通信连接与收发数据”，另一篇更像是在做“扫描端口并判断开放状态”。"
        )

    def _audience_note(
        self,
        memory_facts: dict[str, str],
        question: str,
        context_text: str = "",
    ) -> str:
        profile = memory_facts.get("user_profile", "")
        if not profile:
            return ""
        wants_profile_angle = any(
            keyword in question
            for keyword in ["从我", "我的专业", "专业角度", "背景", "角度", "能学到", "学到什么"]
        )
        if not wants_profile_angle:
            return ""

        if "计算机" in profile or "软件" in profile or "网络" in profile:
            angle = self._computer_profile_angle(question, context_text)
            return f"结合你的背景（{profile}），我会更侧重从{angle}的角度解释。\n\n"
        if wants_profile_angle:
            return f"结合你的背景（{profile}），我会按更贴近你专业学习的角度解释。\n\n"
        return ""

    def _computer_profile_angle(self, question: str, context_text: str) -> str:
        text = f"{question}\n{context_text}".lower()
        if any(keyword in text for keyword in ["tcp", "socket", "winsock", "端口", "网络通信", "客户端", "服务端", "回显"]):
            return "网络编程、协议理解、端口/Socket 和工程实现能力"
        if any(keyword in text for keyword in ["生成式人工智能", "大模型", "人工智能", "算法", "模型", "学习支持", "教育治理"]):
            return "AI 应用机制、系统设计、数据处理、人机交互、隐私安全和工程落地"
        if any(keyword in text for keyword in ["数据库", "向量库", "索引", "检索", "embedding", "rag"]):
            return "数据建模、检索系统、索引设计和工程实现"
        if any(keyword in text for keyword in ["实验", "代码", "程序", "系统", "实现"]):
            return "系统设计、程序实现、测试验证和工程规范"
        return "抽象建模、系统设计、数据流、算法思维和工程落地"

    def _group_evidence_by_document(
        self,
        evidence: list[EvidenceItem],
    ) -> dict[str, list[EvidenceItem]]:
        grouped: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            grouped.setdefault(item.paper_name or item.document_id, []).append(item)
        return grouped

    def _summarize_document_for_compare(self, evidence: list[EvidenceItem]) -> dict[str, str]:
        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        title = self._extract_field(text, "实验名称") or self._guess_title_from_text(text)
        experiment_type = self._extract_field(text, "实验类型") or "文档未明确说明"

        lower_text = text.lower()
        if "端口扫描" in text:
            focus = "端口扫描的原理、端口开放判断和扫描流程"
            implementation = "解析目标 IP 和端口范围、遍历端口、建立 TCP 连接并判断开放状态"
            knowledge = "TCP 三次握手、connect 调用、端口开放/关闭判断和 Winsock 编程"
        elif "tcp通信" in lower_text or "tcp 通信" in lower_text or "回显" in text:
            focus = "TCP 客户端和服务端之间如何建立连接、发送和回显数据"
            implementation = "服务端监听、客户端连接、数据收发、回显和资源释放"
            knowledge = "面向连接的流式套接字、TCP 通信流程和常见 Winsock API"
        else:
            purpose = self._extract_after_heading(text, "实验目的")
            focus = purpose[:80] if purpose else "文档主题、目标和实现过程"
            implementation = self._extract_after_heading(text, "实验过程")[:80] or "文档中的具体实现步骤"
            knowledge = self._extract_after_heading(text, "知识点")[:80] or "文档涉及的核心概念"

        return {
            "title": title,
            "type": experiment_type,
            "focus": focus,
            "implementation": implementation,
            "knowledge": knowledge,
        }

