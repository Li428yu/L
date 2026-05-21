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


class AgentVerificationMixin:
    def _judge_evidence(self, state: PaperAgentState) -> PaperAgentState:
        evidence = state.get("evidence", [])
        question = state["question"]
        strategy = state.get("retrieval_strategy", "")
        if not evidence:
            return {
                **state,
                "evidence_judgments": [],
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="evidence_judge",
                        title="证据裁判",
                        detail="没有可裁判的证据，后续回答会按证据不足处理。",
                    ),
                ],
            }

        strict = self._strict_evidence_judge_question(question)
        allow_tables = self._looks_like_table_question(question)
        kept: list[EvidenceItem] = []
        judgments: list[dict[str, Any]] = []
        verdict_counts = {"direct": 0, "supporting": 0, "background": 0, "reject": 0}

        for item in evidence:
            judgment = self._judge_single_evidence(
                question=question,
                item=item,
                retrieval_strategy=strategy,
                allow_tables=allow_tables,
            )
            verdict = str(judgment["verdict"])
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            judgments.append(judgment)

            if verdict in {"direct", "supporting"}:
                kept.append(item)
            elif verdict == "background" and not strict:
                kept.append(item)

        if kept:
            kept = self._renumber_evidence(kept)
            citation_by_chunk = {item.chunk_id: item.citation_id for item in kept}
            for judgment in judgments:
                next_citation = citation_by_chunk.get(str(judgment.get("chunk_id", "")))
                if next_citation:
                    judgment["citation_id"] = next_citation
        elif not strict:
            kept = evidence

        return {
            **state,
            "evidence": kept,
            "evidence_judgments": judgments,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="evidence_judge",
                    title="证据裁判",
                    detail=(
                        "已逐条判断证据是否直接支撑问题："
                        f"直接 {verdict_counts.get('direct', 0)} 条、辅助 {verdict_counts.get('supporting', 0)} 条、"
                        f"背景 {verdict_counts.get('background', 0)} 条、拒绝 {verdict_counts.get('reject', 0)} 条；"
                        f"最终保留 {len(kept)} 条进入回答。"
                    ),
                ),
            ],
        }

    def _verify_answer(self, state: PaperAgentState) -> PaperAgentState:
        verification = self._cross_verify_answer(
            question=state["question"],
            answer=state.get("answer", ""),
            evidence=state.get("evidence", []),
            answer_strategy=state.get("answer_strategy", ""),
        )
        detail = (
            f"完成引用核对：状态 {verification['status']}；"
            f"引用 {verification['citation_count']} 个，缺失引用 {len(verification['missing_citations'])} 个，"
            f"弱支撑引用 {len(verification['weak_citations'])} 个。{verification['summary']}"
        )
        answer = state.get("answer", "")
        if verification["status"] == "fail" and "交叉验证提示" not in answer:
            answer = (
                f"{answer}\n\n"
                "交叉验证提示：这轮回答中有引用没有找到对应证据，建议以右侧证据面板为准，或重新提问让我重新核对。"
            )
        return {
            **state,
            "answer": answer,
            "verification": verification,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="verifier",
                    title="交叉验证",
                    detail=detail,
                ),
            ],
        }

    def _friendly_answer_strategy(self, strategy: str) -> str:
        labels = {
            "local_compound_answer": "按用户顺序逐项回答复合任务",
            "local_reference_answer": "本地规则整理参考文献",
            "local_field_lookup_answer": "本地规则提取指定字段",
            "local_structured_review_answer": "本地规则生成结构化阅读报告",
            "local_title_alignment_answer": "本地规则判断题目匹配",
            "local_reliability_answer": "本地规则判断可靠性",
            "local_research_limitation_answer": "本地规则分析文章研究局限",
            "local_compare_answer": "本地规则对比多文档",
            "local_document_answer": "本地规则概括文档",
            "missing_evidence_refusal": "证据不足，拒绝硬答",
            "model_answer": "模型基于证据生成",
            "model_unavailable": "模型不可用，未本地兜底",
            "local_fallback_answer": "模型失败后本地降级回答",
        }
        return labels.get(strategy, strategy)

    def _evidence_quality(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        fallback_used: bool,
        answer_strategy: str,
    ) -> str:
        if answer_strategy == "model_unavailable":
            return "unavailable"
        if answer_strategy == "missing_evidence_refusal":
            return "insufficient"
        if fallback_used:
            return "fallback"
        if not evidence:
            return "none"
        best_score = max((item.score for item in evidence), default=0.0)
        best_relevance = max(
            (self._question_relevance_score(question, item.text) for item in evidence),
            default=0.0,
        )
        if answer_strategy.startswith("local_") and len(evidence) >= 1 and best_score >= 0.75:
            return "strong"
        if best_score >= 0.7 or best_relevance >= 0.25:
            return "strong"
        if best_score >= 0.45 or best_relevance >= 0.12:
            return "medium"
        return "weak"

    def _build_trace_diagnosis(
        self,
        *,
        intent: str,
        retrieval_strategy: str,
        answer_strategy: str,
        evidence_quality: str,
        evidence: list[EvidenceItem],
        fallback_used: bool,
        verification: dict[str, Any] | None = None,
    ) -> str:
        verification = verification or {}
        if verification.get("status") == "fail":
            return f"交叉验证未通过：{verification.get('summary', '回答和证据引用存在不一致。')}"
        if verification.get("status") == "warn":
            return f"交叉验证提示：{verification.get('summary', '部分证据支撑较弱，需要谨慎阅读。')}"
        if answer_strategy == "model_unavailable":
            return "模型调用失败或超时；按当前配置，本轮未使用本地规则生成替代答案。"
        if fallback_used:
            return "模型调用失败或超时，本轮已改用原文证据和本地规则生成回答。"
        if answer_strategy == "missing_evidence_refusal":
            return "检索到的段落和问题关联不够强，本轮选择停止硬答并提示证据不足。"
        if not evidence:
            return "本轮没有返回可用证据，需要检查文档是否准备完成或问题是否依赖文档内容。"

        pages = sorted({item.page for item in evidence if item.page})
        page_text = "、".join(str(page) for page in pages[:3]) if pages else "未知位置"
        return (
            f"本轮识别为「{self._friendly_intent(intent)}」，"
            f"使用「{self._friendly_retrieval_strategy(retrieval_strategy)}」，"
            f"命中 {len(evidence)} 条证据，主要来自第 {page_text} 页/段；"
            f"回答策略是「{self._friendly_answer_strategy(answer_strategy)}」。"
        )

    def _build_retrieval_debug(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        retrieval_strategy: str,
        answer: str,
        final_prompt_evidence: list[str],
    ) -> list[RetrievalDebugItem]:
        prompt_text = "\n".join(final_prompt_evidence)
        items: list[RetrievalDebugItem] = []
        for item in evidence:
            matched_keywords = self._debug_matched_keywords(
                question=question,
                text=f"{item.section or ''}\n{item.quote}\n{item.text}",
                retrieval_strategy=retrieval_strategy,
            )
            items.append(
                RetrievalDebugItem(
                    citation_id=item.citation_id,
                    chunk_id=item.chunk_id,
                    page=item.page,
                    section=item.section,
                    score=item.score,
                    retrieval_strategy=retrieval_strategy,
                    selected_by=self._debug_selected_by(retrieval_strategy),
                    matched_keywords=matched_keywords,
                    reason=self._debug_reason_for_evidence(
                        question=question,
                        retrieval_strategy=retrieval_strategy,
                        matched_keywords=matched_keywords,
                        item=item,
                    ),
                    used_in_answer=f"[{item.citation_id}]" in answer,
                    used_in_prompt=f"[{item.citation_id}]" in prompt_text,
                    quote=self._best_quote_for_question(question, item.text, limit=180),
                )
            )
        return items

    def _debug_selected_by(self, retrieval_strategy: str) -> str:
        labels = {
            "compound_request": "按子任务顺序组合检索",
            "reference_section": "规则定位参考文献区",
            "field_lookup": "按字段边界精确提取",
            "comparison_overview": "按文档抽取概览片段",
            "structured_review": "按模板抽取多类证据",
            "title_alignment": "题目与结论专项规则",
            "reliability_check": "可靠性专项规则",
            "research_limitation": "文章研究局限专项规则",
            "document_overview": "整篇文档关键词/结构检索",
            "vector_similarity": "Chroma 向量相似度召回后重排",
            "hybrid_soft": "模型软意图 + 向量/关键词/章节混合重排",
            "hybrid_reference": "参考文献候选 + 混合重排",
            "hybrid_field_lookup": "字段候选 + 混合重排",
            "hybrid_comparison": "对比候选 + 混合重排",
            "hybrid_judgment": "判断类候选 + 混合重排",
            "hybrid_limitation": "局限候选 + 混合重排",
            "hybrid_overview": "全文角色段落 + 混合重排",
            "no_retrieval": "未检索文档",
        }
        return labels.get(retrieval_strategy, retrieval_strategy)

    def _debug_matched_keywords(
        self,
        *,
        question: str,
        text: str,
        retrieval_strategy: str,
    ) -> list[str]:
        strategy_keywords = {
            "compound_request": self._compound_focus_keywords_for_question(question),
            "reference_section": ["参考文献", "References", "[1]", "[2]", "[3]"],
            "field_lookup": self._field_lookup_debug_keywords(question),
            "comparison_overview": ["实验名称", "实验类型", "实验目的", "主题", "方法", "结论"],
            "structured_review": [
                "摘要",
                "本文围绕",
                "采用",
                "文献分析",
                "认知支架",
                "资源重组",
                "课程学习场景",
                "风险",
                "人机协同",
                "未来研究",
                "实证数据",
                "参考文献",
            ],
            "title_alignment": [
                "题目",
                "结论",
                "机制",
                "风险",
                "治理路径",
                "未来研究",
                "实证数据",
            ],
            "reliability_check": [
                "随机生成",
                "论文样稿",
                "采用",
                "文献分析",
                "情境推演",
                "机制建构",
                "未来研究",
                "实证数据",
                "参考文献",
            ],
            "research_limitation": [
                "局限性",
                "研究局限",
                "研究不足",
                "结论与展望",
                "未来研究",
                "实证数据",
                "验证",
                "检验",
                "样本",
                "数据来源",
                "文献分析",
                "情境推演",
                "机制建构",
            ],
            "document_overview": self._overview_focus_keywords(question),
            "vector_similarity": self._question_keywords(question),
        }
        keywords = strategy_keywords.get(retrieval_strategy, self._question_keywords(question))
        matches: list[str] = []
        for keyword in keywords:
            if keyword and keyword in text and keyword not in matches:
                matches.append(keyword)
            if len(matches) >= 8:
                break
        return matches

    def _debug_reason_for_evidence(
        self,
        *,
        question: str,
        retrieval_strategy: str,
        matched_keywords: list[str],
        item: EvidenceItem,
    ) -> str:
        keyword_text = "、".join(matched_keywords) if matched_keywords else "没有明显关键词，主要依赖向量相似度/结构位置"
        if retrieval_strategy == "compound_request":
            tasks = self._parse_compound_tasks(question)
            task_text = " → ".join(task.label for task in tasks) if tasks else "多个子任务"
            return f"用户问题被拆成“{task_text}”，系统按这些子任务分别抽取摘要、方法、参考文献、收获或可靠性相关证据；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "reference_section":
            return f"问题像是在问参考文献，系统优先扫描文末参考文献区；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "field_lookup":
            return f"问题是在提取文档字段，系统只截取目标字段到下一个字段边界之间的内容；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "document_overview":
            return f"问题属于整篇分析/概括，系统优先找和问题目标相关的结构段落；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "structured_review":
            return f"用户提出了多步骤模板要求，系统同时抽取摘要、方法、主体内容、风险治理、结论和参考文献证据；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "reliability_check":
            return f"问题是在判断可靠性，系统优先找文档类型、方法、数据验证、未来研究和参考文献信息；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "research_limitation":
            return f"问题是在问文章本身的研究局限，系统优先找未来研究、方法边界、数据/样本和实证验证缺口；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "title_alignment":
            return f"问题是在判断题目与结论是否匹配，系统优先找题目关键词、结论、风险、治理和不足；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "comparison_overview":
            return f"问题是多文档对比，系统为每篇文档抽取能代表主题和方法的片段；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy.startswith("hybrid_"):
            return f"系统先用模型软判断用户意图，再合并专项候选、向量召回、关键词命中和章节角色段落统一重排；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "vector_similarity":
            return f"系统把问题转成检索向量，在 Chroma 中召回相似 chunk，再按相关度和可读性筛选；该 chunk 的 score 为 {item.score:.3f}，命中 {keyword_text}。"
        return f"该证据由「{self._friendly_retrieval_strategy(retrieval_strategy)}」选中，命中 {keyword_text}。"

    def _strict_evidence_judge_question(self, question: str) -> bool:
        return any(
            checker(question)
            for checker in [
                self._looks_like_reference_question,
                self._looks_like_reliability_question,
                self._looks_like_research_limitation_question,
                self._looks_like_title_alignment_question,
            ]
        )

    def _judge_single_evidence(
        self,
        *,
        question: str,
        item: EvidenceItem,
        retrieval_strategy: str,
        allow_tables: bool,
    ) -> dict[str, Any]:
        text = self._sanitize_evidence_text(f"{item.section or ''}\n{item.quote}\n{item.text}")
        verdict = "background"
        confidence = 0.45
        reason = "证据可作为背景，但和问题目标不是直接对应。"

        if self._is_table_like_text(text) and not allow_tables:
            verdict = "reject"
            confidence = 0.7
            reason = "该片段主要是表格或结构化数据，而问题没有要求核对表格。"
        elif self._looks_like_field_lookup_question(question):
            targets = self._field_lookup_targets(question)
            labels = [self._field_label(target) for target in targets]
            if any(label in text for label in labels):
                verdict = "direct"
                confidence = 0.94
                reason = "片段已经按用户询问的字段边界截取，可直接回答字段提取类问题。"
            else:
                verdict = "reject"
                confidence = 0.78
                reason = "问题只要求特定字段，但该片段没有命中目标字段。"
        elif self._looks_like_reference_question(question):
            if self._looks_like_reference_section_text(text):
                verdict = "direct"
                confidence = 0.92
                reason = "片段位于参考文献区域，可直接回答参考文献类问题。"
            else:
                verdict = "reject"
                confidence = 0.86
                reason = "问题要求参考文献，但该片段不是参考文献区域。"
        elif self._looks_like_research_limitation_question(question):
            score = self._research_limitation_relevance_score(
                text=text,
                section=item.section or "",
                index=0,
                total=1,
            )
            if score >= 2.0:
                verdict = "direct"
                confidence = min(0.96, 0.62 + score * 0.08)
                reason = "片段包含未来研究、实证验证、研究方法或结论边界，可支撑文章研究局限判断。"
            elif score >= 0.85:
                verdict = "supporting"
                confidence = min(0.82, 0.5 + score * 0.08)
                reason = "片段能辅助判断文章研究边界，但不是最直接的局限说明。"
            else:
                verdict = "reject"
                confidence = 0.82
                reason = "片段更像正文研究对象的困难/风险，不能直接当作文章本身的研究局限。"
        elif self._looks_like_reliability_question(question):
            score = self._reliability_relevance_score(text)
            if score >= 8:
                verdict = "direct"
                confidence = 0.88
                reason = "片段包含文档类型、方法、数据来源、验证或未来研究线索，可直接用于可靠性判断。"
            elif score >= 3:
                verdict = "supporting"
                confidence = 0.7
                reason = "片段提供了可靠性判断的辅助线索。"
            else:
                verdict = "reject"
                confidence = 0.76
                reason = "片段缺少方法、数据、验证或结论支撑信息，不适合用于可靠性判断。"
        elif self._looks_like_title_alignment_question(question):
            keywords = ["题目", "机制", "风险", "治理", "结论", "未来研究", "实证数据"]
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits >= 2:
                verdict = "direct"
                confidence = 0.78
                reason = "片段包含题目、机制、风险、治理或结论线索，可用于核对题目与结论匹配。"
            elif hits == 1:
                verdict = "supporting"
                confidence = 0.62
                reason = "片段只命中一个匹配线索，适合作为辅助证据。"
        else:
            relevance = self._question_relevance_score(question, text)
            if relevance >= 0.24 or item.score >= 0.82:
                verdict = "direct"
                confidence = min(0.9, 0.58 + relevance)
                reason = "片段和问题关键词/语义关联较强，可直接进入回答。"
            elif relevance >= 0.08 or item.score >= 0.45:
                verdict = "supporting"
                confidence = 0.62
                reason = "片段与问题有一定关联，可作为辅助证据。"

        return {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "reason": reason,
            "retrieval_strategy": retrieval_strategy,
        }

    def _cross_verify_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
        answer_strategy: str,
    ) -> dict[str, Any]:
        cited_ids = self._citation_ids_from_answer(answer)
        evidence_by_id = {item.citation_id: item for item in evidence}
        missing_citations = [citation_id for citation_id in cited_ids if citation_id not in evidence_by_id]
        weak_citations: list[dict[str, Any]] = []

        for sentence, sentence_citations in self._sentences_with_citations(answer):
            for citation_id in sentence_citations:
                item = evidence_by_id.get(citation_id)
                if not item:
                    continue
                overlap = self._sentence_evidence_overlap(sentence, item.text)
                if overlap < 0.08:
                    weak_citations.append(
                        {
                            "citation_id": citation_id,
                            "overlap": round(overlap, 3),
                            "reason": "回答句子和对应证据的可解释重叠较低，需要人工核对。",
                        }
                    )

        needs_evidence = not self._looks_like_meta_question(question)
        refusal_answer = any(phrase in answer for phrase in ["没有找到", "证据不足", "不能可靠", "不应该硬编"])
        uncited_answer = needs_evidence and evidence and not cited_ids and not refusal_answer

        if missing_citations:
            status = "fail"
            summary = "回答引用了不存在的证据编号。"
        elif uncited_answer:
            status = "warn"
            summary = "回答没有显式引用证据，可信度需要降低。"
        elif weak_citations:
            status = "warn"
            summary = "部分引用和回答句子的直接重叠较低，已标记为弱支撑。"
        else:
            status = "pass"
            summary = "回答中的引用都能在本轮证据中找到。"

        if answer_strategy == "model_unavailable":
            status = "pass"
            summary = "模型没有返回最终答案，本轮没有使用本地规则硬答。"
        elif answer_strategy == "missing_evidence_refusal" or (not evidence and refusal_answer):
            status = "pass"
            summary = "回答已经按证据不足处理，没有继续硬答。"

        return {
            "status": status,
            "summary": summary,
            "citation_count": len(cited_ids),
            "missing_citations": missing_citations,
            "weak_citations": weak_citations[:5],
            "uncited_answer": uncited_answer,
        }

    def _citation_ids_from_answer(self, answer: str) -> list[str]:
        return [f"E{value}" for value in dict.fromkeys(re.findall(r"\[E(\d+)\]", answer))]

    def _sentences_with_citations(self, answer: str) -> list[tuple[str, list[str]]]:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？.!?])\s+|(?<=。)|(?<=！)|(?<=？)|\n+", answer)
            if part.strip()
        ]
        results: list[tuple[str, list[str]]] = []
        for sentence in sentences:
            citation_ids = [f"E{value}" for value in re.findall(r"\[E(\d+)\]", sentence)]
            if citation_ids:
                results.append((sentence, list(dict.fromkeys(citation_ids))))
        return results

    def _sentence_evidence_overlap(self, sentence: str, evidence_text: str) -> float:
        sentence_text = re.sub(r"\[E\d+\]", "", self._sanitize_evidence_text(sentence))
        evidence_text = self._sanitize_evidence_text(evidence_text)
        meaningful_chars = [
            char
            for char in re.findall(r"[\u4e00-\u9fff]", sentence_text)
            if char not in set("这篇份个的了呢吗啊和与及或是在中里上下主要可以说明因此因为所以当前原文证据文章研究局限")
        ]
        if not meaningful_chars:
            return 1.0
        evidence_chars = set(re.findall(r"[\u4e00-\u9fff]", evidence_text))
        hits = sum(1 for char in meaningful_chars if char in evidence_chars)
        return hits / max(len(meaningful_chars), 1)

