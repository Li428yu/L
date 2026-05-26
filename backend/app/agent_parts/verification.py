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
        self._emit_status("正在判断证据是否支撑问题...")
        evidence = state.get("evidence", [])
        question = state["question"]
        strategy = state.get("retrieval_strategy", "")
        if not evidence:
            return {
                **state,
                "evidence_judgments": [],
                "evidence_quality": "none",
                "evidence_quality_trace": state.get("evidence_quality_trace", []),
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
            failure_closed_detail = ""
        else:
            failure_closed_detail = (
                "没有候选证据通过裁判；本轮不会回填原始候选，后续回答会按证据不足处理。"
            )

        evidence_quality = self._evidence_quality(
            question=question,
            evidence=kept,
            fallback_used=False,
            answer_strategy="model_answer",
        )
        merge_quality_trace = getattr(self, "_merge_evidence_judgments_into_quality_trace", None)
        evidence_quality_trace = (
            merge_quality_trace(
                trace=state.get("evidence_quality_trace", []),
                judgments=judgments,
                kept=kept,
            )
            if merge_quality_trace
            else state.get("evidence_quality_trace", [])
        )
        return {
            **state,
            "evidence": kept,
            "evidence_judgments": judgments,
            "evidence_quality_trace": evidence_quality_trace,
            "evidence_quality": evidence_quality,
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
                        f"{failure_closed_detail}"
                    ),
                ),
            ],
        }

    def _verify_answer(self, state: PaperAgentState) -> PaperAgentState:
        self._emit_status("正在核对引用...")
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
            "local_reference_answer": "本地规则整理参考文献",
            "local_field_lookup_answer": "本地规则提取指定字段",
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
        embedding_trace: dict[str, Any] | None = None,
    ) -> str:
        verification = verification or {}
        embedding_trace = embedding_trace or {}
        if verification.get("status") == "fail":
            return f"交叉验证未通过：{verification.get('summary', '回答和证据引用存在不一致。')}"
        if verification.get("status") == "warn":
            return f"交叉验证提示：{verification.get('summary', '部分证据支撑较弱，需要谨慎阅读。')}"
        if answer_strategy == "model_unavailable":
            return "模型调用失败或超时；按当前配置，本轮未使用本地规则生成替代答案。"
        if embedding_trace.get("embedding_used_fallback"):
            reason = str(embedding_trace.get("embedding_fallback_reason") or "").strip()
            suffix = f"原因：{reason}" if reason else "请检查 embedding 服务配置、额度或网络。"
            return f"Embedding 已降级为本地备用检索，本轮召回质量可能下降。{suffix}"
        if fallback_used:
            return "模型调用失败或超时，本轮已改用原文证据和本地规则生成回答。"
        if answer_strategy == "missing_evidence_refusal":
            return "检索到的段落和问题关联不够强，本轮选择停止硬答并提示证据不足。"
        if not evidence:
            return "本轮没有返回可用证据，需要检查文档是否准备完成或问题是否依赖文档内容。"

        pages = sorted({item.page for item in evidence if item.page})
        page_text = "、".join(str(page) for page in pages[:3]) if pages else "未知位置"
        fusion = "，并经过 RRF 融合排序" if retrieval_strategy.startswith("hybrid_") else ""
        return (
            f"本轮识别为「{self._friendly_intent(intent)}」，"
            f"使用「{self._friendly_retrieval_strategy(retrieval_strategy)}」{fusion}，"
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
                    page_start=item.page_start,
                    page_end=item.page_end,
                    section=item.section,
                    score=item.score,
                    vector_score=item.vector_score,
                    sparse_score=item.sparse_score,
                    rule_score=item.rule_score,
                    rrf_score=item.rrf_score,
                    final_score=item.final_score,
                    score_source=item.score_source,
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
                    quality_label=item.quality_label,
                    quality_reasons=item.quality_reasons,
                    selection_status=item.selection_status,
                    rejection_reason=item.rejection_reason,
                )
            )
        return items

    def _debug_selected_by(self, retrieval_strategy: str) -> str:
        labels = {
            "reference_section": "规则定位参考文献区",
            "field_lookup": "按字段边界精确提取",
            "comparison_overview": "按文档抽取概览片段",
            "document_overview": "整篇文档关键词/结构检索",
            "vector_similarity": "Chroma 向量相似度召回",
            "hybrid_soft": "Dense 向量 + BM25 sparse 双路召回，RRF 融合排序",
            "hybrid_reference": "参考文献候选 + 双路召回，RRF 融合排序",
            "hybrid_field_lookup": "字段候选 + 双路召回，RRF 融合排序",
            "hybrid_comparison": "对比候选 + 双路召回，RRF 融合排序",
            "hybrid_overview": "全文角色候选 + 双路召回，RRF 融合排序",
            "hybrid_retry": "扩大候选数量 + 双路召回，RRF 融合排序",
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
            "reference_section": ["参考文献", "References", "[1]", "[2]", "[3]"],
            "field_lookup": self._field_lookup_debug_keywords(question),
            "comparison_overview": ["实验名称", "实验类型", "实验目的", "主题", "方法", "结论"],
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
        if retrieval_strategy == "reference_section":
            return f"问题像是在问参考文献，系统优先扫描文末参考文献区；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "field_lookup":
            return f"问题是在提取文档字段，系统只截取目标字段到下一个字段边界之间的内容；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "document_overview":
            return f"问题属于整篇分析/概括，系统优先找和问题目标相关的结构段落；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "comparison_overview":
            return f"问题是多文档对比，系统为每篇文档抽取能代表主题和方法的片段；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy.startswith("hybrid_"):
            return f"系统合并 Dense 向量召回和 BM25 sparse 召回，并在字段/参考文献等明确结构问题中加入少量结构候选，用 RRF 融合排序；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "vector_similarity":
            return f"系统把问题转成检索向量，在 Chroma 中召回相似 chunk，再按相关度和可读性筛选；该 chunk 的 score 为 {item.score:.3f}，命中 {keyword_text}。"
        return f"该证据由「{self._friendly_retrieval_strategy(retrieval_strategy)}」选中，命中 {keyword_text}。"

    def _strict_evidence_judge_question(self, question: str) -> bool:
        return any(
            checker(question)
            for checker in [
                self._looks_like_reference_question,
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
        bracketed = re.findall(r"\[E(\d+)\]", answer)
        bare = re.findall(r"(?<![A-Za-z0-9])E(\d+)(?![A-Za-z0-9])", answer)
        return [f"E{value}" for value in dict.fromkeys([*bracketed, *bare])]

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
        sentence_tokens = self._verification_overlap_tokens(sentence_text)
        if not sentence_tokens:
            return 1.0
        evidence_tokens = set(self._verification_overlap_tokens(evidence_text))
        if not evidence_tokens:
            return 0.0
        hits = sum(1 for token in sentence_tokens if token in evidence_tokens)
        return hits / max(len(sentence_tokens), 1)

    def _verification_overlap_tokens(self, text: str) -> list[str]:
        normalized = self._sanitize_evidence_text(text).lower()
        blocked_chars = set("这篇份个的了呢吗啊和与及或是在中里上下主要可以说明因此因为所以当前原文证据文章研究局限")
        blocked_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "are",
            "was",
            "were",
            "can",
            "may",
            "into",
            "using",
        }
        tokens: list[str] = [
            char
            for char in re.findall(r"[\u4e00-\u9fff]", normalized)
            if char not in blocked_chars
        ]
        tokens.extend(
            token
            for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", normalized)
            if token not in blocked_words
        )
        bilingual_aliases = {
            "治理": "govern",
            "映射": "map",
            "测量": "measure",
            "度量": "measure",
            "衡量": "measure",
            "管理": "manage",
            "识别": "identify",
            "防护": "protect",
            "保护": "protect",
            "检测": "detect",
            "响应": "respond",
            "恢复": "recover",
            "可信": "trustworthy",
            "风险": "risk",
            "核心功能": "functions",
            "功能": "function",
            "特征": "characteristics",
            "有效": "valid",
            "可靠": "reliable",
            "安全": "safe",
            "弹性": "resilient",
            "可问责": "accountable",
            "透明": "transparent",
            "可解释": "explainable",
            "可解读": "interpretable",
            "隐私": "privacy",
            "公平": "fair",
            "偏见": "bias",
            "数据集": "dataset",
            "训练集": "dataset",
            "训练数据": "dataset",
            "图像": "image",
            "图片": "image",
            "文本": "text",
            "图像文本": "image-text",
            "互联网": "internet",
            "外部链接": "outbound links",
            "抓取": "scraped",
            "网页": "web",
            "点赞": "karma",
            "评分": "karma",
            "掩码": "masks",
            "分割": "segmentation",
            "许可": "licensed",
            "保护隐私": "privacy",
            "隐私保护": "privacy",
            "预训练": "pre-training",
            "掩码语言模型": "masked language model",
            "下一句预测": "next sentence prediction",
            "目标": "objective",
            "零样本": "zero-shot",
        }
        for phrase, alias in bilingual_aliases.items():
            if phrase in normalized:
                tokens.append(alias)
        return list(dict.fromkeys(tokens))

