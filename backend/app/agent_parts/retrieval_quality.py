from __future__ import annotations

from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalQualityMixin:
    def _annotate_retrieval_quality(
        self,
        *,
        question: str,
        candidates: list[EvidenceItem],
        selected: list[EvidenceItem],
        top_k: int,
        retrieval_strategy: str,
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        selected_by_key = {
            self._quality_candidate_key(item): rank
            for rank, item in enumerate(selected, start=1)
        }
        selected_min_score = min((item.score for item in selected), default=0.0)
        seen: set[str] = set()
        trace: list[dict[str, Any]] = []

        for rank, item in enumerate(candidates, start=1):
            key = self._quality_candidate_key(item)
            if key in seen:
                continue
            seen.add(key)
            selected_rank = selected_by_key.get(key)
            label, reasons, metrics = self._candidate_quality_label_and_reasons(
                question=question,
                item=item,
                retrieval_strategy=retrieval_strategy,
            )
            rejection_reason = ""
            selection_status = "selected_by_retrieval_filter" if selected_rank is not None else "filtered_out"
            if selected_rank is None:
                rejection_reason = self._candidate_rejection_reason(
                    question=question,
                    item=item,
                    rank=rank,
                    top_k=top_k,
                    selected_min_score=selected_min_score,
                    metrics=metrics,
                )
            else:
                self._apply_quality_annotation(
                    item=item,
                    label=label,
                    reasons=reasons,
                    selection_status=selection_status,
                    rejection_reason="",
                )
            trace.append(
                self._quality_trace_row(
                    item=item,
                    candidate_rank=rank,
                    selected_rank=selected_rank,
                    selection_status=selection_status,
                    quality_label=label,
                    quality_reasons=reasons,
                    rejection_reason=rejection_reason,
                    metrics=metrics,
                )
            )

        selected_keys_in_trace = {f"{row.get('document_id')}:{row.get('chunk_id')}" for row in trace}
        for selected_rank, item in enumerate(selected, start=1):
            if self._quality_candidate_key(item) in selected_keys_in_trace:
                continue
            label, reasons, metrics = self._candidate_quality_label_and_reasons(
                question=question,
                item=item,
                retrieval_strategy=retrieval_strategy,
            )
            self._apply_quality_annotation(
                item=item,
                label=label,
                reasons=reasons,
                selection_status="selected_by_retrieval_filter",
                rejection_reason="",
            )
            trace.append(
                self._quality_trace_row(
                    item=item,
                    candidate_rank=len(trace) + 1,
                    selected_rank=selected_rank,
                    selection_status="selected_by_retrieval_filter",
                    quality_label=label,
                    quality_reasons=reasons,
                    rejection_reason="",
                    metrics=metrics,
                )
            )

        return selected, trace[: max(24, top_k * 6)]

    def _merge_evidence_judgments_into_quality_trace(
        self,
        *,
        trace: list[dict[str, Any]],
        judgments: list[dict[str, Any]],
        kept: list[EvidenceItem],
    ) -> list[dict[str, Any]]:
        if not trace:
            return []

        judgment_by_chunk = {
            str(judgment.get("chunk_id") or ""): judgment
            for judgment in judgments
            if str(judgment.get("chunk_id") or "")
        }
        kept_by_chunk = {item.chunk_id: item for item in kept}
        merged: list[dict[str, Any]] = []
        for row in trace:
            next_row = dict(row)
            chunk_id = str(next_row.get("chunk_id") or "")
            judgment = judgment_by_chunk.get(chunk_id)
            kept_item = kept_by_chunk.get(chunk_id)
            if judgment:
                verdict = str(judgment.get("verdict") or "")
                next_row["judge_verdict"] = verdict
                next_row["judge_reason"] = str(judgment.get("reason") or "")
                confidence = judgment.get("confidence")
                next_row["judge_confidence"] = float(confidence) if confidence is not None else None
                if kept_item is not None:
                    next_row["selection_status"] = "selected_for_answer"
                    next_row["citation_id"] = kept_item.citation_id
                    self._apply_quality_annotation(
                        item=kept_item,
                        label=str(next_row.get("quality_label") or ""),
                        reasons=list(next_row.get("quality_reasons") or []),
                        selection_status="selected_for_answer",
                        rejection_reason="",
                    )
                elif next_row.get("selection_status") == "selected_by_retrieval_filter":
                    next_row["selection_status"] = "rejected_by_evidence_judge"
                    next_row["rejection_reason"] = f"证据裁判判定为 {verdict}：{next_row['judge_reason']}"
            merged.append(next_row)
        return merged

    def _quality_trace_summary(self, trace: list[dict[str, Any]]) -> str:
        if not trace:
            return "没有可标记的召回候选证据。"
        counts: dict[str, int] = {}
        labels: dict[str, int] = {}
        for row in trace:
            status = str(row.get("selection_status") or "unknown")
            label = str(row.get("quality_label") or "unknown")
            counts[status] = counts.get(status, 0) + 1
            labels[label] = labels.get(label, 0) + 1
        return (
            "已为召回候选打上质量标签："
            f"候选 {len(trace)} 条，保留 {counts.get('selected_by_retrieval_filter', 0)} 条，"
            f"过滤 {counts.get('filtered_out', 0)} 条；"
            f"strong {labels.get('strong', 0)}、medium {labels.get('medium', 0)}、"
            f"weak {labels.get('weak', 0)}、noise {labels.get('noise', 0)}。"
        )

    def _quality_candidate_key(self, item: EvidenceItem) -> str:
        return f"{item.document_id}:{item.chunk_id}"

    def _candidate_quality_label_and_reasons(
        self,
        *,
        question: str,
        item: EvidenceItem,
        retrieval_strategy: str,
    ) -> tuple[str, list[str], dict[str, Any]]:
        text = self._sanitize_evidence_text(f"{item.section or ''}\n{item.quote}\n{item.text}")
        relevance = self._question_relevance_score(question, text)
        readability = self._readable_text_score(text)
        matched_keywords = self._quality_matched_keywords(question=question, text=text)
        table_like = self._is_table_like_text(text)
        noise_flags = self._quality_noise_flags(text=text, table_like=table_like)

        if noise_flags:
            label = "noise"
        elif item.score >= 0.72 or relevance >= 0.24:
            label = "strong"
        elif item.score >= 0.45 or relevance >= 0.1 or readability >= 0.55:
            label = "medium"
        else:
            label = "weak"

        reasons: list[str] = []
        if item.score >= 0.72:
            reasons.append("召回分数较高")
        elif item.score >= 0.45:
            reasons.append("召回分数中等")
        else:
            reasons.append("召回分数偏低")
        if relevance >= 0.24:
            reasons.append("与问题关键词/语义匹配较强")
        elif relevance >= 0.1:
            reasons.append("与问题有一定匹配")
        else:
            reasons.append("与问题直接匹配较弱")
        if readability >= 0.55:
            reasons.append("文本可读性较好")
        elif readability < 0.25:
            reasons.append("文本可读性较低")
        if matched_keywords:
            reasons.append(f"命中关键词：{'、'.join(matched_keywords[:5])}")
        if item.vector_score is not None:
            reasons.append("来自向量召回")
        if item.sparse_score is not None:
            reasons.append("来自 BM25 召回")
        if item.rule_score is not None:
            reasons.append("来自规则/结构候选")
        if item.rrf_score is not None or item.score_source == "rrf_fusion":
            reasons.append("经过 RRF 融合排序")
        if table_like:
            reasons.append("表格/结构化文本")
        if retrieval_strategy:
            reasons.append(f"检索策略：{retrieval_strategy}")
        reasons.extend(noise_flags)

        metrics = {
            "relevance_score": round(relevance, 3),
            "readability_score": round(readability, 3),
            "matched_keywords": matched_keywords[:8],
            "table_like": table_like,
            "noise_flags": noise_flags,
        }
        return label, list(dict.fromkeys(reasons)), metrics

    def _candidate_rejection_reason(
        self,
        *,
        question: str,
        item: EvidenceItem,
        rank: int,
        top_k: int,
        selected_min_score: float,
        metrics: dict[str, Any],
    ) -> str:
        if metrics.get("noise_flags"):
            return f"疑似噪声或元数据：{'、'.join(metrics['noise_flags'])}"
        if metrics.get("table_like") and not self._looks_like_table_question(question):
            return "表格型证据，但问题没有要求表格或结构化数值。"
        if float(metrics.get("relevance_score") or 0.0) < 0.08 and item.score < 0.45:
            return "问题相关度和召回分数都偏低。"
        if rank > max(1, top_k) and selected_min_score and item.score < selected_min_score:
            return "未进入最终证据集：排序分数低于已保留证据。"
        return "未进入最终证据集：被去重、文档均衡或问题类型专项过滤挤出。"

    def _quality_trace_row(
        self,
        *,
        item: EvidenceItem,
        candidate_rank: int,
        selected_rank: int | None,
        selection_status: str,
        quality_label: str,
        quality_reasons: list[str],
        rejection_reason: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "citation_id": item.citation_id if selected_rank is not None else "",
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "paper_name": item.paper_name,
            "page": item.page,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "section": item.section,
            "chunk_type": item.chunk_type,
            "candidate_rank": candidate_rank,
            "selected_rank": selected_rank,
            "selection_status": selection_status,
            "quality_label": quality_label,
            "quality_reasons": quality_reasons,
            "rejection_reason": rejection_reason,
            "score": round(float(item.score), 3),
            "relevance_score": float(metrics.get("relevance_score") or 0.0),
            "readability_score": float(metrics.get("readability_score") or 0.0),
            "vector_score": item.vector_score,
            "sparse_score": item.sparse_score,
            "rule_score": item.rule_score,
            "rrf_score": item.rrf_score,
            "final_score": item.final_score,
            "score_source": item.score_source,
            "matched_keywords": list(metrics.get("matched_keywords") or []),
            "quote": self._best_readable_quote(item.quote or item.text, limit=180),
        }

    def _quality_matched_keywords(self, *, question: str, text: str) -> list[str]:
        normalized = text.lower()
        matches: list[str] = []
        for keyword in self._question_keywords(question)[:16]:
            if keyword and (keyword in text or keyword.lower() in normalized) and keyword not in matches:
                matches.append(keyword)
            if len(matches) >= 8:
                break
        return matches

    def _quality_noise_flags(self, *, text: str, table_like: bool) -> list[str]:
        flags: list[str] = []
        if self._looks_like_front_matter_noise(text):
            flags.append("疑似封面/元数据噪声")
        if self._optional_bool("_looks_like_field_or_metadata_sentence", text):
            flags.append("疑似字段/元数据句")
        if self._optional_bool("_looks_like_submission_or_assignment_noise", text):
            flags.append("疑似作业提交信息噪声")
        if self._optional_bool("_looks_like_code_heavy_text", text):
            flags.append("疑似代码密集片段")
        return flags

    def _optional_bool(self, method_name: str, value: str) -> bool:
        method = getattr(self, method_name, None)
        if method is None:
            return False
        try:
            return bool(method(value))
        except TypeError:
            return False

    def _apply_quality_annotation(
        self,
        *,
        item: EvidenceItem,
        label: str,
        reasons: list[str],
        selection_status: str,
        rejection_reason: str,
    ) -> None:
        item.quality_label = label
        item.quality_reasons = reasons
        item.selection_status = selection_status
        item.rejection_reason = rejection_reason
