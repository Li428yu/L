from __future__ import annotations

from typing import Any

from backend.app.models import EvidenceItem


class AgentEvidenceCoverageMixin:
    def _evidence_coverage_decision(
        self,
        *,
        state: dict[str, Any],
        prompt_evidence: list[EvidenceItem],
    ) -> dict[str, Any]:
        question = str(state.get("question") or "")
        evidence = list(state.get("evidence") or [])
        if not state.get("needs_retrieval"):
            return self._coverage_pass("question_does_not_need_retrieval")

        profile = self._coverage_question_profile(question)
        trace = list(state.get("evidence_quality_trace") or [])
        judgments = list(state.get("evidence_judgments") or [])
        quality = str(state.get("evidence_quality") or "")
        if not quality:
            quality = self._coverage_fallback_quality(question=question, evidence=evidence)

        metrics = self._coverage_metrics(
            question=question,
            evidence=evidence,
            prompt_evidence=prompt_evidence,
            trace=trace,
            judgments=judgments,
            quality=quality,
        )

        if profile["reference"] or profile["field_lookup"]:
            return self._coverage_pass("deterministic_local_task", metrics=metrics)

        if not evidence:
            return self._coverage_refuse(
                reason_code="no_evidence_after_judge",
                reason="证据裁判后没有留下可用于回答的原文证据。",
                metrics=metrics,
            )

        multi_coverage = state.get("multi_document_coverage") or {}
        requested_count = int(multi_coverage.get("requested_document_count") or 0)
        covered_count = int(multi_coverage.get("covered_document_count") or 0)
        if profile["multi_document"] and requested_count > 1 and covered_count < requested_count:
            missing_names = multi_coverage.get("missing_document_names") or []
            suffix = f"缺少：{', '.join(str(name) for name in missing_names[:4])}" if missing_names else ""
            return self._coverage_refuse(
                reason_code="multi_document_coverage_incomplete",
                reason=f"这是多文档问题，但当前证据只覆盖 {covered_count}/{requested_count} 份文档。{suffix}".strip(),
                metrics=metrics,
            )

        if profile["visual"] and not metrics["has_image_evidence"]:
            return self._coverage_refuse(
                reason_code="missing_visual_evidence",
                reason="问题需要图片/图表证据，但最终证据里没有可确认的图片或图表证据。",
                metrics=metrics,
            )

        if profile["ocr"] and not metrics["has_ocr_text"]:
            return self._coverage_refuse(
                reason_code="missing_ocr_text",
                reason="问题需要 OCR/扫描文字证据，但最终证据里没有可确认的 OCR 文本。",
                metrics=metrics,
            )

        if judgments and not metrics["has_direct_or_supporting_judgment"] and not profile["broad_or_document_wide"]:
            return self._coverage_refuse(
                reason_code="no_direct_supporting_judgment",
                reason="证据裁判没有给出 direct 或 supporting 级别的支持，只剩背景性材料。",
                metrics=metrics,
            )

        if quality in {"none", "weak"}:
            return self._coverage_refuse(
                reason_code=f"{quality}_evidence_quality",
                reason=f"当前证据整体质量为 {quality}，不足以支撑硬回答。",
                metrics=metrics,
            )

        if metrics["all_final_labels_weak_or_noise"] and not self._weak_label_set_still_has_usable_support(metrics):
            return self._coverage_refuse(
                reason_code="all_final_evidence_weak_or_noise",
                reason="最终证据的质量标签全部是 weak/noise，没有强支撑证据。",
                metrics=metrics,
            )

        if (
            not profile["broad_or_document_wide"]
            and metrics["final_evidence_count"] == 1
            and quality != "strong"
            and metrics["best_relevance"] < 0.18
        ):
            return self._coverage_refuse(
                reason_code="single_non_strong_evidence",
                reason="最终只剩一条非强证据，且与问题的直接相关度不足。",
                metrics=metrics,
            )

        missing_direct_checker = getattr(self, "_should_decline_for_missing_direct_evidence", None)
        if callable(missing_direct_checker) and missing_direct_checker(question, evidence):
            return self._coverage_refuse(
                reason_code="missing_requested_direct_evidence",
                reason="问题要求的关键事实没有在最终证据中直接出现。",
                metrics=metrics,
            )

        return self._coverage_pass("coverage_sufficient", metrics=metrics)

    def _build_evidence_coverage_refusal_answer(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        decision: dict[str, Any],
    ) -> str:
        reason = str(decision.get("reason") or "当前证据覆盖不足。")
        metrics = dict(decision.get("metrics") or {})
        candidate_count = int(metrics.get("candidate_count") or 0)
        rejected_count = int(metrics.get("judge_rejected_count") or 0)
        filtered_count = int(metrics.get("filtered_count") or 0)
        details = [
            f"召回候选 {candidate_count} 条",
            f"过滤 {filtered_count} 条",
            f"裁判拒绝 {rejected_count} 条",
            f"最终可用 {len(evidence)} 条",
        ]
        citations = self._join_citations([item.citation_id for item in evidence[:2] if item.citation_id])
        citation_text = f"\n\n可参考的相近证据：{citations}" if citations else ""
        return (
            f"我不能可靠回答这个问题，因为当前证据覆盖不足：{reason}\n\n"
            f"本轮证据检查结果：{'；'.join(details)}。"
            "为了避免把弱相关片段硬编成答案，这轮我选择拒答。\n\n"
            "下一步可以换一个更贴近原文措辞的问题，或重新准备/索引文档后再问。"
            f"{citation_text}"
        )

    def _coverage_question_profile(self, question: str) -> dict[str, bool]:
        profile = {
            "reference": self._call_bool("_looks_like_reference_question", question),
            "field_lookup": self._call_bool("_looks_like_field_lookup_question", question),
            "compare": self._call_bool("_looks_like_compare_question", question),
            "document_wide": self._call_bool("_looks_like_document_wide_question", question),
            "broad": self._call_bool("_looks_like_broad_overview_question", question),
            "multi_topic": self._call_bool("_looks_like_multi_document_topic_question", question),
            "visual": self._call_bool("_looks_like_visual_retrieval_question", question),
            "ocr": self._call_bool("_looks_like_ocr_question", question),
        }
        profile["multi_document"] = profile["compare"] or profile["multi_topic"]
        profile["broad_or_document_wide"] = profile["broad"] or profile["document_wide"] or profile["compare"]
        return profile

    def _coverage_metrics(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        prompt_evidence: list[EvidenceItem],
        trace: list[dict[str, Any]],
        judgments: list[dict[str, Any]],
        quality: str,
    ) -> dict[str, Any]:
        relevance_scores = [self._question_relevance_score(question, item.text) for item in evidence]
        labels = [str(item.quality_label or "") for item in evidence if str(item.quality_label or "")]
        if not labels:
            labels = [
                str(row.get("quality_label") or "")
                for row in trace
                if str(row.get("selection_status") or "") in {"selected_for_answer", "selected_by_retrieval_filter"}
            ]
        return {
            "evidence_quality": quality,
            "candidate_count": len(trace) or len(evidence),
            "final_evidence_count": len(evidence),
            "prompt_evidence_count": len(prompt_evidence),
            "filtered_count": sum(1 for row in trace if row.get("selection_status") == "filtered_out"),
            "judge_rejected_count": sum(
                1 for row in trace if row.get("selection_status") == "rejected_by_evidence_judge"
            ),
            "best_score": max((item.score for item in evidence), default=0.0),
            "best_relevance": max(relevance_scores or [0.0]),
            "has_direct_or_supporting_judgment": any(
                str(judgment.get("verdict") or "") in {"direct", "supporting"} for judgment in judgments
            )
            or not judgments,
            "all_final_labels_weak_or_noise": bool(labels) and all(label in {"weak", "noise"} for label in labels),
            "has_image_evidence": any(self._evidence_has_image_signal(item) for item in evidence),
            "has_ocr_text": any(self._evidence_has_ocr_text_signal(item) for item in evidence),
        }

    def _coverage_fallback_quality(self, *, question: str, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return "none"
        best_score = max((item.score for item in evidence), default=0.0)
        best_relevance = max((self._question_relevance_score(question, item.text) for item in evidence), default=0.0)
        if best_score >= 0.7 or best_relevance >= 0.25:
            return "strong"
        if best_score >= 0.45 or best_relevance >= 0.12:
            return "medium"
        return "weak"

    def _coverage_pass(self, reason_code: str, *, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "should_refuse": False,
            "reason_code": reason_code,
            "reason": "",
            "metrics": metrics or {},
        }

    def _coverage_refuse(self, *, reason_code: str, reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "should_refuse": True,
            "reason_code": reason_code,
            "reason": reason,
            "metrics": metrics,
        }

    def _weak_label_set_still_has_usable_support(self, metrics: dict[str, Any]) -> bool:
        if str(metrics.get("evidence_quality") or "") != "strong":
            return False
        if not metrics.get("has_direct_or_supporting_judgment"):
            return False
        if int(metrics.get("prompt_evidence_count") or 0) <= 0:
            return False
        return float(metrics.get("best_relevance") or 0.0) >= 0.24

    def _evidence_has_image_signal(self, item: EvidenceItem) -> bool:
        chunk_type = (item.chunk_type or "").lower()
        return bool(item.image_id or item.image_path) or any(
            marker in chunk_type for marker in ["image", "figure", "chart"]
        )

    def _evidence_has_ocr_text_signal(self, item: EvidenceItem) -> bool:
        if any(str(getattr(image, "ocr_text", "") or "").strip() for image in item.related_images):
            return True
        if any(
            self._visual_summary_has_text_recognition_signal(str(getattr(image, "vision_summary", "") or ""))
            for image in item.related_images
        ):
            return True
        text = str(item.text or "")
        if "OCR:" in text:
            return bool(text.split("OCR:", 1)[1].strip())
        if self._visual_summary_has_text_recognition_signal(f"{item.quote or ''}\n{text}"):
            return True
        return "ocr" in (item.chunk_type or "").lower() and bool(text.strip())

    def _visual_summary_has_text_recognition_signal(self, text: str) -> bool:
        normalized = str(text or "").lower()
        if len(normalized.strip()) < 12:
            return False
        markers = [
            "文字识别",
            "文字清晰",
            "文字均",
            "可识别",
            "图片内文字",
            "文本页面",
            "说明文本",
            "产品说明",
            "ocr",
            "readable text",
            "scanned text",
        ]
        return any(marker in normalized for marker in markers)

    def _call_bool(self, method_name: str, question: str) -> bool:
        method = getattr(self, method_name, None)
        if method is None:
            return False
        try:
            return bool(method(question))
        except TypeError:
            return False
