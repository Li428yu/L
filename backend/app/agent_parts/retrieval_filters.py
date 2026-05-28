from __future__ import annotations

import re

from backend.app.models import EvidenceItem


class AgentRetrievalFilterMixin:
    def _filter_evidence_for_question(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
        target_document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []

        reference_question = self._looks_like_reference_question(question)
        field_lookup_question = self._looks_like_field_lookup_question(question)
        allow_tables = self._looks_like_table_question(question)
        unique_documents = [
            document_id
            for document_id in dict.fromkeys(item.document_id for item in evidence if item.document_id)
        ]
        coverage_document_ids = self._coverage_target_document_ids(
            target_document_ids=target_document_ids,
            candidate_document_ids=unique_documents,
        )
        if reference_question:
            selected: list[EvidenceItem] = []
            seen_references: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_references:
                    continue
                seen_references.add(item.chunk_id)
                item.quote = self._best_reference_quote(item.text)
                selected.append(item)
                if len(selected) >= max(2, min(top_k + 1, 6)):
                    break
            selected = self._repair_final_evidence_selection(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(2, min(top_k + 1, 6)),
            )
            return self._finalize_filtered_evidence(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(2, min(top_k + 1, 6)),
            )

        if self._looks_like_broad_overview_question(question):
            if len(unique_documents) > 1:
                selected = self._select_document_balanced_evidence(
                    question=question,
                    evidence=evidence,
                    limit=max(top_k, min(len(evidence), len(unique_documents) * 2)),
                    target_document_ids=coverage_document_ids,
                )
            else:
                selected = self._select_diverse_overview_evidence(question=question, evidence=evidence, limit=3)
            selected = self._repair_final_evidence_selection(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(3, min(top_k, len(evidence))),
            )
            selected = self._ensure_multi_document_coverage_if_needed(
                question=question,
                selected=selected,
                evidence=evidence,
                target_document_ids=coverage_document_ids,
                limit=max(3, min(top_k, len(evidence))),
            )
            return self._finalize_filtered_evidence(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(3, min(top_k, len(evidence))),
            )

        if field_lookup_question:
            selected: list[EvidenceItem] = []
            seen_fields: set[str] = set()
            field_evidence = [item for item in evidence if self._is_field_evidence_item(item)]
            source_evidence = field_evidence or evidence
            for item in source_evidence:
                key = f"{item.document_id}:{item.quote or item.text}"
                if key in seen_fields:
                    continue
                seen_fields.add(key)
                item.quote = self._sanitize_evidence_text(item.quote or item.text)
                selected.append(item)
            selected = self._repair_final_evidence_selection(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(len(selected), top_k),
            )
            return self._finalize_filtered_evidence(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=max(len(selected), top_k),
            )

        if allow_tables:
            table_evidence = [
                item
                for item in evidence
                if "table" in (item.chunk_type or "").lower()
                or self._is_table_like_text(self._sanitize_evidence_text(item.text))
            ]
            if table_evidence:
                focused_tables = [
                    item
                    for item in table_evidence
                    if any(marker in self._sanitize_evidence_text(item.text) for marker in ["实验分数构成", "实验过程", "实验结果", "实验总分"])
                ] or table_evidence
                scored_tables = sorted(
                    [
                        (
                            item.score
                            + self._question_relevance_score(question, f"{item.paper_name}\n{item.text}") * 0.75
                            + self._metric_table_bonus(question, item.text),
                            position,
                            item,
                        )
                        for position, item in enumerate(focused_tables)
                    ],
                    key=lambda row: (row[0], -row[1]),
                    reverse=True,
                )
                selected = []
                seen_chunks: set[str] = set()
                for _, _, item in scored_tables:
                    if item.chunk_id in seen_chunks:
                        continue
                    if self._looks_like_metric_result_question(question):
                        item.quote = self._keyword_focused_quote(
                            text=self._sanitize_evidence_text(item.text),
                            terms=[
                                "table 1",
                                "table 2",
                                "table 3",
                                "benchmark",
                                "metric",
                                "score",
                                "result",
                                "accuracy",
                                "error",
                                "f1",
                                "auc",
                                "zero-shot",
                                "perplexity",
                            ],
                            bonus_phrases=[
                                "state-of-the-art",
                                "outperforms",
                                "experimental results",
                                "benchmark results",
                                "results are shown",
                            ],
                            limit=900,
                        )
                    else:
                        item.quote = self._truncate_readable_text(self._sanitize_evidence_text(item.text), limit=320)
                    selected.append(item)
                    seen_chunks.add(item.chunk_id)
                    if len(selected) >= top_k:
                        break
                selected = self._repair_final_evidence_selection(
                    question=question,
                    selected=selected,
                    candidates=evidence,
                    limit=top_k,
                )
                selected = self._ensure_multi_document_coverage_if_needed(
                    question=question,
                    selected=selected,
                    evidence=evidence,
                    target_document_ids=coverage_document_ids,
                    limit=top_k,
                )
                return self._finalize_filtered_evidence(
                    question=question,
                    selected=selected,
                    candidates=evidence,
                    limit=top_k,
                )

        if self._looks_like_framework_function_question(question):
            selected = self._select_framework_function_evidence(
                question=question,
                evidence=evidence,
                limit=top_k,
            )
            if selected:
                selected = self._ensure_selected_document_coverage(
                    question=question,
                    selected=selected,
                    evidence=evidence,
                    target_document_ids=coverage_document_ids,
                    limit=top_k,
                )
                selected = self._repair_final_evidence_selection(
                    question=question,
                    selected=selected,
                    candidates=evidence,
                    limit=top_k,
                )
                selected = self._ensure_multi_document_coverage_if_needed(
                    question=question,
                    selected=selected,
                    evidence=evidence,
                    target_document_ids=coverage_document_ids,
                    limit=top_k,
                )
                return self._finalize_filtered_evidence(
                    question=question,
                    selected=selected,
                    candidates=evidence,
                    limit=top_k,
                )

        multi_document_question = len(unique_documents) > 1 and (
            self._looks_like_compare_question(question)
            or self._looks_like_multi_document_topic_question(question)
        )
        keyword_limit = min(top_k, 2) if "abstract" in question.lower() else top_k
        keyword_selected = [] if multi_document_question else self._select_special_keyword_evidence(
            question=question,
            evidence=evidence,
            limit=keyword_limit,
        )
        if keyword_selected:
            keyword_selected = self._ensure_selected_document_coverage(
                question=question,
                selected=keyword_selected,
                evidence=evidence,
                target_document_ids=coverage_document_ids,
                limit=keyword_limit,
            )
            keyword_selected = self._repair_final_evidence_selection(
                question=question,
                selected=keyword_selected,
                candidates=evidence,
                limit=keyword_limit,
            )
            keyword_selected = self._ensure_multi_document_coverage_if_needed(
                question=question,
                selected=keyword_selected,
                evidence=evidence,
                target_document_ids=coverage_document_ids,
                limit=keyword_limit,
            )
            return self._finalize_filtered_evidence(
                question=question,
                selected=keyword_selected,
                candidates=evidence,
                limit=keyword_limit,
            )

        if len(unique_documents) > 1 and (
            self._looks_like_compare_question(question)
            or self._looks_like_multi_document_topic_question(question)
        ):
            limit = max(top_k, min(len(evidence), len(unique_documents) * 2))
            selected = self._select_document_balanced_evidence(
                question=question,
                evidence=evidence,
                limit=limit,
                target_document_ids=coverage_document_ids,
            )
            selected = self._repair_final_evidence_selection(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=limit,
            )
            selected = self._ensure_comparative_role_coverage_if_needed(
                question=question,
                selected=selected,
                evidence=evidence,
                target_document_ids=coverage_document_ids,
                limit=limit,
            )
            selected = self._ensure_multi_document_coverage_if_needed(
                question=question,
                selected=selected,
                evidence=evidence,
                target_document_ids=coverage_document_ids,
                limit=limit,
            )
            return self._finalize_filtered_evidence(
                question=question,
                selected=selected,
                candidates=evidence,
                limit=limit,
            )

        seen: set[str] = set()
        scored: list[tuple[float, int, EvidenceItem]] = []
        fallback: list[EvidenceItem] = []

        for position, item in enumerate(evidence):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            item.quote = self._best_readable_quote(item.quote or item.text)
            text = self._sanitize_evidence_text(item.text)
            quality = self._readable_text_score(text)
            relevance = self._question_relevance_score(question, text)
            table_like = self._is_table_like_text(text)
            if table_like and not allow_tables and quality < 0.45:
                fallback.append(item)
                continue

            adjusted_score = (
                item.score
                + quality * 0.18
                + relevance * 0.75
                + self._scientific_evidence_shape_bonus(question=question, item=item, text=text) * 0.55
            )
            scored.append((adjusted_score, position, item))

        target_count = max(1, top_k)
        if not scored:
            selected = fallback[:target_count] or evidence[:target_count]
        else:
            scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            selected = [item for _, _, item in scored[:target_count]]

        selected = self._repair_final_evidence_selection(
            question=question,
            selected=selected,
            candidates=evidence,
            limit=target_count,
        )
        selected = self._ensure_multi_document_coverage_if_needed(
            question=question,
            selected=selected,
            evidence=evidence,
            target_document_ids=coverage_document_ids,
            limit=target_count,
        )
        return self._finalize_filtered_evidence(
            question=question,
            selected=selected,
            candidates=evidence,
            limit=target_count,
        )

    def _finalize_filtered_evidence(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        candidates: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        stabilizer = getattr(self, "_stabilize_final_evidence_citations", None)
        if callable(stabilizer):
            selected = stabilizer(
                question=question,
                selected=selected,
                candidates=candidates,
                limit=limit,
            )
        return self._renumber_evidence(selected[: max(1, limit)])

    def _repair_final_evidence_selection(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        candidates: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        """Keep direct support evidence from being displaced by modality or balance rules."""
        if not candidates:
            return selected[: max(1, limit)]
        limit = max(1, limit)
        selected = list(selected)
        selected_chunks = {item.chunk_id for item in selected}
        visual_question = self._question_requires_visual_evidence(question)

        scored_direct: list[tuple[float, int, EvidenceItem]] = []
        seen_candidates: set[str] = set()
        for position, item in enumerate(candidates):
            if item.chunk_id in seen_candidates:
                continue
            seen_candidates.add(item.chunk_id)
            score = self._final_evidence_direct_support_score(question=question, item=item)
            if score < 1.35:
                continue
            scored_direct.append((score, position, item))

        scored_direct.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        max_repairs = (
            2
            if (
                self._looks_like_metric_result_question(question)
                or self._looks_like_framework_function_question(question)
                or self._looks_like_trustworthy_characteristics_question(question)
                or self._looks_like_efficiency_or_resource_question(question)
                or self._looks_like_method_adaptation_question(question)
                or self._looks_like_design_or_training_factor_question(question)
            )
            else 1
        )
        repairs = 0
        for score, _, item in scored_direct:
            if item.chunk_id in selected_chunks:
                self._set_final_support_quote(question=question, item=item)
                continue
            if repairs >= max_repairs:
                break
            if not visual_question and self._is_visual_evidence_item(item):
                continue
            selected = self._insert_or_replace_final_evidence(
                question=question,
                selected=selected,
                item=item,
                item_score=score,
                limit=limit,
            )
            selected_chunks = {candidate.chunk_id for candidate in selected}
            if item.chunk_id in selected_chunks:
                repairs += 1

        if visual_question and not any(self._is_visual_evidence_item(item) for item in selected):
            visual_candidates: list[tuple[float, int, EvidenceItem]] = []
            for position, item in enumerate(candidates):
                if item.chunk_id in selected_chunks or not self._is_visual_evidence_item(item):
                    continue
                text = self._sanitize_evidence_text(f"{item.paper_name}\n{item.section or ''}\n{item.quote}\n{item.text}")
                support_score = self._final_evidence_direct_support_score(question=question, item=item)
                relevance = self._question_relevance_score(question, text)
                visual_score = support_score + relevance + item.score * 0.25
                if visual_score >= 1.0 or relevance >= 0.18:
                    visual_candidates.append((visual_score, position, item))
            visual_candidates.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            if visual_candidates:
                _, _, visual_item = visual_candidates[0]
                self._set_final_support_quote(question=question, item=visual_item)
                selected = [item for item in selected if item.chunk_id != visual_item.chunk_id]
                if len(selected) >= limit:
                    replacement_scores = [
                        (
                            self._final_evidence_direct_support_score(question=question, item=item),
                            position,
                        )
                        for position, item in enumerate(selected)
                        if not self._is_visual_evidence_item(item)
                    ]
                    if replacement_scores:
                        _, drop_index = min(replacement_scores, key=lambda row: (row[0], -row[1]))
                        selected.pop(drop_index)
                    else:
                        selected = selected[: max(0, limit - 1)]
                selected.insert(0, visual_item)
                selected_chunks = {candidate.chunk_id for candidate in selected}

        if visual_question:
            visual_selected = [item for item in selected if self._is_visual_evidence_item(item)]
            if visual_selected:
                visual_selected.sort(
                    key=lambda item: (
                        self._final_evidence_direct_support_score(question=question, item=item),
                        self._question_relevance_score(question, self._sanitize_evidence_text(item.text)),
                        item.score,
                    ),
                    reverse=True,
                )
                lead_visual = visual_selected[0]
                selected = [lead_visual, *[item for item in selected if item.chunk_id != lead_visual.chunk_id]]

        return selected[:limit]

    def _insert_or_replace_final_evidence(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        item: EvidenceItem,
        item_score: float,
        limit: int,
    ) -> list[EvidenceItem]:
        self._set_final_support_quote(question=question, item=item)
        result = [candidate for candidate in selected if candidate.chunk_id != item.chunk_id]
        if len(result) < limit:
            return [item, *result]

        visual_question = self._question_requires_visual_evidence(question)
        replacement_scores = [
            (
                self._final_evidence_direct_support_score(question=question, item=candidate)
                - (0.45 if self._is_visual_evidence_item(candidate) and not visual_question else 0.0),
                position,
            )
            for position, candidate in enumerate(result)
        ]
        replacement_scores.sort(key=lambda row: (row[0], -row[1]))
        worst_score, worst_index = replacement_scores[0]
        if item_score <= worst_score + 0.15:
            return result[:limit]
        result[worst_index] = item
        return result[:limit]

    def _set_final_support_quote(self, *, question: str, item: EvidenceItem) -> None:
        text = self._sanitize_evidence_text(item.text)
        if self._looks_like_metric_result_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "table",
                    "benchmark",
                    "metric",
                    "score",
                    "result",
                    "accuracy",
                    "error",
                    "f1",
                    "auc",
                    "state-of-the-art",
                ],
                bonus_phrases=[
                    "outperforms",
                    "state-of-the-art",
                    "experimental results",
                    "benchmark results",
                ],
                limit=620,
            )
            return
        if self._looks_like_framework_function_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "function",
                    "functions",
                    "core",
                    "component",
                    "components",
                    "capability",
                    "capabilities",
                    "stage",
                    "process",
                    "role",
                    "strategy",
                    "policy",
                    "outcome",
                    "outcomes",
                ],
                bonus_phrases=[
                    "core functions",
                    "composed of",
                    "consists of",
                    "organized into",
                    "functions organize",
                ],
                limit=620,
            )
            return
        if self._looks_like_trustworthy_characteristics_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "trustworthy",
                    "characteristics",
                    "valid and reliable",
                    "safe",
                    "secure and resilient",
                    "accountable",
                    "transparent",
                    "explainable",
                    "interpretable",
                    "privacy",
                    "fair",
                    "harmful bias",
                ],
                bonus_phrases=["characteristics of trustworthy ai systems include", "valid and reliable"],
                limit=620,
            )
            return
        if self._looks_like_efficiency_or_resource_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "trainable parameter",
                    "trainable parameters",
                    "number of trainable",
                    "parameters",
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
                    "reduce",
                    "reduced",
                    "times",
                    "no additional",
                ],
                bonus_phrases=[
                    "reduce the number",
                    "memory requirement",
                    "higher training throughput",
                    "no additional",
                    "fewer trainable parameters",
                ],
                limit=720,
            )
            return
        if self._looks_like_method_adaptation_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "adaptation",
                    "fine-tuning",
                    "pre-trained",
                    "pretrained",
                    "weights",
                    "frozen",
                    "freeze",
                    "rank",
                    "low-rank",
                    "decomposition",
                    "matrix",
                    "matrices",
                    "downstream",
                    "update",
                    "inject",
                ],
                bonus_phrases=["we propose", "we introduce", "our approach", "pre-trained weights", "rank decomposition"],
                limit=720,
            )
            return
        if self._looks_like_design_or_training_factor_question(question):
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "design",
                    "training",
                    "factor",
                    "factors",
                    "component",
                    "components",
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
                    "ablation",
                    "study",
                ],
                bonus_phrases=["critical role", "systematically study", "we show", "we find", "training steps"],
                limit=720,
            )
            return
        item.quote = self._best_quote_for_question(question, text)

    def _final_evidence_direct_support_score(self, *, question: str, item: EvidenceItem) -> float:
        text = self._sanitize_evidence_text(f"{item.section or ''}\n{item.quote}\n{item.text}")
        if not text.strip():
            return 0.0
        score = item.score + self._question_relevance_score(question, text) * 0.8
        if self._is_table_like_text(text) and not self._looks_like_table_question(question):
            score -= 0.35
        if self._is_visual_evidence_item(item) and not self._question_requires_visual_evidence(question):
            score -= 0.55
        else:
            score += 0.10
        framework_bonus = self._framework_function_support_bonus(question=question, text=text, item=item)
        if self._looks_like_framework_function_question(question) and framework_bonus <= 0.0:
            return 0.0
        score += framework_bonus
        score += self._trustworthy_characteristics_support_bonus(question=question, text=text)
        score += self._metric_result_support_bonus(question=question, text=text, item=item)
        score += self._scientific_paper_support_bonus(question=question, text=text)
        score += self._scientific_evidence_shape_bonus(question=question, item=item, text=text)
        return score

    def _scientific_paper_support_bonus(self, *, question: str, text: str) -> float:
        normalized_question = question.lower()
        normalized_text = text.lower()
        score = 0.0
        if any(term in question for term in ["概括", "核心贡献", "主要贡献"]) or "summary" in normalized_question:
            if any(phrase in normalized_text for phrase in ["we propose", "we present", "we introduce", "contribution"]):
                score += 0.9
            if any(phrase in normalized_text for phrase in ["abstract", "conclusion", "our approach", "this paper"]):
                score += 0.5
        if any(term in question for term in ["方法", "机制", "结构", "架构", "组成"]) or any(
            term in normalized_question for term in ["method", "architecture"]
        ):
            if any(
                phrase in normalized_text
                for phrase in [
                    "model architecture",
                    "consists of",
                    "objective",
                    "method",
                    "framework",
                ]
            ):
                score += 0.7
        if any(term in question for term in ["是否证明", "能否证明", "证明了"]) or "prove" in normalized_question:
            if any(phrase in normalized_text for phrase in ["experiment", "evaluation", "task", "dataset", "domain"]):
                score += 0.4
        return score

    def _scientific_evidence_shape_bonus(self, *, question: str, item: EvidenceItem, text: str) -> float:
        normalized_question = question.lower()
        normalized_text = self._sanitize_evidence_text(text).lower()
        section = (item.section or "").lower()
        if not normalized_text:
            return 0.0

        score = 0.0
        if "abstract" in normalized_question:
            if "abstract" in section or " abstract " in f" {normalized_text[:900]} ":
                score += 1.15
            elif "introduction" in section:
                score += 0.35

        if self._looks_like_efficiency_or_resource_question(question):
            resource_terms = [
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
            ]
            hits = sum(1 for term in resource_terms if term in normalized_text)
            score += min(1.05, hits * 0.16)
            if re.search(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:x|times|%|m|b|k|million|billion)?\b", normalized_text):
                score += 0.25
            if any(phrase in normalized_text for phrase in ["we propose", "we present", "we introduce", "we show"]):
                score += 0.2

        if self._looks_like_method_adaptation_question(question):
            method_terms = [
                "adaptation",
                "fine-tuning",
                "pre-trained",
                "pretrained",
                "weights",
                "frozen",
                "freeze",
                "rank",
                "decomposition",
                "matrix",
                "matrices",
                "downstream",
                "update",
                "inject",
            ]
            hits = sum(1 for term in method_terms if term in normalized_text)
            score += min(1.0, hits * 0.13)
            if any(phrase in normalized_text for phrase in ["we propose", "we present", "we introduce", "our approach"]):
                score += 0.35
            if any(marker in section for marker in ["abstract", "introduction", "method"]):
                score += 0.25

        if self._looks_like_design_or_training_factor_question(question):
            factor_terms = [
                "design",
                "training",
                "factor",
                "factors",
                "component",
                "components",
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
                "ablation",
                "study",
            ]
            hits = sum(1 for term in factor_terms if term in normalized_text)
            score += min(1.15, hits * 0.12)
            if any(phrase in normalized_text for phrase in ["we show", "we find", "systematically study", "critical role"]):
                score += 0.3
            if any(marker in section for marker in ["abstract", "introduction", "result", "method"]):
                score += 0.2

        if "references" in section:
            score -= 0.6
        if "appendix" in section or "additional experiment" in normalized_text[:240]:
            score -= 0.18
        if self._is_table_evidence_item(item) and not self._looks_like_table_question(question):
            score -= 0.18
        if self._is_visual_evidence_item(item) and not self._question_requires_visual_evidence(question):
            score -= 0.3
        return score

    def _framework_function_support_bonus(self, *, question: str, text: str, item: EvidenceItem) -> float:
        if not self._looks_like_framework_function_question(question):
            return 0.0
        normalized = text.lower()
        hits = {
            term
            for term in [
                "function",
                "functions",
                "core",
                "component",
                "components",
                "capability",
                "capabilities",
                "stage",
                "stages",
                "process",
                "processes",
                "role",
                "roles",
                "policy",
                "strategy",
                "outcome",
                "outcomes",
            ]
            if re.search(rf"\b{re.escape(term)}\b", normalized)
        }
        if len(hits) < 2:
            return 0.0
        bonus = min(1.0, len(hits) * 0.14)
        if any(
            phrase in normalized
            for phrase in [
                "core functions",
                "composed of",
                "consists of",
                "organized into",
                "functions organize",
            ]
        ):
            bonus += 0.55
        role_markers = ["strategy", "policy", "organizational", "outcome", "outcomes", "responsibility", "scope"]
        role_hits = sum(1 for marker in role_markers if marker in normalized)
        bonus += min(0.8, role_hits * 0.16)
        if self._is_visual_evidence_item(item) and not self._question_requires_visual_evidence(question):
            bonus -= 0.45
        return bonus

    def _trustworthy_characteristics_support_bonus(self, *, question: str, text: str) -> float:
        if not self._looks_like_trustworthy_characteristics_question(question):
            return 0.0
        normalized = text.lower()
        terms = [
            "valid and reliable",
            "safe",
            "secure and resilient",
            "accountable",
            "transparent",
            "explainable",
            "interpretable",
            "privacy",
            "fair",
            "harmful bias",
        ]
        hits = sum(1 for term in terms if term in normalized)
        bonus = min(1.35, hits * 0.18)
        if "characteristics of trustworthy ai systems include" in normalized:
            bonus += 0.9
        return bonus

    def _metric_result_support_bonus(self, *, question: str, text: str, item: EvidenceItem) -> float:
        if not self._looks_like_metric_result_question(question):
            return 0.0
        normalized = text.lower()
        terms = [
            "benchmark",
            "metric",
            "score",
            "result",
            "accuracy",
            "error",
            "f1",
            "auc",
            "table",
            "state-of-the-art",
        ]
        hits = sum(1 for term in terms if term in normalized)
        bonus = min(1.3, hits * 0.16)
        if "outperforms" in normalized or "state-of-the-art" in normalized:
            bonus += 0.45
        if "table" in (item.chunk_type or "").lower():
            bonus += 0.25
        return bonus

    def _question_requires_visual_evidence(self, question: str) -> bool:
        visual_checker = getattr(self, "_looks_like_visual_evidence_question", None)
        if callable(visual_checker) and visual_checker(question):
            return True
        retrieval_checker = getattr(self, "_looks_like_visual_retrieval_question", None)
        return bool(callable(retrieval_checker) and retrieval_checker(question))

    def _is_visual_evidence_item(self, item: EvidenceItem) -> bool:
        chunk_type = (item.chunk_type or "").lower()
        return bool(item.image_id) or any(marker in chunk_type for marker in ["image", "figure", "chart"])

    def _is_table_evidence_item(self, item: EvidenceItem) -> bool:
        chunk_type = (item.chunk_type or "").lower()
        text = self._sanitize_evidence_text(item.text)
        return "table" in chunk_type or self._is_table_like_text(text)

    def _metric_table_bonus(self, question: str, text: str) -> float:
        if not self._looks_like_metric_result_question(question):
            return 0.0
        normalized = self._sanitize_evidence_text(text).lower()
        terms = [
            "table 1",
            "table 2",
            "table 3",
            "benchmark",
            "metric",
            "score",
            "result",
            "zero-shot",
            "accuracy",
            "error",
            "f1",
            "auc",
            "perplexity",
        ]
        hits = sum(1 for term in terms if term in normalized)
        bonus = min(2.2, hits * 0.22)
        if "table" in normalized and any(term in normalized for term in ["score", "accuracy", "error", "result"]):
            bonus += 1.0
        return bonus

    def _select_framework_function_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        scored: list[tuple[float, int, EvidenceItem]] = []
        expected_terms = self._expected_framework_function_terms(question)
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            normalized = text.lower()
            hits = {
                term
                for term in [
                    "function",
                    "functions",
                    "core",
                    "component",
                    "components",
                    "capability",
                    "capabilities",
                    "stage",
                    "stages",
                    "process",
                    "processes",
                    "role",
                    "roles",
                    "policy",
                    "strategy",
                    "outcome",
                    "outcomes",
                ]
                if re.search(rf"\b{re.escape(term)}\b", normalized)
            }
            min_hits = 1 if expected_terms and len(expected_terms) == 1 else 2
            if len(hits) < min_hits:
                continue
            if expected_terms and not expected_terms.issubset(hits):
                continue
            score = item.score + min(0.8, len(hits) * 0.12)
            if "core is composed" in normalized or "functions organize" in normalized or "consists of" in normalized:
                score += 0.7
            if "function" in normalized or "functions" in normalized:
                score += 0.25
            if any(marker in (item.chunk_type or "").lower() for marker in ["image", "figure", "table"]):
                score += 0.25 if self._question_requires_visual_evidence(question) else -0.35
            if expected_terms and len(expected_terms) == 1:
                role_markers = [
                    "strategy",
                    "policy",
                    "organizational",
                    "outcome",
                    "outcomes",
                    "responsibility",
                    "scope",
                ]
                score += min(1.1, sum(1 for marker in role_markers if marker in normalized) * 0.18)
            score += self._question_relevance_score(question, f"{item.paper_name}\n{text}") * 0.4
            scored.append((score, position, item))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        selected: list[EvidenceItem] = []
        seen: set[str] = set()
        for _, _, item in scored:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            text = self._sanitize_evidence_text(item.text)
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "function",
                    "functions",
                    "component",
                    "components",
                    "capability",
                    "capabilities",
                    "stage",
                    "process",
                    "role",
                    "strategy",
                    "policy",
                    "core",
                    "outcome",
                    "outcomes",
                ],
                bonus_phrases=[
                    "core is composed",
                    "functions organize",
                    "core functions",
                    "composed of",
                    "consists of",
                    "organized into",
                ],
                limit=520,
            )
            selected.append(item)
            if len(selected) >= max(1, limit):
                break
        return selected

    def _expected_framework_function_terms(self, question: str) -> set[str]:
        # Keep framework-function retrieval generic; document-specific function names
        # should come from retrieved text, not from runtime expectations.
        return set()

    def _coverage_target_document_ids(
        self,
        *,
        target_document_ids: list[str] | None,
        candidate_document_ids: list[str],
    ) -> list[str]:
        candidate_ids = list(dict.fromkeys(document_id for document_id in candidate_document_ids if document_id))
        if not target_document_ids:
            return candidate_ids
        candidate_set = set(candidate_ids)
        scoped_ids = [
            document_id
            for document_id in dict.fromkeys(target_document_ids)
            if document_id and document_id in candidate_set
        ]
        return scoped_ids or candidate_ids

    def _ensure_multi_document_coverage_if_needed(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        evidence: list[EvidenceItem],
        target_document_ids: list[str],
        limit: int,
    ) -> list[EvidenceItem]:
        if not self._should_enforce_multi_document_coverage(
            question=question,
            target_document_ids=target_document_ids,
            limit=limit,
        ):
            return selected[:limit]
        return self._ensure_selected_document_coverage(
            question=question,
            selected=selected,
            evidence=evidence,
            target_document_ids=target_document_ids,
            limit=limit,
        )

    def _should_enforce_multi_document_coverage(
        self,
        *,
        question: str,
        target_document_ids: list[str],
        limit: int,
    ) -> bool:
        target_count = len([document_id for document_id in target_document_ids if document_id])
        if target_count <= 1 or target_count > max(4, limit):
            return False
        return (
            self._looks_like_compare_question(question)
            or self._looks_like_multi_document_topic_question(question)
            or self._looks_like_broad_overview_question(question)
        )

    def _ensure_comparative_role_coverage_if_needed(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        evidence: list[EvidenceItem],
        target_document_ids: list[str],
        limit: int,
    ) -> list[EvidenceItem]:
        roles = self._comparative_balance_roles(question=question, document_count=len(target_document_ids), limit=limit)
        if not roles or len(target_document_ids) <= 1:
            return selected[:limit]

        result = list(selected[:limit])
        selected_chunks = {item.chunk_id for item in result}
        document_ids = [document_id for document_id in target_document_ids if document_id]
        for role in roles:
            for document_id in document_ids:
                if self._document_has_comparative_role(question=question, items=result, document_id=document_id, role=role):
                    continue
                candidate = self._best_comparative_role_candidate(
                    question=question,
                    evidence=evidence,
                    document_id=document_id,
                    role=role,
                    excluded_chunks=selected_chunks,
                )
                if not candidate:
                    continue
                self._set_comparative_role_quote(question=question, item=candidate, role=role)
                if len(result) < limit:
                    result.append(candidate)
                    selected_chunks.add(candidate.chunk_id)
                    continue
                drop_index = self._comparative_role_replacement_index(
                    question=question,
                    selected=result,
                    incoming_document_id=document_id,
                    incoming_role=role,
                    protected_role=roles[0],
                )
                if drop_index is None:
                    continue
                selected_chunks.discard(result[drop_index].chunk_id)
                result[drop_index] = candidate
                selected_chunks.add(candidate.chunk_id)
        return result[:limit]

    def _comparative_balance_roles(self, *, question: str, document_count: int, limit: int) -> list[str]:
        if document_count <= 1 or not (
            self._looks_like_compare_question(question) or self._looks_like_multi_document_topic_question(question)
        ):
            return []
        if self._looks_like_visual_retrieval_question(question):
            return []
        if self._looks_like_metric_result_question(question):
            return ["result"]

        normalized = question.lower()
        mechanism_markers = [
            "how",
            "approach",
            "method",
            "mechanism",
            "strategy",
            "pursue",
            "achieve",
            "via",
            "through",
            "objective",
            "structure",
            "architecture",
            "training",
            "design",
        ]
        asks_for_mechanism = any(marker in normalized for marker in mechanism_markers)
        if self._looks_like_efficiency_or_resource_question(question):
            if limit >= document_count * 2 and asks_for_mechanism:
                return ["mechanism", "efficiency"]
            return ["mechanism"]
        if asks_for_mechanism:
            return ["mechanism"]
        return []

    def _document_has_comparative_role(
        self,
        *,
        question: str,
        items: list[EvidenceItem],
        document_id: str,
        role: str,
    ) -> bool:
        for item in items:
            if item.document_id != document_id:
                continue
            if self._comparative_role_score(question=question, item=item, role=role) >= 1.05:
                return True
        return False

    def _best_comparative_role_candidate(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        document_id: str,
        role: str,
        excluded_chunks: set[str],
    ) -> EvidenceItem | None:
        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            if item.document_id != document_id or item.chunk_id in excluded_chunks:
                continue
            text = self._sanitize_evidence_text(item.text)
            if not text.strip() or self._looks_like_front_matter_noise(text):
                continue
            if self._looks_like_reference_section_text(text):
                continue
            score = self._comparative_role_score(question=question, item=item, role=role)
            if score < 1.05:
                continue
            scored.append((score, position, item))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return scored[0][2] if scored else None

    def _comparative_role_replacement_index(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        incoming_document_id: str,
        incoming_role: str,
        protected_role: str,
    ) -> int | None:
        document_counts: dict[str, int] = {}
        for item in selected:
            if item.document_id:
                document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1

        scored_drops: list[tuple[float, int]] = []
        for index, item in enumerate(selected):
            same_document = item.document_id == incoming_document_id
            keeps_document_coverage = bool(item.document_id and document_counts.get(item.document_id, 0) <= 1)
            protects_primary_role = (
                item.document_id != incoming_document_id
                and self._comparative_role_score(question=question, item=item, role=protected_role) >= 1.05
            )
            if keeps_document_coverage or protects_primary_role:
                continue
            current_role_score = self._comparative_role_score(question=question, item=item, role=incoming_role)
            direct_score = self._final_evidence_direct_support_score(question=question, item=item)
            drop_score = direct_score + current_role_score * 0.4
            if same_document:
                drop_score -= 0.35
            scored_drops.append((drop_score, index))
        if not scored_drops:
            return None
        scored_drops.sort(key=lambda row: (row[0], -row[1]))
        return scored_drops[0][1]

    def _comparative_role_score(self, *, question: str, item: EvidenceItem, role: str) -> float:
        text = self._sanitize_evidence_text(f"{item.section or ''}\n{item.quote}\n{item.text}")
        if not text.strip():
            return 0.0
        normalized = text.lower()
        score = (
            item.score * 0.35
            + self._question_relevance_score(question, text) * 0.8
            + self._readable_text_score(text) * 0.15
            + self._scientific_evidence_shape_bonus(question=question, item=item, text=text) * 0.35
        )
        if self._is_table_evidence_item(item) and role != "result":
            score -= 0.25
        if self._is_visual_evidence_item(item) and not self._question_requires_visual_evidence(question):
            score -= 0.35

        section = (item.section or "").lower()
        if role == "mechanism":
            terms = [
                "method",
                "approach",
                "mechanism",
                "strategy",
                "objective",
                "design",
                "architecture",
                "structure",
                "scale",
                "scaling",
                "coefficient",
                "width",
                "depth",
                "resolution",
                "freeze",
                "frozen",
                "pre-trained",
                "pretrained",
                "weight",
                "weights",
                "rank",
                "decomposition",
                "matrix",
                "matrices",
                "inject",
                "update",
                "adapter",
                "fine-tuning",
                "fine tuning",
                "training objective",
                "loss",
                "cross entropy",
                "contrastive",
                "augmentation",
                "component",
            ]
            hits = sum(1 for term in terms if term in normalized)
            score += min(1.45, hits * 0.16)
            if any(marker in section for marker in ["abstract", "introduction", "method", "approach"]):
                score += 0.25
            if any(phrase in normalized for phrase in ["we propose", "we introduce", "our approach", "we present"]):
                score += 0.28
            if any(marker in section for marker in ["result", "experiment"]) and not any(
                term in normalized for term in ["method", "approach", "design", "architecture", "strategy"]
            ):
                score -= 0.35
        elif role == "efficiency":
            terms = [
                "efficient",
                "efficiency",
                "parameter",
                "parameters",
                "memory",
                "storage",
                "compute",
                "flops",
                "throughput",
                "latency",
                "speedup",
                "faster",
                "smaller",
                "reduce",
                "reduced",
                "fewer",
                "cost",
            ]
            hits = sum(1 for term in terms if term in normalized)
            score += min(1.35, hits * 0.15)
            if re.search(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:x|times|%|m|b|k|million|billion)\b", normalized):
                score += 0.3
        elif role == "result":
            terms = [
                "result",
                "results",
                "benchmark",
                "outperform",
                "outperforms",
                "state-of-the-art",
                "accuracy",
                "error",
                "score",
                "table",
                "evaluation",
            ]
            hits = sum(1 for term in terms if term in normalized)
            score += min(1.35, hits * 0.16)
            if self._is_table_evidence_item(item):
                score += 0.25
        return score

    def _set_comparative_role_quote(self, *, question: str, item: EvidenceItem, role: str) -> None:
        text = self._sanitize_evidence_text(item.text)
        if role == "mechanism":
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "method",
                    "approach",
                    "mechanism",
                    "strategy",
                    "objective",
                    "design",
                    "architecture",
                    "scale",
                    "scaling",
                    "coefficient",
                    "width",
                    "depth",
                    "resolution",
                    "freeze",
                    "frozen",
                    "pre-trained",
                    "pretrained",
                    "weights",
                    "rank",
                    "decomposition",
                    "matrix",
                    "matrices",
                    "inject",
                    "update",
                ],
                bonus_phrases=["we propose", "we introduce", "our approach", "we present"],
                limit=720,
            )
            return
        if role == "efficiency":
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=[
                    "efficient",
                    "efficiency",
                    "parameter",
                    "parameters",
                    "memory",
                    "compute",
                    "throughput",
                    "latency",
                    "faster",
                    "smaller",
                    "reduce",
                    "reduced",
                    "fewer",
                ],
                bonus_phrases=["fewer trainable parameters", "memory requirement", "higher training throughput"],
                limit=720,
            )
            return
        if role == "result":
            item.quote = self._keyword_focused_quote(
                text=text,
                terms=["result", "benchmark", "outperform", "state-of-the-art", "accuracy", "error", "score", "table"],
                bonus_phrases=["experimental results", "benchmark results", "state-of-the-art"],
                limit=720,
            )
            return
        item.quote = self._best_quote_for_question(question, text)

    def _ensure_selected_document_coverage(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        evidence: list[EvidenceItem],
        target_document_ids: list[str],
        limit: int,
    ) -> list[EvidenceItem]:
        if len(target_document_ids) <= 1 or len(target_document_ids) > max(4, limit):
            return selected
        selected_ids = {item.document_id for item in selected if item.document_id}
        missing_ids = [document_id for document_id in target_document_ids if document_id not in selected_ids]
        if not missing_ids:
            return selected[:limit]

        selected_chunks = {item.chunk_id for item in selected}
        additions: list[EvidenceItem] = []
        for document_id in missing_ids:
            candidates = [
                item
                for item in evidence
                if item.document_id == document_id and item.chunk_id not in selected_chunks
            ]
            if not candidates:
                continue
            ranked = sorted(
                [
                    (
                        self._final_evidence_direct_support_score(question=question, item=item)
                        + self._question_relevance_score(question, f"{item.paper_name}\n{item.text}") * 0.8
                        + self._readable_text_score(self._sanitize_evidence_text(item.text)) * 0.15,
                        position,
                        item,
                    )
                    for position, item in enumerate(candidates)
                ],
                key=lambda row: (row[0], -row[1]),
                reverse=True,
            )
            item_score, _, item = ranked[0]
            relevance = self._question_relevance_score(question, f"{item.paper_name}\n{item.text}")
            if item_score < 1.15 and relevance < 0.12:
                continue
            item.quote = self._best_quote_for_question(question, item.text)
            additions.append(item)
            selected_chunks.add(item.chunk_id)

        if not additions:
            return selected[:limit]
        merged = [*selected, *additions]
        if len(merged) <= limit:
            return merged
        target_id_set = set(target_document_ids)
        protected_chunks = {item.chunk_id for item in additions}
        protected_document_ids: set[str] = set()
        for item in selected:
            if item.document_id not in target_id_set or item.document_id in protected_document_ids:
                continue
            protected_chunks.add(item.chunk_id)
            protected_document_ids.add(item.document_id)
        kept = [item for item in merged if item.chunk_id in protected_chunks]
        for item in merged:
            if item.chunk_id in protected_chunks:
                continue
            kept.append(item)
            if len(kept) >= limit:
                break
        return kept[:limit]

    def _looks_like_efficiency_or_resource_question(self, question: str) -> bool:
        normalized = question.lower()
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

    def _looks_like_method_adaptation_question(self, question: str) -> bool:
        normalized = question.lower()
        idea_markers = ["main idea", "core idea", "method idea", "adaptation idea", "approach", "mechanism"]
        adaptation_markers = [
            "adaptation",
            "fine-tuning",
            "pre-trained",
            "pretrained",
            "weights",
            "rank",
            "matrix",
            "matrices",
            "parameter",
        ]
        return any(marker in normalized for marker in idea_markers) and any(
            marker in normalized for marker in adaptation_markers
        )

    def _looks_like_design_or_training_factor_question(self, question: str) -> bool:
        normalized = question.lower()
        topic_markers = ["design", "training", "factor", "factors", "ablation", "component", "components"]
        ask_markers = ["important", "identify", "affect", "effect", "learned representation", "representation"]
        return any(marker in normalized for marker in topic_markers) and any(
            marker in normalized for marker in ask_markers
        )

    def _select_special_keyword_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        specs: list[tuple[bool, list[str], list[str], int]] = [
            (
                self._looks_like_trustworthy_characteristics_question(question),
                [
                    "trustworthy",
                    "valid and reliable",
                    "safe",
                    "secure and resilient",
                    "accountable",
                    "transparent",
                    "explainable",
                    "interpretable",
                    "privacy",
                    "fair",
                    "harmful bias",
                ],
                ["characteristics of trustworthy ai systems include", "fig. 4"],
                3,
            ),
            (
                self._looks_like_pretraining_objective_question(question),
                [
                    "pre-training objective",
                    "training objective",
                    "objective function",
                    "self-supervised",
                    "supervised objective",
                    "pretext task",
                    "loss function",
                ],
                ["pre-training objective", "training objective", "loss function"],
                2,
            ),
            (
                self._looks_like_dataset_or_scale_question(question),
                [
                    "dataset",
                    "training set",
                    "pre-training dataset",
                    "training data",
                    "corpus",
                    "data collection",
                    "data source",
                    "samples",
                    "image-text",
                    "pairs",
                    "million",
                    "billion",
                    "data engine",
                ],
                ["training set", "pre-training dataset", "data collection", "image-text pairs"],
                2,
            ),
            (
                self._looks_like_efficiency_or_resource_question(question),
                [
                    "trainable parameter",
                    "trainable parameters",
                    "number of trainable",
                    "parameters",
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
                    "reduce",
                    "reduced",
                    "times",
                    "no additional",
                ],
                [
                    "reduce the number",
                    "memory requirement",
                    "higher training throughput",
                    "no additional",
                    "fewer trainable parameters",
                ],
                2,
            ),
            (
                self._looks_like_method_adaptation_question(question),
                [
                    "adaptation",
                    "fine-tuning",
                    "pre-trained",
                    "pretrained",
                    "weights",
                    "frozen",
                    "freeze",
                    "rank",
                    "low-rank",
                    "decomposition",
                    "matrix",
                    "matrices",
                    "downstream",
                    "update",
                    "inject",
                ],
                ["we propose", "we introduce", "our approach", "pre-trained weights", "rank decomposition"],
                3,
            ),
            (
                self._looks_like_design_or_training_factor_question(question),
                [
                    "design",
                    "training",
                    "factor",
                    "factors",
                    "component",
                    "components",
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
                    "ablation",
                    "study",
                ],
                ["critical role", "systematically study", "we show", "we find", "training steps"],
                3,
            ),
            (
                self._looks_like_metric_result_question(question),
                [
                    "table 1",
                    "table 2",
                    "table 3",
                    "benchmark",
                    "metric",
                    "score",
                    "result",
                    "zero-shot",
                    "accuracy",
                    "error",
                    "f1",
                    "auc",
                    "perplexity",
                ],
                ["state-of-the-art", "outperforms", "experimental results", "benchmark results"],
                2,
            ),
        ]
        active = [spec for spec in specs if spec[0]]
        if not active:
            return []

        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            normalized = text.lower()
            best_score = 0.0
            for _, terms, bonus_phrases, min_hits in active:
                hits = [term for term in terms if term in normalized]
                if len(hits) < min_hits:
                    continue
                score = item.score + min(0.85, len(set(hits)) * 0.11)
                score += self._question_relevance_score(question, f"{item.paper_name}\n{text}") * 0.45
                score += self._scientific_evidence_shape_bonus(question=question, item=item, text=text) * 0.5
                for phrase in bonus_phrases:
                    if phrase in normalized:
                        score += 0.18
                if "table" in (item.chunk_type or "").lower():
                    score += 0.12
                if any(marker in (item.chunk_type or "").lower() for marker in ["image", "figure", "chart"]):
                    score += 0.08
                best_score = max(best_score, score)
            if best_score <= 0:
                continue
            scored.append((best_score, position, item))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        selected: list[EvidenceItem] = []
        seen: set[str] = set()
        active_terms = [term for _, terms, _, _ in active for term in terms]
        active_bonus = [phrase for _, _, phrases, _ in active for phrase in phrases]
        for _, _, item in scored:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            text = self._sanitize_evidence_text(item.text)
            if "table" in (item.chunk_type or "").lower() or any(
                marker in (item.chunk_type or "").lower() for marker in ["image", "figure", "chart"]
            ):
                item.quote = self._keyword_focused_quote(
                    text=text,
                    terms=active_terms,
                    bonus_phrases=active_bonus,
                    limit=520,
                )
            else:
                item.quote = self._keyword_focused_quote(
                    text=text,
                    terms=active_terms,
                    bonus_phrases=active_bonus,
                    limit=520,
                )
            selected.append(item)
            if len(selected) >= max(1, limit):
                break
        return selected

    def _keyword_focused_quote(
        self,
        *,
        text: str,
        terms: list[str],
        bonus_phrases: list[str],
        limit: int,
    ) -> str:
        normalized_terms = [term.lower() for term in terms if term]
        normalized_bonus = [phrase.lower() for phrase in bonus_phrases if phrase]
        clean_text = self._sanitize_evidence_text(text)
        if self._is_table_like_text(clean_text):
            return self._best_table_quote_for_question(
                " ".join([*terms, *bonus_phrases]),
                clean_text,
                limit=limit,
            )
        quote = self._focused_sentence_quote(
            question=" ".join([*terms, *bonus_phrases]),
            text=clean_text,
            preferred_keywords=[],
            extra_terms=normalized_terms,
            bonus_phrases=normalized_bonus,
            limit=limit,
        )
        return quote or self._truncate_readable_text(clean_text, limit=limit)

    def _select_document_balanced_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        limit: int,
        target_document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        scored: list[tuple[float, int, EvidenceItem]] = []
        allow_tables = self._looks_like_table_question(question)
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            if not text.strip():
                continue
            table_like = self._is_table_like_text(text)
            if self._looks_like_front_matter_noise(text) and not (allow_tables and table_like):
                continue
            score = (
                item.score
                + self._question_relevance_score(question, text) * 0.75
                + self._readable_text_score(text) * 0.18
                + self._scientific_evidence_shape_bonus(question=question, item=item, text=text) * 0.4
            )
            if allow_tables and table_like:
                score += 1.2
            scored.append((score, position, item))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        if not scored:
            return evidence[:limit]

        selected: list[EvidenceItem] = []
        selected_chunks: set[str] = set()
        scored_document_ids = list(dict.fromkeys(item.document_id for _, _, item in scored if item.document_id))
        if target_document_ids:
            scored_id_set = set(scored_document_ids)
            document_ids = [
                document_id
                for document_id in dict.fromkeys(target_document_ids)
                if document_id and document_id in scored_id_set
            ] or scored_document_ids
        else:
            document_ids = scored_document_ids

        roles = self._comparative_balance_roles(question=question, document_count=len(document_ids), limit=limit)
        for role in roles:
            for document_id in document_ids:
                candidate = self._best_comparative_role_candidate(
                    question=question,
                    evidence=evidence,
                    document_id=document_id,
                    role=role,
                    excluded_chunks=selected_chunks,
                )
                if not candidate:
                    continue
                self._set_comparative_role_quote(question=question, item=candidate, role=role)
                selected.append(candidate)
                selected_chunks.add(candidate.chunk_id)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

        for document_id in document_ids:
            if any(item.document_id == document_id for item in selected):
                continue
            for _, _, item in scored:
                if item.document_id != document_id or item.chunk_id in selected_chunks:
                    continue
                item.quote = self._best_quote_for_question(question, item.text)
                selected.append(item)
                selected_chunks.add(item.chunk_id)
                break

        per_document_cap = max(2, limit // max(len(document_ids), 1) + 1)
        per_document_counts = {
            item.document_id: 1 for item in selected if item.document_id
        }
        for _, _, item in scored:
            if item.chunk_id in selected_chunks:
                continue
            count = per_document_counts.get(item.document_id, 0)
            if count >= per_document_cap:
                continue
            item.quote = self._best_quote_for_question(question, item.text)
            selected.append(item)
            selected_chunks.add(item.chunk_id)
            per_document_counts[item.document_id] = count + 1
            if len(selected) >= limit:
                break
        return selected[:limit] or evidence[:limit]

    def _select_diverse_overview_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []

        selected: list[EvidenceItem] = []
        seen: set[str] = set()
        roles = ["purpose", "approach", "conclusion", "claim"]
        if self._looks_like_experiment_content_overview_question(question):
            roles = ["purpose", "approach", "claim"]

        def example_penalty(text: str) -> float:
            markers = ["例如", "案例", "设想", "在某高校", "课程中", "论文初稿", "学生在提交"]
            return 1.0 if any(marker in text for marker in markers) else 0.0

        def pick_for_role(role: str) -> EvidenceItem | None:
            scored: list[tuple[float, int, EvidenceItem]] = []
            for position, item in enumerate(evidence):
                if item.chunk_id in seen:
                    continue
                text = self._sanitize_evidence_text(item.text)
                if not text.strip():
                    continue
                if self._looks_like_front_matter_noise(text):
                    continue
                if self._looks_like_reference_section_text(text):
                    continue
                if self._looks_like_submission_or_assignment_noise(text) and not self._has_experiment_overview_anchor(text):
                    continue
                role_scores = self._semantic_role_scores(
                    text=text,
                    section=item.section or "",
                    index=0,
                    total=1,
                )
                role_value = role_scores.get(role, 0.0)
                if role_value <= 0:
                    continue
                score = (
                    role_value
                    + item.score * 0.25
                    + self._question_relevance_score(question, text) * 0.35
                    + self._overview_structure_score(text) * 0.25
                    - example_penalty(text)
                )
                scored.append((score, position, item))
            scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            return scored[0][2] if scored else None

        for role in roles:
            item = pick_for_role(role)
            if not item:
                continue
            item.quote = self._best_quote_for_question(question, item.text)
            selected.append(item)
            seen.add(item.chunk_id)
            if len(selected) >= limit:
                break

        for item in evidence:
            if item.chunk_id in seen:
                continue
            text = self._sanitize_evidence_text(item.text)
            if self._looks_like_front_matter_noise(text):
                continue
            if self._looks_like_reference_section_text(text):
                continue
            item.quote = self._best_quote_for_question(question, item.text)
            selected.append(item)
            seen.add(item.chunk_id)
            if len(selected) >= limit:
                break

        return selected[:limit] or evidence[:limit]
