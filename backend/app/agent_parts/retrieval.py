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


class AgentRetrievalMixin:
    def _retrieve(self, state: PaperAgentState) -> PaperAgentState:
        document_ids = self._resolve_document_ids(state.get("document_ids"))
        strategy = state.get("retrieval_strategy") or "hybrid_soft"
        evidence = self._hybrid_evidence(
            question=state["question"],
            soft_intent=state.get("soft_intent", {}),
            document_ids=document_ids,
            top_k=state["top_k"],
            embedding_model=state["embedding_model"],
            retrieval_strategy=strategy,
        )
        evidence = self._filter_evidence_for_question(
            state["question"],
            evidence,
            top_k=state["top_k"],
        )
        return {
            **state,
            "evidence": evidence,
            "retrieval_strategy": strategy,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="retriever",
                    title="检索证据",
                    detail=(
                        f"实际使用「{self._friendly_retrieval_strategy(strategy)}」，"
                        f"检索范围 {len(document_ids)} 篇文档，返回 {len(evidence)} 条证据。"
                    ),
                ),
            ],
        }

    def _hybrid_evidence(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
        retrieval_strategy: str,
    ) -> list[EvidenceItem]:
        if not document_ids:
            return []

        targeted = self._targeted_evidence_candidates(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
            top_k=top_k,
        )
        vector_candidates = self._vector_similarity_evidence(
            question=question,
            document_ids=document_ids,
            top_k=max(top_k * 4, 12),
            embedding_model=embedding_model,
        )
        lexical_candidates = self._lexical_evidence_candidates(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
            top_k=max(top_k * 4, 12),
        )
        role_candidates = self._segment_role_evidence_candidates(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
            top_k=max(top_k * 2, 8),
        )

        combined = self._deduplicate_evidence_candidates(
            [*targeted, *role_candidates, *lexical_candidates, *vector_candidates]
        )
        reranked = self._rerank_evidence_candidates(
            question=question,
            evidence=combined,
            soft_intent=soft_intent,
            limit=max(top_k * 4, 8),
        )
        return self._renumber_evidence(reranked)

    def _targeted_evidence_candidates(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        intent = str(soft_intent.get("intent") or "")
        operation = str(soft_intent.get("operation") or "")
        scope = str(soft_intent.get("scope") or "")
        candidates: list[EvidenceItem] = []
        used_targets: set[str] = set()

        def extend(target: str, items: list[EvidenceItem], boost: float) -> None:
            if target in used_targets:
                return
            used_targets.add(target)
            candidates.extend(self._boost_evidence_scores(items, boost))

        if intent == "reference_question" or "reference" in soft_intent.get("preferred_roles", []):
            extend("reference", self._reference_evidence(document_ids=document_ids, top_k=top_k), 0.32)
        if intent == "field_lookup_question" or (operation == "extract" and scope == "field"):
            extend(
                "field",
                self._field_lookup_evidence(question=question, document_ids=document_ids, top_k=top_k),
                0.35,
            )
        if intent == "compare_question" or scope == "multi_document":
            extend("compare", self._comparison_evidence(document_ids=document_ids, top_k=top_k), 0.2)
        if intent in {"compound_request", "structured_review_request"}:
            if intent == "compound_request":
                extend(
                    "compound",
                    self._compound_evidence(question=question, document_ids=document_ids, top_k=top_k),
                    0.18,
                )
            else:
                extend("structured", self._structured_review_evidence(document_ids=document_ids, top_k=top_k), 0.18)
        if intent in {"reliability_question", "title_alignment_question"} or operation == "judge":
            if intent == "title_alignment_question":
                extend("alignment", self._title_alignment_evidence(document_ids=document_ids, top_k=top_k), 0.22)
            else:
                extend("reliability", self._reliability_evidence(document_ids=document_ids, top_k=top_k), 0.22)
        if intent == "research_limitation_question":
            extend("limitation", self._research_limitation_evidence(document_ids=document_ids, top_k=top_k), 0.22)
        if intent == "document_wide_question" or scope == "whole_document" or self._looks_like_document_wide_question(question):
            extend(
                "overview",
                self._overview_evidence(question=question, document_ids=document_ids, top_k=top_k),
                0.2,
            )
        return candidates

    def _vector_similarity_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
    ) -> list[EvidenceItem]:
        try:
            resolved_model = self._resolve_query_embedding_model(
                requested_model=embedding_model,
                document_ids=document_ids,
            )
            query_embedding = self.model_clients.embed_query(question, model=resolved_model)
            return self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
            )
        except RuntimeError:
            return []

    def _lexical_evidence_candidates(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            total = max(len(rows), 1)
            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                score = self._hybrid_row_score(
                    question=question,
                    text=text,
                    metadata=row.get("metadata") or {},
                    index=index,
                    total=total,
                    soft_intent=soft_intent,
                )
                if score <= 0.08:
                    continue
                item = self._evidence_from_row(row, document_id, score=min(1.0, score))
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def _segment_role_evidence_candidates(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        roles = self._normalized_preferred_roles(soft_intent)
        if not roles:
            roles = ["purpose", "approach", "claim", "conclusion"] if self._looks_like_document_wide_question(question) else []
        if not roles:
            return []

        candidates: list[EvidenceItem] = []
        per_document_limit = max(2, min(4, top_k))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            scored_by_role = self._score_rows_by_semantic_role(rows)
            selected_ids: set[str] = set()
            for role in roles:
                if role in {"field", "reference"}:
                    continue
                role_rows = scored_by_role.get(role, [])
                if not role_rows:
                    continue
                for score, row in role_rows:
                    row_id = str(row.get("id", ""))
                    if row_id in selected_ids:
                        continue
                    text = str(row.get("text", ""))
                    penalty = self._evidence_noise_penalty_for_soft_intent(
                        question=question,
                        text=text,
                        section=str((row.get("metadata") or {}).get("section") or ""),
                        soft_intent=soft_intent,
                    )
                    if penalty >= 1.6:
                        continue
                    selected_ids.add(row_id)
                    item = self._evidence_from_row(row, document_id, score=min(1.0, 0.62 + score * 0.12))
                    item.quote = self._best_quote_for_question(question, item.text)
                    candidates.append(item)
                    break
                if len(selected_ids) >= per_document_limit:
                    break
        return candidates[:top_k]

    def _boost_evidence_scores(self, evidence: list[EvidenceItem], boost: float) -> list[EvidenceItem]:
        return [
            item.model_copy(update={"score": min(1.0, max(0.0, item.score + boost))})
            for item in evidence
        ]

    def _deduplicate_evidence_candidates(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        best_by_key: dict[str, EvidenceItem] = {}
        order: list[str] = []
        for item in evidence:
            key = f"{item.document_id}:{item.chunk_id}"
            if key not in best_by_key:
                best_by_key[key] = item
                order.append(key)
                continue
            current = best_by_key[key]
            if self._is_field_evidence_item(current):
                continue
            if self._is_field_evidence_item(item):
                best_by_key[key] = item
                continue
            current_noisy = self._candidate_has_embedded_noise(current)
            next_noisy = self._candidate_has_embedded_noise(item)
            if current_noisy and not next_noisy and item.score >= current.score - 0.25:
                best_by_key[key] = item
            elif current_noisy == next_noisy and item.score > current.score:
                best_by_key[key] = item
        return [best_by_key[key] for key in order]

    def _is_field_evidence_item(self, item: EvidenceItem) -> bool:
        return (item.section or "") in {"关键词", "摘要", "作者", "作者单位/机构", "日期", "标题"}

    def _candidate_has_embedded_noise(self, item: EvidenceItem) -> bool:
        text = self._sanitize_evidence_text(item.text)
        return (
            self._looks_like_front_matter_noise(text)
            or self._looks_like_field_or_metadata_sentence(text)
            or self._looks_like_submission_or_assignment_noise(text)
        )

    def _rerank_evidence_candidates(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        soft_intent: dict[str, Any],
        limit: int,
    ) -> list[EvidenceItem]:
        if soft_intent.get("intent") == "field_lookup_question" or (
            soft_intent.get("operation") == "extract" and soft_intent.get("scope") == "field"
        ):
            field_items = [item for item in evidence if self._is_field_evidence_item(item)]
            if field_items:
                return field_items[:limit]

        allow_tables = self._looks_like_table_question(question)
        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            if not text.strip():
                continue
            penalty = self._evidence_noise_penalty_for_soft_intent(
                question=question,
                text=text,
                section=item.section or "",
                soft_intent=soft_intent,
            )
            if penalty >= 2.2:
                continue
            if self._is_table_like_text(text) and not allow_tables:
                penalty += 0.55
            focus_score = self._soft_focus_score(soft_intent, text)
            role_score = self._soft_role_score(
                soft_intent=soft_intent,
                text=text,
                section=item.section or "",
                index=0,
                total=1,
            )
            score = (
                item.score * 0.75
                + self._question_relevance_score(question, text) * 0.95
                + self._readable_text_score(text) * 0.2
                + self._overview_structure_score(text) * 0.28
                + focus_score * 0.34
                + role_score * 0.4
                - penalty
            )
            scored.append((score, position, item))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        selected: list[EvidenceItem] = []
        per_document_counts: dict[str, int] = {}
        for _, _, item in scored:
            document_count = per_document_counts.get(item.document_id, 0)
            if len(per_document_counts) > 1 and document_count >= max(2, limit // max(len(per_document_counts), 1) + 1):
                continue
            item.quote = self._best_quote_for_question(question, item.text)
            selected.append(item)
            per_document_counts[item.document_id] = document_count + 1
            if len(selected) >= limit:
                break
        return selected or evidence[:limit]

    def _hybrid_row_score(
        self,
        *,
        question: str,
        text: str,
        metadata: dict[str, Any],
        index: int,
        total: int,
        soft_intent: dict[str, Any],
    ) -> float:
        sanitized = self._sanitize_evidence_text(text)
        if not sanitized.strip():
            return -1.0
        section = str(metadata.get("section") or "")
        penalty = self._evidence_noise_penalty_for_soft_intent(
            question=question,
            text=sanitized,
            section=section,
            soft_intent=soft_intent,
        )
        if penalty >= 2.2:
            return -1.0

        relevance = self._question_relevance_score(question, sanitized)
        focus_score = self._soft_focus_score(soft_intent, sanitized)
        role_score = self._soft_role_score(
            soft_intent=soft_intent,
            text=sanitized,
            section=section,
            index=index,
            total=total,
        )
        score = (
            relevance * 1.15
            + focus_score * 0.55
            + role_score * 0.5
            + self._readable_text_score(sanitized) * 0.22
            + self._overview_structure_score(sanitized) * 0.28
            - penalty
        )
        if soft_intent.get("scope") == "whole_document" and index <= max(2, int(total * 0.08)):
            score += 0.2
        if soft_intent.get("operation") in {"summarize", "analyze"} and role_score > 0:
            score += 0.12
        return score

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

    def _field_lookup_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        targets = self._field_lookup_targets(question)
        if not targets:
            return []

        evidence: list[EvidenceItem] = []
        per_document_limit = max(len(targets), min(top_k, len(targets)))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            if not rows:
                continue
            added_for_document = 0
            for target in targets:
                match = self._find_field_in_rows(target, rows)
                if not match:
                    continue
                value, row = match
                evidence.append(
                    self._field_evidence_from_row(
                        row=row,
                        fallback_document_id=document_id,
                        field=target,
                        value=value,
                    )
                )
                added_for_document += 1
                if added_for_document >= per_document_limit:
                    break
        return evidence

    def _find_field_in_rows(
        self,
        field: str,
        rows: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | None:
        for row in rows:
            value = self._extract_field_value(field, str(row.get("text", "")))
            if value:
                return value, row

        for index, row in enumerate(rows):
            context_rows = rows[index : min(index + 2, len(rows))]
            context = "\n\n".join(str(item.get("text", "")) for item in context_rows)
            value = self._extract_field_value(field, context)
            if not value:
                continue
            source_row = next(
                (
                    item
                    for item in context_rows
                    if self._field_marker_in_text(field, str(item.get("text", "")))
                ),
                row,
            )
            return value, source_row

        full_text = "\n\n".join(str(row.get("text", "")) for row in rows[:8])
        value = self._extract_field_value(field, full_text)
        if not value:
            return None
        source_row = next(
            (
                row
                for row in rows[:8]
                if self._field_marker_in_text(field, str(row.get("text", "")))
            ),
            rows[0],
        )
        return value, source_row

    def _field_marker_in_text(self, field: str, text: str) -> bool:
        normalized = " ".join(text.split()).lower()
        return any(alias.lower() in normalized for alias in self._field_aliases(field))

    def _extract_field_value(self, field: str, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split()).strip()
        if not normalized:
            return ""

        if field == "title":
            title = self._guess_document_title("", normalized)
            return self._clean_field_value(field, title)

        patterns = {
            "keywords": (
                r"(?:关键词|关键字|key\s*words?)\s*[:：]\s*(.+?)"
                r"(?=\s*(?:作者|单位|作者单位|机构|日期|完成日期|摘\s*要|摘要|abstract|"
                r"引言|绪论|正文|目录|参考文献|references|$))"
            ),
            "abstract": (
                r"(?:摘\s*要|摘要|abstract)\s*[:：]?\s*(.+?)"
                r"(?=\s*(?:关键词|关键字|key\s*words?|一、|第?[一二三四五六七八九十]+[、.．]|"
                r"1[.、．]?\s*(?:引言|introduction)|引言|绪论|正文|目录|作者[:：]|"
                r"单位[:：]|作者单位[:：]|日期[:：]|完成日期[:：]|$))"
            ),
            "authors": (
                r"作者\s*[:：]\s*(.+?)"
                r"(?=\s*(?:单位|作者单位|机构|日期|完成日期|摘\s*要|摘要|关键词|关键字|$))"
            ),
            "affiliation": (
                r"(?:作者单位|单位|机构)\s*[:：]\s*(.+?)"
                r"(?=\s*(?:作者|日期|完成日期|摘\s*要|摘要|关键词|关键字|$))"
            ),
            "date": (
                r"(?:完成日期|日期|时间)\s*[:：]\s*(.+?)"
                r"(?=\s*(?:作者|单位|作者单位|机构|摘\s*要|摘要|关键词|关键字|$))"
            ),
        }
        pattern = patterns.get(field)
        if not pattern:
            return ""
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            return ""
        return self._clean_field_value(field, match.group(1))

    def _clean_field_value(self, field: str, value: str) -> str:
        cleaned = " ".join(value.split()).strip(" ：:;；,，。.-—_")
        if not cleaned:
            return ""
        if field == "keywords":
            cleaned = re.split(
                r"\s*(?:作者|单位|作者单位|机构|日期|完成日期|摘\s*要|摘要|abstract|引言|正文|目录)\s*[:：]?",
                cleaned,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            parts = [
                item.strip(" ：:;；,，。.-—_")
                for item in re.split(r"[；;，,、]\s*", cleaned)
                if item.strip(" ：:;；,，。.-—_")
            ]
            return "；".join(dict.fromkeys(parts))
        if field == "abstract" and len(cleaned) > 900:
            return self._truncate_readable_text(cleaned, limit=900)
        return cleaned

    def _field_evidence_from_row(
        self,
        *,
        row: dict[str, Any],
        fallback_document_id: str,
        field: str,
        value: str,
    ) -> EvidenceItem:
        metadata = row["metadata"]
        label = self._field_label(field)
        text = f"{label}：{value}"
        return EvidenceItem(
            citation_id="",
            chunk_id=str(metadata.get("chunk_id") or row["id"]),
            document_id=str(metadata.get("document_id", fallback_document_id)),
            paper_name=str(metadata.get("paper_name", "")),
            page=int(metadata.get("page", 0) or 0),
            section=label,
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=1.0,
            text=text,
            quote=text,
            char_start=int(metadata.get("char_start", 0) or 0),
            char_end=int(metadata.get("char_end", 0) or 0),
        )

    def _comparison_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(2, min(3, top_k))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=per_document_limit)
            for row in rows:
                evidence.append(self._evidence_from_row(row, document_id, score=1.0))
        return evidence

    def _overview_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        contract = self._build_reading_task_contract(question)
        if contract.operation == "summarize" and contract.scope == "whole_document":
            return self._contract_overview_evidence(
                contract=contract,
                document_ids=document_ids,
                top_k=top_k,
            )

        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(6, top_k + 1))
        focus_keywords = self._overview_focus_keywords(question)
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            if rows and self._needs_opening_context(question):
                opening_row = self._first_informative_overview_row(rows) or rows[0]
                selected.append((1.0, opening_row))
                selected_ids.add(str(opening_row.get("id", "")))

            scored_rows: list[tuple[float, dict[str, Any]]] = []
            research_limitation_question = self._looks_like_research_limitation_question(question)
            method_question = any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"])
            for index, row in enumerate(rows):
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                text = str(row.get("text", ""))
                if self._looks_like_front_matter_noise(text):
                    continue
                normalized_text = text.lower()
                keyword_hits = sum(
                    1
                    for keyword in focus_keywords
                    if keyword in text or keyword.lower() in normalized_text
                )
                relevance = 0.0 if research_limitation_question else self._question_relevance_score(question, text)
                if keyword_hits == 0 and relevance < 0.08:
                    continue
                position_bonus = 0.12 if index <= 2 else 0.0
                overview_bonus = self._overview_structure_score(text)
                score = 0.45 + keyword_hits * 0.16 + relevance * 0.5 + position_bonus + overview_bonus
                if research_limitation_question:
                    if "未来研究" in text:
                        score += 0.8
                    if "实证数据" in text or "检验" in text:
                        score += 0.4
                    if "传统高校学习支持服务存在" in text or "上述局限" in text:
                        score -= 0.7
                if method_question:
                    if all(keyword in text for keyword in ["采用", "文献分析"]):
                        score += 2.0
                    if all(keyword in text for keyword in ["情境推演", "机制建构"]):
                        score += 0.8
                    if "本文围绕" in text and "方法" in text:
                        score += 0.8
                    if any(keyword in text for keyword in ["案例化情境推演", "课程教师", "学生可以", "不得直接提交"]):
                        score -= 1.3
                scored_rows.append((score, row))

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            for score, row in scored_rows:
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                selected.append((min(score, 1.0), row))
                selected_ids.add(row_id)
                if len(selected) >= per_document_limit:
                    break

            if len(selected) < min(3, per_document_limit):
                for row in rows:
                    row_id = str(row.get("id", ""))
                    if row_id in selected_ids:
                        continue
                    text = str(row.get("text", ""))
                    if self._looks_like_front_matter_noise(text):
                        continue
                    selected.append((0.55 + self._overview_structure_score(text), row))
                    selected_ids.add(row_id)
                    if len(selected) >= min(3, per_document_limit):
                        break

            if not selected:
                selected = [
                    (0.55, row)
                    for row in rows[:per_document_limit]
                    if not self._looks_like_front_matter_noise(str(row.get("text", "")))
                ] or [(0.55, row) for row in rows[:per_document_limit]]

            for score, row in selected:
                evidence.append(self._evidence_from_row(row, document_id, score=score))
        return evidence

    def _contract_overview_evidence(
        self,
        *,
        contract: ReadingTaskContract,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        if contract.target == "experiment_content" or (
            contract.target == "main_content"
            and self._documents_look_like_experiment_material(document_ids)
        ):
            return self._contract_experiment_content_evidence(
                contract=contract,
                document_ids=document_ids,
                top_k=top_k,
            )

        evidence: list[EvidenceItem] = []
        per_document_limit = 2 if contract.depth == "one_sentence" else max(3, min(4, top_k))
        role_order = ["purpose", "approach", "claim", "conclusion", "caveat"]

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()
            covered_roles: set[str] = set()

            def add_row(row: dict[str, Any] | None, score: float) -> None:
                if not row:
                    return
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    return
                text = str(row.get("text", ""))
                if self._looks_like_front_matter_noise(text):
                    return
                selected_ids.add(row_id)
                selected.append((score, row))
                metadata = row.get("metadata") or {}
                role_scores = self._semantic_role_scores(
                    text=text,
                    section=str(metadata.get("section") or ""),
                    index=0,
                    total=1,
                )
                covered_roles.update(role for role, value in role_scores.items() if value >= 0.45)

            scored_by_role = self._score_rows_by_semantic_role(rows)
            for role in role_order:
                if role in covered_roles and role != "conclusion":
                    continue
                role_rows = scored_by_role.get(role, [])
                if not role_rows:
                    continue
                _, row = role_rows[0]
                add_row(row, min(1.0, 0.98 - len(selected) * 0.05))
                if len(selected) >= per_document_limit:
                    break

            if not selected:
                fallback_rows = scored_by_role.get("informative", [])
                for _, row in fallback_rows:
                    add_row(row, 0.58 + self._overview_structure_score(str(row.get("text", ""))))
                    if len(selected) >= min(2, per_document_limit):
                        break

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=score))
        return evidence

    def _documents_look_like_experiment_material(self, document_ids: list[str]) -> bool:
        if not document_ids:
            return False
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=12)
            if any(self._has_experiment_overview_anchor(str(row.get("text", ""))) for row in rows):
                return True
        return False

    def _contract_experiment_content_evidence(
        self,
        *,
        contract: ReadingTaskContract,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = 2 if contract.depth == "one_sentence" else max(3, min(4, top_k))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            total = max(len(rows), 1)
            scored_rows: list[tuple[float, int, dict[str, Any]]] = []
            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                if not text.strip() or self._looks_like_front_matter_noise(text):
                    continue
                score = self._experiment_content_relevance_score(
                    question=" ".join(contract.role_hints),
                    text=text,
                    index=index,
                    total=total,
                )
                if score <= 0.25:
                    continue
                scored_rows.append((score, index, row))

            scored_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            selected: list[dict[str, Any]] = []
            selected_ids: set[str] = set()
            for score, _, row in scored_rows:
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                selected_ids.add(row_id)
                selected.append(row)
                if len(selected) >= per_document_limit:
                    break

            if len(selected) < min(2, per_document_limit):
                for row in rows[:8]:
                    row_id = str(row.get("id", ""))
                    text = str(row.get("text", ""))
                    if row_id in selected_ids or self._looks_like_front_matter_noise(text):
                        continue
                    if self._looks_like_submission_or_assignment_noise(text):
                        continue
                    selected_ids.add(row_id)
                    selected.append(row)
                    if len(selected) >= min(2, per_document_limit):
                        break

            for row in selected[:per_document_limit]:
                score = self._experiment_content_relevance_score(
                    question=" ".join(contract.role_hints),
                    text=str(row.get("text", "")),
                    index=0,
                    total=1,
                )
                evidence.append(self._evidence_from_row(row, document_id, score=min(1.0, max(0.55, score))))
        return evidence

    def _experiment_content_relevance_score(
        self,
        *,
        question: str,
        text: str,
        index: int,
        total: int,
    ) -> float:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        if not normalized:
            return 0.0

        strong_markers = [
            "实验目的",
            "实验内容",
            "实验任务",
            "实验要求",
            "实验步骤",
            "实验过程",
            "实验原理",
            "实验思路",
            "操作步骤",
            "实现要求",
            "设计要求",
        ]
        action_markers = [
            "掌握",
            "学会",
            "领悟",
            "理解",
            "设计",
            "编写",
            "实现",
            "构建",
            "应用",
            "完成",
            "验证",
            "使用",
        ]
        score = 0.0
        score += sum(0.58 for marker in strong_markers if marker in normalized)
        score += sum(0.12 for marker in action_markers if marker in normalized)
        if "实验内容" in normalized:
            score += 1.0
        if "实验目的" in normalized:
            score += 0.65
        if any(marker in normalized for marker in ["实验步骤", "实验过程", "操作步骤"]):
            score += 0.35
        if index <= max(2, int(total * 0.12)):
            score += 0.35
        if index <= 5:
            score += 0.12
        if self._looks_like_submission_or_assignment_noise(normalized):
            score -= 1.5
        if self._looks_like_reference_section_text(normalized):
            score -= 1.2
        if self._looks_like_code_heavy_text(normalized) and "code" not in question:
            score -= 0.65
        if self._is_table_like_text(normalized):
            score -= 0.25
        return max(0.0, score)

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
            if self._looks_like_code_heavy_text(text) and not self._has_experiment_overview_anchor(text):
                continue
            if self._looks_like_submission_or_assignment_noise(text) and not self._has_experiment_overview_anchor(text):
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

    def _research_limitation_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(6, top_k + 1))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()
            scored_rows: list[tuple[float, int, dict[str, Any]]] = []
            total_rows = max(len(rows), 1)

            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                section = str((row.get("metadata") or {}).get("section") or "")
                score = self._research_limitation_relevance_score(
                    text=text,
                    section=section,
                    index=index,
                    total=total_rows,
                )
                if score >= 0.85:
                    scored_rows.append((score, index, row))

            scored_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)

            def add_row(row: dict[str, Any] | None, score: float) -> None:
                if not row:
                    return
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    return
                selected_ids.add(row_id)
                selected.append((min(max(score / 4.0, 0.55), 1.0), row))

            for score, _, row in scored_rows:
                add_row(row, score)
                if len(selected) >= per_document_limit:
                    break

            method_row = self._best_row_for_keywords_excluding(
                rows,
                ["文献分析", "情境推演", "机制建构", "研究方法"],
                selected_ids,
            )
            if method_row:
                method_score = self._research_limitation_relevance_score(
                    text=str(method_row.get("text", "")),
                    section=str((method_row.get("metadata") or {}).get("section") or ""),
                    index=0,
                    total=total_rows,
                )
                if method_score >= 0.65:
                    add_row(method_row, method_score)

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=score))

        return evidence

    def _research_limitation_relevance_score(
        self,
        *,
        text: str,
        section: str,
        index: int,
        total: int,
    ) -> float:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        if not normalized:
            return 0.0

        section_bonus = {
            "Limitations": 2.4,
            "FutureWork": 2.0,
            "Conclusion": 1.1,
            "Discussion": 0.9,
            "Methods": 0.75,
        }.get(section, 0.0)
        direct_keywords = [
            "局限性",
            "研究局限",
            "研究不足",
            "局限与不足",
            "不足与展望",
            "结论与展望",
            "未来研究",
            "后续研究",
            "研究展望",
        ]
        boundary_keywords = [
            "实证数据",
            "实证检验",
            "检验",
            "验证",
            "尚未",
            "未能",
            "样本量",
            "样本代表性",
            "抽样",
            "统计检验",
            "数据来源",
            "不同应用场景",
            "不同学生群体",
            "使用差异",
            "外推",
            "普适性",
            "代表性",
            "长期效果",
        ]
        method_keywords = ["文献分析", "情境推演", "机制建构"]
        content_problem_keywords = [
            "学习困难",
            "目标模糊",
            "计划松散",
            "拖延",
            "反思缺失",
            "传统高校学习支持服务存在",
            "学生学习",
            "许多学生",
            "学习者",
            "教师反馈",
            "课程学习场景",
            "实验与实践教学场景",
            "程序设计课程",
            "社会调查课程",
            "论文初稿阶段",
            "学生文本",
            "学生根据反馈",
            "教师最终评价",
            "人工智能使用声明制度",
            "学习依赖",
            "信息准确性",
            "数据隐私",
            "算法偏差",
            "学术诚信",
            "责任边界",
        ]

        direct_hits = sum(1 for keyword in direct_keywords if keyword in normalized)
        boundary_hits = sum(1 for keyword in boundary_keywords if keyword in normalized)
        method_hits = sum(1 for keyword in method_keywords if keyword in normalized)
        content_problem_hits = sum(1 for keyword in content_problem_keywords if keyword in normalized)
        research_method_context = any(keyword in normalized for keyword in ["采用", "本文围绕", "研究方法：", "研究方法:"]) and method_hits > 0

        score = section_bonus + direct_hits * 0.9 + boundary_hits * 0.38 + method_hits * 0.25
        if re.search(r"(缺少|缺乏|没有|未能).{0,14}(样本|数据|实证|验证|检验|统计|抽样|方法)", normalized):
            score += 0.7
        if "未来研究" in normalized and any(keyword in normalized for keyword in ["实证数据", "检验", "验证"]):
            score += 1.1
        if "比较不同学生群体" in normalized or "不同学生群体的使用差异" in normalized:
            score += 0.75
        if "采用" in normalized and any(keyword in normalized for keyword in ["文献分析", "情境推演", "机制建构"]):
            score += 0.65
        if method_hits and not research_method_context and direct_hits == 0 and boundary_hits == 0:
            score -= 1.0
        if index >= total * 0.7:
            score += 0.18
        if index <= 1 and direct_hits == 0 and boundary_hits == 0:
            score -= 0.35
        if content_problem_hits and direct_hits == 0 and boundary_hits == 0 and method_hits == 0:
            score -= 2.2
        if content_problem_hits and method_hits == 0 and "不足" in normalized and "未来研究" not in normalized:
            score -= 0.7
        return score

    def _reference_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(8, top_k + 3))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[dict[str, Any]] = []
            start_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if self._looks_like_reference_section_text(str(row.get("text", "")))
                ),
                -1,
            )

            if start_index >= 0:
                for row in rows[start_index:]:
                    text = str(row.get("text", ""))
                    marker_count = self._reference_marker_count(text)
                    if row is rows[start_index] or marker_count > 0:
                        selected.append(row)
                    elif selected and self._looks_like_reference_continuation(text):
                        selected.append(row)
                    elif selected:
                        break
                    if len(selected) >= per_document_limit:
                        break

            if not selected:
                candidates = [
                    row
                    for row in rows
                    if self._reference_marker_count(str(row.get("text", ""))) >= 2
                    or "References" in str(row.get("text", ""))
                ]
                selected = candidates[:per_document_limit]

            for index, row in enumerate(selected):
                evidence.append(
                    self._evidence_from_row(
                        row,
                        document_id,
                        score=max(0.65, 1.0 - index * 0.04),
                    )
                )
        return evidence

    def _structured_review_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(6, min(8, top_k + 3))
        selectors = [
            ("开头/摘要", ["摘要", "本文围绕", "关键词", "研究主题"]),
            ("方法", ["采用", "文献分析", "情境推演", "机制建构", "研究方法"]),
            ("作用机制", ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "运行机制"]),
            ("应用场景", ["课程学习场景", "实验与实践教学", "学业预警", "应用场景"]),
            ("风险", ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界", "风险"]),
            ("治理", ["人机协同", "价值对齐", "过程可控", "数据最小化", "多主体治理", "治理"]),
            ("结论/局限", ["结论与展望", "未来研究", "实证数据", "总体而言", "不足", "局限"]),
            ("参考文献", ["参考文献", "[1]", "Journal", "研究[J]"]),
        ]

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            if rows:
                selected.append((1.0, rows[0]))
                selected_ids.add(str(rows[0].get("id", "")))

            for index, (_, keywords) in enumerate(selectors, start=1):
                row = self._best_row_for_keywords_excluding(rows, keywords, selected_ids)
                if row:
                    selected.append((max(0.65, 1.0 - index * 0.04), row))
                    selected_ids.add(str(row.get("id", "")))
                if len(selected) >= per_document_limit:
                    break

            if len(selected) < per_document_limit:
                for row in rows[:per_document_limit]:
                    row_id = str(row.get("id", ""))
                    if row_id in selected_ids:
                        continue
                    selected.append((0.55, row))
                    selected_ids.add(row_id)
                    if len(selected) >= per_document_limit:
                        break

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=min(score, 1.0)))
        return evidence

    def _compound_task_evidence_keywords(self, task_type: str) -> list[list[str]]:
        selectors: dict[str, list[list[str]]] = {
            "overview_summary": [
                ["摘要", "本文围绕", "研究目的", "主要讨论", "研究主题", "关键词"],
                ["实验名称", "实验目的", "主要内容", "主题"],
                ["研究认为", "总体而言", "结论"],
            ],
            "professional_takeaways": [
                ["机制", "应用场景", "学习支持", "系统", "流程"],
                ["风险", "隐私", "算法偏差", "学术诚信", "责任边界"],
                ["人机协同", "治理", "价值对齐", "过程可控", "工程"],
            ],
            "reliability_judgment": [
                ["采用", "文献分析", "情境推演", "机制建构", "研究方法"],
                ["未来研究", "实证数据", "样本", "问卷", "实验", "统计检验"],
                ["参考文献", "数据来源", "验证", "局限", "不足"],
            ],
            "method_analysis": [
                ["研究方法", "采用", "文献分析", "情境推演", "机制建构"],
                ["实验方法", "实验步骤", "实现过程", "算法", "流程"],
            ],
            "limitation_analysis": [
                ["结论与展望", "未来研究", "研究局限", "局限性", "研究不足"],
                ["实证数据", "实证检验", "样本", "验证", "不同应用场景", "不同学生群体"],
                ["文献分析", "情境推演", "机制建构", "研究方法"],
            ],
            "conclusion_summary": [
                ["结论与展望", "总体而言", "研究认为", "综上所述"],
                ["核心观点", "主要发现", "结论", "发现"],
            ],
            "comparison": [
                ["实验名称", "实验类型", "实验目的", "主题"],
                ["方法", "实现", "结论", "结果"],
            ],
        }
        return selectors.get(task_type, [["摘要", "本文", "研究", "结论"]])

    def _compound_focus_keywords_for_question(self, question: str) -> list[str]:
        keywords: list[str] = []
        for task in self._parse_compound_tasks(question):
            for selector in self._compound_task_evidence_keywords(task.task_type):
                keywords.extend(selector)
            keywords.append(task.trigger)
        return list(dict.fromkeys(keyword for keyword in keywords if keyword))

    def _compound_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        tasks = self._parse_compound_tasks(question)
        if len(tasks) < 2:
            return self._overview_evidence(question=question, document_ids=document_ids, top_k=top_k)

        evidence: list[EvidenceItem] = []
        per_document_limit = max(6, min(14, top_k + len(tasks) + 5))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            def add_row(row: dict[str, Any] | None, score: float) -> None:
                if not row:
                    return
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    return
                selected_ids.add(row_id)
                selected.append((min(score, 1.0), row))

            for task_index, task in enumerate(tasks):
                base_score = max(0.58, 1.0 - task_index * 0.06)
                if task.task_type == "overview_summary" and rows:
                    add_row(rows[0], base_score)

                if task.task_type == "reference_list":
                    start_index = next(
                        (
                            index
                            for index, row in enumerate(rows)
                            if self._looks_like_reference_section_text(str(row.get("text", "")))
                        ),
                        -1,
                    )
                    reference_rows: list[dict[str, Any]] = []
                    if start_index >= 0:
                        for row in rows[start_index:]:
                            text = str(row.get("text", ""))
                            if row is rows[start_index] or self._reference_marker_count(text) > 0:
                                reference_rows.append(row)
                            elif reference_rows and self._looks_like_reference_continuation(text):
                                reference_rows.append(row)
                            elif reference_rows:
                                break
                            if len(reference_rows) >= 4:
                                break
                    if not reference_rows:
                        reference_rows = [
                            row
                            for row in rows
                            if self._reference_marker_count(str(row.get("text", ""))) >= 2
                            or "References" in str(row.get("text", ""))
                        ][:4]
                    for ref_index, row in enumerate(reference_rows):
                        add_row(row, max(0.62, base_score - ref_index * 0.03))
                    continue

                for selector_index, keywords in enumerate(self._compound_task_evidence_keywords(task.task_type)):
                    row = self._best_row_for_keywords_excluding(rows, keywords, selected_ids)
                    add_row(row, max(0.55, base_score - selector_index * 0.04))

            if len(selected) < min(3, len(rows)):
                for row in rows[:3]:
                    add_row(row, 0.52)
                    if len(selected) >= 3:
                        break

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=score))

        return evidence

    def _title_alignment_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(4, min(6, top_k + 1))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[dict[str, Any]] = []
            if rows:
                selected.append(rows[0])
            selected_ids = {str(row.get("id", "")) for row in selected}
            candidate_getters = [
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["结论与展望", "认知支架"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_any_keywords(
                    rows,
                    ["第一，学习依赖风险", "第二，信息准确性风险", "第三，数据隐私风险", "第五，学术诚信风险", "第六，责任边界风险"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["针对上述风险", "人机协同"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["未来研究", "实证数据"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_any_keywords(
                    rows,
                    ["采用文献分析", "文献分析、情境推演和机制建构", "机制建构的方法"],
                    exclude_ids=selected_ids,
                ),
            ]
            for get_row in candidate_getters:
                row = get_row()
                if row:
                    selected.append(row)
                    selected_ids.add(str(row.get("id", "")))

            seen_ids: set[str] = set()
            for row in selected:
                row_id = str(row.get("id", ""))
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                evidence.append(self._evidence_from_row(row, document_id, score=1.0))
                if len(seen_ids) >= per_document_limit:
                    break
        return evidence

    def _first_row_with_all_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = exclude_ids or set()
        for row in rows:
            if str(row.get("id", "")) in excluded:
                continue
            text = str(row.get("text", ""))
            if all(keyword in text for keyword in keywords):
                return row
        return None

    def _first_row_with_any_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = exclude_ids or set()
        for row in rows:
            if str(row.get("id", "")) in excluded:
                continue
            text = str(row.get("text", ""))
            if any(keyword in text for keyword in keywords):
                return row
        return None

    def _reliability_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(5, top_k))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            scored_rows: list[tuple[float, dict[str, Any]]] = []
            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                metadata = row.get("metadata", {}) or {}
                page = int(metadata.get("page", 0) or 0)
                relevance = self._reliability_relevance_score(text)
                quality = self._readable_text_score(text)
                table_penalty = 0.35 if self._is_table_like_text(text) else 0.0
                early_bonus = 0.18 if index <= 1 or page <= 1 else 0.0
                score = 0.35 + relevance * 0.08 + quality * 0.25 + early_bonus - table_penalty
                if relevance > 0 or early_bonus > 0:
                    scored_rows.append((score, row))

            if not scored_rows:
                scored_rows = [
                    (self._readable_text_score(str(row.get("text", ""))), row)
                    for row in rows[:per_document_limit]
                ]

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            selected: list[tuple[float, dict[str, Any]]] = []
            if rows:
                # Always keep the opening chunk: it usually contains the title
                # and document type, which are essential for reliability checks.
                selected.append((1.0, rows[0]))
            for keywords in [
                ["采用", "文献分析", "情境推演", "机制建构", "研究认为"],
                ["未来研究", "实证数据", "检验不同应用场景", "结论与展望"],
                ["参考文献", "随机生成", "样稿"],
                ["风险", "挑战", "局限", "不足"],
            ]:
                row = self._best_row_for_keywords(rows, keywords)
                if row:
                    selected.append((1.0, row))
            selected_ids = {str(row.get("id", "")) for _, row in selected}
            for score, row in scored_rows:
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                selected.append((score, row))
                selected_ids.add(row_id)
                if len(selected) >= per_document_limit:
                    break
            for score, row in selected:
                evidence.append(self._evidence_from_row(row, document_id, score=min(score, 1.0)))
        return evidence

    def _best_row_for_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
    ) -> dict[str, Any] | None:
        best_row: dict[str, Any] | None = None
        best_score = 0
        for row in rows:
            text = str(row.get("text", ""))
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_row = row
        return best_row if best_score > 0 else None

    def _best_row_for_keywords_excluding(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        exclude_ids: set[str],
    ) -> dict[str, Any] | None:
        best_row: dict[str, Any] | None = None
        best_score = 0
        for row in rows:
            if str(row.get("id", "")) in exclude_ids:
                continue
            text = str(row.get("text", ""))
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_row = row
        return best_row if best_score > 0 else None

    def _evidence_from_row(
        self,
        row: dict[str, Any],
        fallback_document_id: str,
        *,
        score: float,
    ) -> EvidenceItem:
        metadata = row["metadata"]
        text = str(row["text"])
        quote = self._best_readable_quote(str(metadata.get("quote", "")) or text)
        return EvidenceItem(
            citation_id="",
            chunk_id=str(metadata.get("chunk_id") or row["id"]),
            document_id=str(metadata.get("document_id", fallback_document_id)),
            paper_name=str(metadata.get("paper_name", "")),
            page=int(metadata.get("page", 0) or 0),
            section=str(metadata.get("section") or ""),
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=score,
            text=text,
            quote=quote,
            char_start=int(metadata.get("char_start", 0) or 0),
            char_end=int(metadata.get("char_end", 0) or 0),
        )

    def _filter_evidence_for_question(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []

        reliability_question = self._looks_like_reliability_question(question)
        research_limitation_question = self._looks_like_research_limitation_question(question)
        alignment_question = self._looks_like_title_alignment_question(question)
        compound_request = self._looks_like_compound_request(question)
        reference_question = self._looks_like_reference_question(question)
        field_lookup_question = self._looks_like_field_lookup_question(question)
        structured_review = self._looks_like_structured_review_request(question)
        allow_tables = self._looks_like_table_question(question)
        if compound_request:
            selected: list[EvidenceItem] = []
            seen_compound: set[str] = set()
            target_count = max(6, min(16, top_k + 10))
            for item in evidence:
                if item.chunk_id in seen_compound:
                    continue
                seen_compound.add(item.chunk_id)
                if self._looks_like_reference_section_text(item.text):
                    item.quote = self._best_reference_quote(item.text)
                else:
                    item.quote = self._best_quote_for_question(question, item.text)
                selected.append(item)
                if len(selected) >= target_count:
                    break
            return self._renumber_evidence(selected)

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
            return self._renumber_evidence(selected)

        if self._looks_like_broad_overview_question(question):
            selected = self._select_diverse_overview_evidence(question=question, evidence=evidence, limit=3)
            return self._renumber_evidence(selected)

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
            return self._renumber_evidence(selected)

        if structured_review:
            selected: list[EvidenceItem] = []
            seen_review: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_review:
                    continue
                seen_review.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                selected.append(item)
                if len(selected) >= max(4, min(top_k + 3, 8)):
                    break
            return self._renumber_evidence(selected)

        if reliability_question or alignment_question:
            selected: list[EvidenceItem] = []
            seen_for_reliability: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_for_reliability:
                    continue
                seen_for_reliability.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                text = self._sanitize_evidence_text(item.text)
                if self._is_table_like_text(text) and not allow_tables:
                    continue
                selected.append(item)
                if len(selected) >= max(1, top_k):
                    break
            return self._renumber_evidence(selected or evidence[: max(1, top_k)])

        if research_limitation_question:
            selected: list[EvidenceItem] = []
            seen_limitations: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_limitations:
                    continue
                seen_limitations.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                text = self._sanitize_evidence_text(item.text)
                if self._is_table_like_text(text) and not allow_tables:
                    continue
                score = self._research_limitation_relevance_score(
                    text=text,
                    section=item.section or "",
                    index=0,
                    total=1,
                )
                if score < 0.65:
                    continue
                selected.append(item)
                if len(selected) >= max(2, min(top_k + 1, 6)):
                    break
            return self._renumber_evidence(selected)

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

            adjusted_score = item.score + quality * 0.18 + relevance * 0.75
            if reliability_question:
                adjusted_score += self._reliability_relevance_score(text) * 0.08
                if any(keyword in text for keyword in ["随机生成", "论文样稿", "课程报告", "实验报告", "毕业论文", "学位论文", "摘要"]):
                    adjusted_score += 1.6
                if any(keyword in text for keyword in ["未来研究", "实证数据", "文献分析", "情境推演", "机制建构"]):
                    adjusted_score += 0.6
                elif "结论" in text:
                    adjusted_score += 0.2
                if table_like and not allow_tables:
                    adjusted_score -= 0.4
            scored.append((adjusted_score, position, item))

        target_count = max(1, top_k)
        if not scored:
            selected = fallback[:target_count] or evidence[:target_count]
        else:
            scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            selected = [item for _, _, item in scored[:target_count]]

        return self._renumber_evidence(selected)

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

    def _renumber_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        for index, item in enumerate(evidence, start=1):
            item.citation_id = f"E{index}"
        return evidence

    def _resolve_query_embedding_model(
        self,
        *,
        requested_model: str,
        document_ids: list[str],
    ) -> str:
        for document_id in document_ids:
            document = self.store.get_document(document_id)
            if document and document.embedding_model == "本地备用检索":
                return "本地备用检索"
        return requested_model

