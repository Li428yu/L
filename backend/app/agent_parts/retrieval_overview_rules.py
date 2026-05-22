from __future__ import annotations

from typing import Any

from backend.app.agent_parts.state import ReadingTaskContract
from backend.app.models import EvidenceItem


class AgentRetrievalOverviewRuleMixin:
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
                relevance = self._question_relevance_score(question, text)
                if keyword_hits == 0 and relevance < 0.08:
                    continue
                position_bonus = 0.12 if index <= 2 else 0.0
                overview_bonus = self._overview_structure_score(text)
                score = 0.45 + keyword_hits * 0.16 + relevance * 0.5 + position_bonus + overview_bonus
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
