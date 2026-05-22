from __future__ import annotations

from backend.app.models import EvidenceItem


class AgentRetrievalFilterMixin:
    def _filter_evidence_for_question(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []

        reference_question = self._looks_like_reference_question(question)
        field_lookup_question = self._looks_like_field_lookup_question(question)
        allow_tables = self._looks_like_table_question(question)
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
