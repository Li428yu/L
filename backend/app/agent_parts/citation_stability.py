from __future__ import annotations

import re
from collections import Counter

from backend.app.agent_parts.citation_profiles import CitationStabilityProfile, citation_stability_profiles
from backend.app.models import EvidenceItem


class AgentCitationStabilityMixin:
    def _stabilize_final_evidence_citations(
        self,
        *,
        question: str,
        selected: list[EvidenceItem],
        candidates: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        if not candidates:
            return selected[: max(1, limit)]

        profile = self._citation_stability_profile(question)
        if profile is None:
            return selected[: max(1, limit)]

        selected = list(selected)
        self._refresh_citation_stability_quotes(profile=profile, selected=selected)
        if selected and self._citation_item_supports_profile(profile=profile, item=selected[0]):
            return selected[: max(1, limit)]

        best = self._best_citation_stability_candidate(profile=profile, candidates=candidates)
        if best is None:
            return selected[: max(1, limit)]

        return self._insert_citation_stability_candidate(
            profile=profile,
            question=question,
            selected=selected,
            item=best,
            limit=max(1, limit),
        )

    def _citation_stability_profile(self, question: str) -> CitationStabilityProfile | None:
        normalized = self._citation_stability_normalize(question)
        for profile in citation_stability_profiles():
            if self._citation_groups_present(normalized, profile.question_groups):
                return profile
        return None

    def _best_citation_stability_candidate(
        self,
        *,
        profile: CitationStabilityProfile,
        candidates: list[EvidenceItem],
    ) -> EvidenceItem | None:
        scored: list[tuple[float, int, EvidenceItem]] = []
        seen: set[str] = set()
        for position, item in enumerate(candidates):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            text = self._citation_evidence_text(item)
            if not self._citation_groups_present(text, profile.evidence_groups):
                continue
            scored.append((self._citation_candidate_score(profile, item, text), position, item))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return scored[0][2] if scored else None

    def _insert_citation_stability_candidate(
        self,
        *,
        profile: CitationStabilityProfile,
        question: str,
        selected: list[EvidenceItem],
        item: EvidenceItem,
        limit: int,
    ) -> list[EvidenceItem]:
        self._set_citation_stability_quote(profile=profile, item=item)
        existing_index = next((index for index, candidate in enumerate(selected) if candidate.chunk_id == item.chunk_id), None)
        if existing_index is not None:
            result = [candidate for candidate in selected if candidate.chunk_id != item.chunk_id]
            return [item, *result][:limit]

        result = [candidate for candidate in selected if candidate.chunk_id != item.chunk_id]
        if len(result) < limit:
            return [item, *result][:limit]

        drop_index = self._citation_stability_drop_index(
            profile=profile,
            question=question,
            selected=result,
            incoming=item,
        )
        if drop_index is None:
            return result[:limit]

        result.pop(drop_index)
        return [item, *result][:limit]

    def _citation_stability_drop_index(
        self,
        *,
        profile: CitationStabilityProfile,
        question: str,
        selected: list[EvidenceItem],
        incoming: EvidenceItem,
    ) -> int | None:
        document_counts = Counter(item.document_id for item in selected if item.document_id)
        visual_question = self._citation_question_requires_visual_evidence(question)
        scored: list[tuple[int, float, int]] = []
        for index, item in enumerate(selected):
            if visual_question and self._citation_is_visual_item(item):
                continue
            if self._citation_drop_would_break_document_coverage(item, incoming, document_counts):
                continue
            text = self._citation_evidence_text(item)
            lacks_direct_support = not self._citation_groups_present(text, profile.evidence_groups)
            has_avoid_term = self._citation_has_any(text, profile.avoid_terms)
            if not lacks_direct_support and not has_avoid_term:
                continue
            priority = 2 if has_avoid_term else 1
            scored.append((priority, -self._citation_candidate_score(profile, item, text), index))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _set_citation_stability_quote(self, *, profile: CitationStabilityProfile, item: EvidenceItem) -> None:
        text = self._citation_clean_text(item.text or item.quote)
        focused_quote = getattr(self, "_keyword_focused_quote", None)
        if callable(focused_quote):
            quote = focused_quote(
                text=text,
                terms=list(profile.quote_terms),
                bonus_phrases=list(profile.quote_bonus),
                limit=profile.quote_limit,
            )
            if not self._citation_groups_present(self._citation_stability_normalize(quote), profile.evidence_groups):
                quote = self._citation_required_group_quote(text=text, profile=profile)
            item.quote = quote
            return
        item.quote = self._citation_required_group_quote(text=text, profile=profile)

    def _citation_required_group_quote(self, *, text: str, profile: CitationStabilityProfile) -> str:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", text)
            if part.strip()
        ]
        picked: list[str] = []
        for group in profile.evidence_groups:
            for sentence in sentences or [text]:
                if sentence in picked:
                    continue
                if self._citation_group_present(self._citation_stability_normalize(sentence), group):
                    picked.append(sentence)
                    break
        quote = " ".join(picked) if picked else text
        return self._citation_truncate(quote, limit=profile.quote_limit)

    def _citation_truncate(self, text: str, *, limit: int) -> str:
        truncator = getattr(self, "_truncate_readable_text", None)
        if callable(truncator):
            return truncator(text, limit=limit)
        return text[:limit]

    def _refresh_citation_stability_quotes(
        self,
        *,
        profile: CitationStabilityProfile,
        selected: list[EvidenceItem],
    ) -> None:
        for item in selected:
            if self._citation_item_supports_profile(profile=profile, item=item):
                self._set_citation_stability_quote(profile=profile, item=item)

    def _citation_item_supports_profile(
        self,
        *,
        profile: CitationStabilityProfile,
        item: EvidenceItem,
    ) -> bool:
        return self._citation_groups_present(self._citation_evidence_text(item), profile.evidence_groups)

    def _citation_candidate_score(
        self,
        profile: CitationStabilityProfile,
        item: EvidenceItem,
        normalized_text: str,
    ) -> float:
        score = float(item.score or 0.0)
        score += sum(1 for group in profile.evidence_groups if self._citation_group_present(normalized_text, group))
        score += sum(0.08 for term in profile.quote_terms if self._citation_alias_present(normalized_text, term))
        score += sum(0.18 for term in profile.quote_bonus if self._citation_alias_present(normalized_text, term))
        score -= sum(0.45 for term in profile.avoid_terms if self._citation_alias_present(normalized_text, term))
        if "table" in (item.chunk_type or "").lower():
            score += 0.2
        return score

    def _citation_drop_would_break_document_coverage(
        self,
        item: EvidenceItem,
        incoming: EvidenceItem,
        document_counts: Counter[str],
    ) -> bool:
        if not item.document_id or item.document_id == incoming.document_id:
            return False
        return len(document_counts) > 1 and document_counts[item.document_id] <= 1

    def _citation_question_requires_visual_evidence(self, question: str) -> bool:
        checker = getattr(self, "_question_requires_visual_evidence", None)
        if callable(checker):
            return bool(checker(question))
        visual_checker = getattr(self, "_looks_like_visual_evidence_question", None)
        return bool(callable(visual_checker) and visual_checker(question))

    def _citation_is_visual_item(self, item: EvidenceItem) -> bool:
        checker = getattr(self, "_is_visual_evidence_item", None)
        if callable(checker):
            return bool(checker(item))
        chunk_type = (item.chunk_type or "").lower()
        return bool(item.image_id) or any(marker in chunk_type for marker in ["image", "figure", "chart"])

    def _citation_groups_present(self, text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
        return all(self._citation_group_present(text, group) for group in groups)

    def _citation_group_present(self, text: str, group: tuple[str, ...]) -> bool:
        return any(self._citation_alias_present(text, alias) for alias in group)

    def _citation_has_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(self._citation_alias_present(text, term) for term in terms)

    def _citation_alias_present(self, normalized_text: str, alias: str) -> bool:
        normalized_alias = self._citation_stability_normalize(alias)
        if not normalized_alias:
            return False
        if normalized_alias in normalized_text:
            return True
        compact_text = re.sub(r"[\s,._-]+", "", normalized_text)
        compact_alias = re.sub(r"[\s,._-]+", "", normalized_alias)
        return bool(compact_alias and compact_alias in compact_text)

    def _citation_evidence_text(self, item: EvidenceItem) -> str:
        related = " ".join(
            " ".join(
                part
                for part in [
                    getattr(image, "caption_text", ""),
                    getattr(image, "ocr_text", ""),
                    getattr(image, "vision_summary", ""),
                ]
                if part
            )
            for image in getattr(item, "related_images", [])
        )
        return self._citation_stability_normalize(
            f"{item.paper_name} {item.section or ''} {item.quote or ''} {item.text or ''} {related}"
        )

    def _citation_clean_text(self, text: str) -> str:
        sanitizer = getattr(self, "_sanitize_evidence_text", None)
        if callable(sanitizer):
            return sanitizer(text)
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _citation_stability_normalize(self, text: str) -> str:
        normalized = self._citation_clean_text(text).lower()
        normalized = normalized.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        return re.sub(r"\s+", " ", normalized).strip()
