from __future__ import annotations

import re
from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalFieldRuleMixin:
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
