from __future__ import annotations

from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalEvidenceMixin:
    def _is_field_evidence_item(self, item: EvidenceItem) -> bool:
        return (item.section or "") in {"关键词", "摘要", "作者", "作者单位/机构", "日期", "标题"}

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
            page_start=int(metadata.get("page_start", metadata.get("page", 0)) or 0),
            page_end=int(metadata.get("page_end", metadata.get("page", 0)) or 0),
            section=label,
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=1.0,
            rule_score=1.0,
            final_score=1.0,
            score_source="field_rule",
            text=text,
            quote=text,
            char_start=int(metadata.get("char_start", 0) or 0),
            char_end=int(metadata.get("char_end", 0) or 0),
            token_count=int(metadata.get("token_count", 0) or 0),
            chunk_type=str(metadata.get("chunk_type") or "field"),
            parent_id=str(metadata.get("parent_id") or "") or None,
        )

    def _evidence_from_row(
        self,
        row: dict[str, Any],
        fallback_document_id: str,
        *,
        score: float,
        vector_score: float | None = None,
        sparse_score: float | None = None,
        rule_score: float | None = None,
        rrf_score: float | None = None,
        final_score: float | None = None,
        score_source: str = "rule",
    ) -> EvidenceItem:
        metadata = row["metadata"]
        text = str(metadata.get("parent_text") or row["text"])
        quote = self._best_readable_quote(str(metadata.get("quote", "")) or text)
        resolved_final_score = final_score if final_score is not None else score
        resolved_rule_score = rule_score
        if (
            resolved_rule_score is None
            and vector_score is None
            and sparse_score is None
            and rrf_score is None
        ):
            resolved_rule_score = score
        return EvidenceItem(
            citation_id="",
            chunk_id=str(metadata.get("chunk_id") or row["id"]),
            document_id=str(metadata.get("document_id", fallback_document_id)),
            paper_name=str(metadata.get("paper_name", "")),
            page=int(metadata.get("page", 0) or 0),
            page_start=int(metadata.get("parent_page_start", metadata.get("page_start", metadata.get("page", 0))) or 0),
            page_end=int(metadata.get("parent_page_end", metadata.get("page_end", metadata.get("page", 0))) or 0),
            section=str(metadata.get("section") or ""),
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=resolved_final_score,
            vector_score=vector_score,
            sparse_score=sparse_score,
            rule_score=resolved_rule_score,
            rrf_score=rrf_score,
            final_score=resolved_final_score,
            score_source=score_source,
            text=text,
            quote=quote,
            char_start=int(metadata.get("parent_char_start", metadata.get("char_start", 0)) or 0),
            char_end=int(metadata.get("parent_char_end", metadata.get("char_end", 0)) or 0),
            token_count=int(metadata.get("parent_token_count", metadata.get("token_count", 0)) or 0),
            chunk_type=str(metadata.get("chunk_type") or "text"),
            parent_id=str(metadata.get("parent_id") or "") or None,
        )

    def _renumber_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        for index, item in enumerate(evidence, start=1):
            item.citation_id = f"E{index}"
        return evidence
