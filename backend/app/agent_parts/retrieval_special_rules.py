from __future__ import annotations

from backend.app.models import EvidenceItem


class AgentRetrievalSpecialRuleMixin:
    def _reference_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(8, top_k + 3))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[dict] = []
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
