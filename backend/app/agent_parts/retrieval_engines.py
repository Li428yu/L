from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from backend.app.models import EvidenceItem


class AgentRetrievalEngineMixin:
    def _bm25_sparse_evidence(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        query_tokens = self._bm25_query_tokens(question=question, soft_intent=soft_intent)
        if not query_tokens:
            return []

        rows_with_document: list[tuple[str, int, dict[str, Any]]] = []
        tokenized_rows: list[list[str]] = []
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            for index, row in enumerate(rows):
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                if not text.strip():
                    continue
                penalty = self._evidence_noise_penalty_for_soft_intent(
                    question=question,
                    text=text,
                    section=str((row.get("metadata") or {}).get("section") or ""),
                    soft_intent=soft_intent,
                )
                if penalty >= 2.2:
                    continue
                rows_with_document.append((document_id, index, row))
                tokenized_rows.append(self._bm25_tokens(text))

        if not rows_with_document:
            return []

        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_rows:
            doc_freq.update(set(tokens))
        avg_doc_len = sum(len(tokens) for tokens in tokenized_rows) / max(len(tokenized_rows), 1)
        document_count = len(tokenized_rows)
        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, ((document_id, index, row), tokens) in enumerate(zip(rows_with_document, tokenized_rows)):
            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=tokens,
                document_frequencies=doc_freq,
                document_count=document_count,
                avg_doc_len=avg_doc_len,
            )
            if score <= 0:
                continue
            metadata = row.get("metadata") or {}
            section = str(metadata.get("section") or "")
            text = self._sanitize_evidence_text(str(row.get("text", "")))
            role_score = self._soft_role_score(
                soft_intent=soft_intent,
                text=text,
                section=section,
                index=index,
                total=document_count,
            )
            focus_score = self._soft_focus_score(soft_intent, text)
            score += role_score * 0.8 + focus_score * 0.55
            if self._is_table_like_text(text) and not self._looks_like_table_question(question):
                score -= 1.1
            sparse_score = self._normalize_sparse_score(score)
            item = self._evidence_from_row(
                row,
                document_id,
                score=sparse_score,
                sparse_score=sparse_score,
                final_score=sparse_score,
                score_source="bm25_sparse",
            )
            item.quote = self._best_quote_for_question(question, item.text)
            scored.append((score, position, item))

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def _bm25_query_tokens(self, *, question: str, soft_intent: dict[str, Any]) -> list[str]:
        weighted_terms: list[str] = []
        weighted_terms.extend(self._question_keywords(question))
        for item in soft_intent.get("focus", []):
            weighted_terms.extend(self._bm25_tokens(str(item)))
            weighted_terms.extend(self._bm25_tokens(str(item)))
        for role in soft_intent.get("preferred_roles", []):
            weighted_terms.extend(self._bm25_role_terms(str(role)))

        cleaned: list[str] = []
        seen_counts: Counter[str] = Counter()
        for token in weighted_terms:
            if len(token) < 2:
                continue
            if seen_counts[token] >= 3:
                continue
            seen_counts[token] += 1
            cleaned.append(token)
            if len(cleaned) >= 80:
                break
        return cleaned

    def _bm25_role_terms(self, role: str) -> list[str]:
        role_terms = {
            "purpose": ["摘要", "目的", "本文", "研究目的", "in", "paper"],
            "approach": ["方法", "采用", "通过", "实验", "模型", "method"],
            "claim": ["结果", "发现", "表明", "提出", "认为", "result"],
            "conclusion": ["结论", "总结", "展望", "启示", "conclusion"],
            "caveat": ["局限", "不足", "风险", "挑战", "limitation"],
            "reference": ["参考文献", "references"],
            "field": ["摘要", "关键词", "作者", "标题"],
        }
        return role_terms.get(role, [])

    def _bm25_tokens(self, text: str) -> list[str]:
        normalized = self._sanitize_evidence_text(text).lower()
        tokens = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,}", normalized)
        expanded: list[str] = []
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
                if len(token) <= 4:
                    expanded.append(token)
                expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
                if len(token) >= 4:
                    expanded.extend(token[index : index + 3] for index in range(len(token) - 2))
            else:
                expanded.append(token)
        blocked = {"这个", "那个", "论文", "文档", "报告", "内容", "主要", "一下", "请问"}
        return [token for token in expanded if len(token) >= 2 and token not in blocked][:1600]

    def _bm25_score(
        self,
        *,
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequencies: Counter[str],
        document_count: int,
        avg_doc_len: float,
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        term_counts = Counter(document_tokens)
        k1 = 1.5
        b = 0.75
        doc_len = len(document_tokens)
        score = 0.0
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf <= 0:
                continue
            df = document_frequencies.get(token, 0)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
            score += idf * (tf * (k1 + 1)) / max(denominator, 1e-9)
        return score

    def _normalize_sparse_score(self, score: float) -> float:
        if score <= 0:
            return 0.0
        return max(0.0, min(1.0, score / (score + 3.0)))

    def _rrf_fuse_evidence_candidates(
        self,
        *,
        candidate_lists: list[list[EvidenceItem]],
        weights: list[float],
        limit: int,
    ) -> list[EvidenceItem]:
        rrf_k = 60.0
        fused_scores: dict[str, float] = {}
        best_by_key: dict[str, EvidenceItem] = {}
        score_parts_by_key: dict[str, dict[str, float | None]] = {}
        first_seen: dict[str, int] = {}
        seen_order = 0
        for list_index, candidates in enumerate(candidate_lists):
            weight = weights[list_index] if list_index < len(weights) else 1.0
            for rank, item in enumerate(candidates, start=1):
                key = f"{item.document_id}:{item.chunk_id}"
                if key not in first_seen:
                    first_seen[key] = seen_order
                    seen_order += 1
                fused_scores[key] = fused_scores.get(key, 0.0) + weight / (rrf_k + rank)
                score_parts = score_parts_by_key.setdefault(
                    key,
                    {
                        "vector_score": None,
                        "sparse_score": None,
                        "rule_score": None,
                    },
                )
                self._merge_score_part(score_parts, "vector_score", item.vector_score)
                self._merge_score_part(score_parts, "sparse_score", item.sparse_score)
                self._merge_score_part(score_parts, "rule_score", item.rule_score)
                current = best_by_key.get(key)
                if current is None or item.score > current.score or (
                    self._candidate_has_embedded_noise(current)
                    and not self._candidate_has_embedded_noise(item)
                ):
                    best_by_key[key] = item

        ranked = sorted(
            fused_scores.items(),
            key=lambda row: (row[1], -first_seen[row[0]]),
            reverse=True,
        )
        if not ranked:
            return []
        max_score = max(score for _, score in ranked) or 1.0
        fused: list[EvidenceItem] = []
        for key, score in ranked[:limit]:
            item = best_by_key[key]
            rrf_score = min(1.0, score / max_score)
            final_score = max(item.score, rrf_score)
            score_parts = score_parts_by_key.get(key, {})
            fused.append(
                item.model_copy(
                    update={
                        "score": final_score,
                        "vector_score": score_parts.get("vector_score"),
                        "sparse_score": score_parts.get("sparse_score"),
                        "rule_score": score_parts.get("rule_score"),
                        "rrf_score": rrf_score,
                        "final_score": final_score,
                        "score_source": "rrf_fusion",
                    }
                )
            )
        return fused

    def _merge_score_part(
        self,
        score_parts: dict[str, float | None],
        key: str,
        value: float | None,
    ) -> None:
        if value is None:
            return
        current = score_parts.get(key)
        score_parts[key] = value if current is None else max(current, value)

    def _select_rrf_ranked_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        scored = sorted(
            [(item.score, position, item) for position, item in enumerate(evidence)],
            key=lambda row: (row[0], -row[1]),
            reverse=True,
        )
        selected: list[EvidenceItem] = []
        per_document_counts: dict[str, int] = {}
        for score, _, item in scored:
            document_count = per_document_counts.get(item.document_id, 0)
            if len(per_document_counts) > 1 and document_count >= max(2, limit // max(len(per_document_counts), 1) + 1):
                continue
            selected.append(
                item.model_copy(
                    update={
                        "score": max(0.0, min(1.0, score)),
                        "final_score": max(0.0, min(1.0, score)),
                        "score_source": item.score_source or "rrf_fusion",
                        "quote": self._best_quote_for_question(question, item.text),
                    }
                )
            )
            per_document_counts[item.document_id] = document_count + 1
            if len(selected) >= limit:
                break
        return selected or evidence[:limit]
