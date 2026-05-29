from __future__ import annotations

import re
from collections import Counter
from typing import Any

from backend.app.agent_parts.state import PaperAgentState
from backend.app.agent_parts.retrieval_engines import AgentRetrievalEngineMixin
from backend.app.agent_parts.retrieval_evidence import AgentRetrievalEvidenceMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.retrieval_quality import AgentRetrievalQualityMixin
from backend.app.agent_parts.retrieval_rules import AgentRetrievalRuleMixin
from backend.app.agent_parts.retrieval_scoring import AgentRetrievalScoringMixin
from backend.app.llm_clients import LOCAL_FALLBACK_EMBEDDING_PROVIDER
from backend.app.models import EvidenceItem, RuntimeStep


class AgentRetrievalMixin(
    AgentRetrievalEngineMixin,
    AgentRetrievalEvidenceMixin,
    AgentRetrievalScoringMixin,
    AgentRetrievalQualityMixin,
    AgentRetrievalRuleMixin,
    AgentRetrievalFilterMixin,
):
    def _retrieve(self, state: PaperAgentState) -> PaperAgentState:
        attempt = int(state.get("retrieval_attempts", 0) or 0)
        if attempt:
            self._emit_status("证据还不够稳，正在扩大范围重新检索...")
        else:
            self._emit_status("正在从当前文档里查找证据...")
        requested_document_ids = self._resolve_document_ids(state.get("document_ids"))
        document_ids, scope_reason = self._scope_document_ids_by_question(
            question=state["question"],
            document_ids=requested_document_ids,
        )
        strategy = state.get("retrieval_strategy") or "hybrid_soft"
        retrieval_queries = self._build_retrieval_queries(
            question=state["question"],
            soft_intent=state.get("soft_intent", {}),
            document_ids=document_ids,
        )
        candidate_evidence, retrieval_pipeline, ranking_method, embedding_trace = self._hybrid_evidence(
            question=state["question"],
            retrieval_queries=retrieval_queries,
            soft_intent=state.get("soft_intent", {}),
            document_ids=document_ids,
            top_k=state["top_k"],
            embedding_model=state["embedding_model"],
            retrieval_strategy=strategy,
        )
        evidence = self._filter_evidence_for_question(
            state["question"],
            candidate_evidence,
            top_k=state["top_k"],
            target_document_ids=document_ids,
        )
        evidence, evidence_quality_trace = self._annotate_retrieval_quality(
            question=state["question"],
            candidates=candidate_evidence,
            selected=evidence,
            top_k=state["top_k"],
            retrieval_strategy=strategy,
        )
        multi_document_cards = self._build_multi_document_cards(
            question=state["question"],
            evidence=evidence,
            target_document_ids=document_ids,
        )
        document_relation_map = self._infer_document_relations(
            question=state["question"],
            cards=multi_document_cards,
        )
        multi_document_coverage = self._build_multi_document_coverage(
            target_document_ids=document_ids,
            cards=multi_document_cards,
        )
        runtime_steps = [*state.get("runtime", [])]
        if scope_reason:
            runtime_steps.append(
                RuntimeStep(
                    node="document_scope",
                    title="识别题目点名文献",
                    detail=scope_reason,
                )
            )
        runtime_steps.append(
            RuntimeStep(
                    node="retrieval_agent",
                    title="检索 Agent 查找证据",
                detail=(
                    f"实际使用「{self._friendly_retrieval_strategy(strategy)}」，"
                    f"管线为 {retrieval_pipeline}，排序方式为 {ranking_method}；"
                    f"检索范围 {len(document_ids)} 篇文档，top-k 为 {state['top_k']}，"
                    f"这是第 {attempt + 1} 次检索，使用 {len(retrieval_queries)} 个子查询，"
                    f"返回 {len(evidence)} 条证据。"
                ),
            )
        )
        runtime_steps.append(
            RuntimeStep(
                node="evidence_quality_trace",
                title="标记证据质量",
                detail=self._quality_trace_summary(evidence_quality_trace),
            )
        )
        if len(document_ids) > 1:
            covered_count = int(multi_document_coverage.get("covered_document_count", 0) or 0)
            requested_count = int(multi_document_coverage.get("requested_document_count", len(document_ids)) or 0)
            missing_names = multi_document_coverage.get("missing_document_names", [])
            missing_detail = f"；缺少证据：{', '.join(missing_names)}" if missing_names else ""
            runtime_steps.append(
                RuntimeStep(
                    node="multi_document_relation",
                    title="整理多文献关系",
                    detail=(
                        f"已按文献生成 {len(multi_document_cards)} 张证据卡，"
                        f"覆盖 {covered_count}/{requested_count} 篇文档，"
                        f"推断 {len(document_relation_map)} 条文献关系{missing_detail}。"
                    ),
                )
            )
        return {
            **state,
            "evidence": evidence,
            "retrieval_strategy": strategy,
            "retrieval_queries": retrieval_queries,
            "retrieval_pipeline": retrieval_pipeline,
            "ranking_method": ranking_method,
            "embedding_trace": embedding_trace,
            "evidence_quality_trace": evidence_quality_trace,
            "fallback_used": bool(state.get("fallback_used", False)) or bool(embedding_trace.get("embedding_used_fallback")),
            "retrieval_document_ids": document_ids,
            "requested_document_ids": requested_document_ids,
            "multi_document_cards": multi_document_cards,
            "document_relation_map": document_relation_map,
            "multi_document_coverage": multi_document_coverage,
            "runtime": runtime_steps,
        }

    def _build_retrieval_queries(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
    ) -> list[str]:
        queries: list[str] = []

        def add(value: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        add(question)
        keyphrases = self._question_keyphrases(question)
        if keyphrases:
            add(f"{question} {' '.join(keyphrases[:6])}")
        for item in soft_intent.get("focus", [])[:4]:
            add(f"{question} {item}")

        for role in self._paper_structure_roles_for_question(question, soft_intent=soft_intent):
            terms = self._paper_structure_role_terms(role, kind="query")[:10]
            if terms:
                add(f"{question} {' '.join(terms)}")

        return queries[:6]

    def _scope_document_ids_by_question(
        self,
        *,
        question: str,
        document_ids: list[str],
    ) -> tuple[list[str], str]:
        if len(document_ids) <= 1:
            return document_ids, ""
        if self._looks_like_compare_question(question) or self._looks_like_multi_document_topic_question(question):
            return document_ids, ""

        matches: list[tuple[float, int, str, str]] = []
        for position, document_id in enumerate(document_ids):
            document = self.store.get_document(document_id)
            if not document:
                continue
            score = self._document_mention_score(question, document.file_name)
            if score >= 1.0:
                matches.append((score, position, document_id, document.file_name))

        if not matches:
            return document_ids, ""

        matches.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        best_score = matches[0][0]
        selected = [
            row
            for row in matches
            if row[0] >= 1.6 or row[0] >= max(1.0, best_score - 0.8)
        ]
        selected_ids = [row[2] for row in sorted(selected, key=lambda row: row[1])]
        if not selected_ids or len(selected_ids) == len(document_ids):
            return document_ids, ""

        names = [row[3] for row in sorted(selected, key=lambda row: row[1])]
        reason = (
            f"问题中明确提到 {', '.join(names)}，"
            f"本轮先把检索范围从 {len(document_ids)} 份文档收窄到 {len(selected_ids)} 份，"
            "避免未被点名的文献抢占证据位。"
        )
        return selected_ids, reason

    def _document_mention_score(self, question: str, file_name: str) -> float:
        normalized_question = self._normalize_document_mention_text(question)
        if not normalized_question:
            return 0.0

        stem = re.sub(r"\.[A-Za-z0-9]+$", "", file_name)
        normalized_stem = self._normalize_document_mention_text(stem)
        compact_question = normalized_question.replace(" ", "")
        compact_stem = normalized_stem.replace(" ", "")
        score = 0.0

        if normalized_stem and normalized_stem in normalized_question:
            score += 5.0
        elif compact_stem and len(compact_stem) >= 8 and compact_stem in compact_question:
            score += 4.5

        for alias, weight in self._document_aliases(file_name):
            normalized_alias = self._normalize_document_mention_text(alias)
            compact_alias = normalized_alias.replace(" ", "")
            if normalized_alias and normalized_alias in normalized_question:
                score += weight
            elif compact_alias and len(compact_alias) >= 4 and compact_alias in compact_question:
                score += max(1.0, weight - 0.2)

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized_stem)
            if token not in self._document_mention_stopwords()
        ]
        token_hits = sum(1 for token in set(tokens) if token in normalized_question)
        if token_hits >= 2:
            score += min(1.4, token_hits * 0.35)

        return score

    def _document_aliases(self, file_name: str) -> list[tuple[str, float]]:
        aliases: list[tuple[str, float]] = []

        def add(alias: str, weight: float = 1.8) -> None:
            if alias and alias not in {value for value, _ in aliases}:
                aliases.append((alias, weight))

        stem = re.sub(r"\.[A-Za-z0-9]+$", "", file_name)
        normalized_stem = self._normalize_document_mention_text(stem)
        add(normalized_stem, 2.2)
        compact_stem = normalized_stem.replace(" ", "")
        if len(compact_stem) >= 8:
            add(compact_stem, 1.8)

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized_stem)
            if token not in self._document_mention_stopwords()
        ]
        if 2 <= len(tokens) <= 8:
            initials = "".join(token[0] for token in tokens if re.match(r"[a-z]", token))
            if 2 <= len(initials) <= 10:
                add(initials, 1.4)
        for size in range(2, min(4, len(tokens)) + 1):
            for start in range(0, len(tokens) - size + 1):
                add(" ".join(tokens[start : start + size]), 1.2)

        return aliases

    def _normalize_document_mention_text(self, text: str) -> str:
        normalized = str(text).lower()
        normalized = re.sub(r"[_\-–—/\\.:：,，;；()（）\[\]【】《》“”\"'!?！？]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _document_mention_stopwords(self) -> set[str]:
        return {
            "pdf",
            "docx",
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "all",
            "you",
            "need",
            "learning",
            "models",
            "model",
            "pretraining",
            "framework",
            "network",
            "programming",
        }

    def _build_multi_document_cards(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        target_document_ids: list[str],
    ) -> list[dict[str, Any]]:
        if len(target_document_ids) < 2:
            return []

        grouped: dict[str, list[EvidenceItem]] = {document_id: [] for document_id in target_document_ids}
        for item in evidence:
            if item.document_id not in grouped:
                grouped[item.document_id] = []
            grouped[item.document_id].append(item)

        cards: list[dict[str, Any]] = []
        for document_id in dict.fromkeys([*target_document_ids, *grouped.keys()]):
            items = grouped.get(document_id, [])
            paper_name = self._document_display_name(document_id, items)
            best_quote = self._best_multi_document_quote(question=question, evidence=items)
            role_rows = self._multi_document_role_rows(items)
            key_terms = self._multi_document_key_terms(question=question, evidence=items, paper_name=paper_name)
            pages = list(
                dict.fromkeys(
                    self._evidence_page_label(item)
                    for item in items
                    if self._evidence_page_label(item) and self._evidence_page_label(item) != "0"
                )
            )
            citation_ids = [item.citation_id for item in items if item.citation_id]
            evidence_types = list(dict.fromkeys(item.chunk_type or "text" for item in items))
            image_evidence_count = sum(
                1
                for item in items
                if item.image_id or "image" in (item.chunk_type or "").lower() or "figure" in (item.chunk_type or "").lower()
            )
            cards.append(
                {
                    "document_id": document_id,
                    "paper_name": paper_name,
                    "covered": bool(items),
                    "evidence_count": len(items),
                    "citation_ids": citation_ids[:6],
                    "pages": pages[:8],
                    "key_terms": key_terms[:8],
                    "roles": role_rows[:5],
                    "best_quote": best_quote,
                    "evidence_types": evidence_types[:5],
                    "image_evidence_count": image_evidence_count,
                }
            )
        return cards

    def _document_display_name(self, document_id: str, evidence: list[EvidenceItem]) -> str:
        if evidence and evidence[0].paper_name:
            return evidence[0].paper_name
        document = self.store.get_document(document_id)
        return document.file_name if document else document_id

    def _best_multi_document_quote(self, *, question: str, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return ""
        scored: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            text = self._sanitize_evidence_text(item.text)
            score = (
                item.score
                + self._question_relevance_score(question, text) * 0.7
                + self._readable_text_score(text) * 0.2
            )
            scored.append((score, position, item))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        best = scored[0][2]
        return self._truncate_readable_text(best.quote or self._best_quote_for_question(question, best.text), limit=260)

    def _multi_document_role_rows(self, evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
        if not evidence:
            return []
        role_totals: Counter[str] = Counter()
        total = max(len(evidence), 1)
        for index, item in enumerate(evidence):
            role_scores = self._semantic_role_scores(
                text=item.text,
                section=item.section or "",
                index=index,
                total=total,
            )
            for role, score in role_scores.items():
                role_totals[role] += float(score)
        role_labels = {
            "purpose": "研究目的/主题",
            "approach": "方法/设计",
            "claim": "发现/主张",
            "conclusion": "结论/启示",
            "caveat": "局限/风险",
            "example": "案例/场景",
            "informative": "背景信息",
        }
        return [
            {
                "role": role,
                "label": role_labels.get(role, role),
                "score": round(score, 3),
            }
            for role, score in role_totals.most_common()
        ]

    def _multi_document_key_terms(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        paper_name: str,
    ) -> list[str]:
        text = self._sanitize_evidence_text(" ".join([paper_name, *[item.quote or item.text for item in evidence]]))
        normalized = text.lower()
        terms: list[str] = []
        for term in self._question_keywords(question)[:16]:
            lowered = term.lower()
            if (term in text or lowered in normalized) and term not in terms:
                terms.append(term)

        english_blocked = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "are",
            "was",
            "were",
            "paper",
            "study",
            "research",
        }
        english_counts = Counter(
            token.lower()
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", text)
            if token.lower() not in english_blocked
        )
        for token, _ in english_counts.most_common(8):
            if token not in terms:
                terms.append(token)

        cjk_counts = Counter()
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,12}", text):
            if len(sequence) <= 8:
                cjk_counts[sequence] += 1
                continue
            for index in range(max(0, len(sequence) - 3)):
                cjk_counts[sequence[index : index + 4]] += 1
        blocked_cjk = {"本文", "研究", "文档", "论文", "内容", "主要", "通过", "进行", "可以", "用户"}
        for token, _ in cjk_counts.most_common(12):
            if token in blocked_cjk or token in terms:
                continue
            terms.append(token)
            if len(terms) >= 10:
                break
        return terms[:10]

    def _build_multi_document_coverage(
        self,
        *,
        target_document_ids: list[str],
        cards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(target_document_ids) < 2:
            return {}
        card_by_id = {str(card.get("document_id")): card for card in cards}
        per_document: list[dict[str, Any]] = []
        missing_document_ids: list[str] = []
        missing_document_names: list[str] = []
        for document_id in target_document_ids:
            card = card_by_id.get(document_id) or {}
            covered = bool(card.get("covered"))
            name = str(card.get("paper_name") or self._document_display_name(document_id, []))
            per_document.append(
                {
                    "document_id": document_id,
                    "paper_name": name,
                    "covered": covered,
                    "evidence_count": int(card.get("evidence_count", 0) or 0),
                    "citation_ids": card.get("citation_ids", []),
                }
            )
            if not covered:
                missing_document_ids.append(document_id)
                missing_document_names.append(name)

        requested_count = len(target_document_ids)
        covered_count = requested_count - len(missing_document_ids)
        return {
            "requested_document_count": requested_count,
            "covered_document_count": covered_count,
            "coverage_ratio": round(covered_count / max(requested_count, 1), 3),
            "missing_document_ids": missing_document_ids,
            "missing_document_names": missing_document_names,
            "per_document": per_document,
        }

    def _infer_document_relations(
        self,
        *,
        question: str,
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        covered_cards = [card for card in cards if card.get("covered")]
        if len(cards) < 2:
            return []
        relation_rows: list[dict[str, Any]] = []
        compare_intent = self._looks_like_compare_question(question)
        consistency_intent = any(keyword in question for keyword in ["一致", "冲突", "矛盾", "相反", "支持", "反驳", "互证"])
        synthesis_intent = self._looks_like_multi_document_topic_question(question)

        cards_for_pairs = covered_cards if len(covered_cards) >= 2 else cards
        for left_index, left in enumerate(cards_for_pairs):
            for right in cards_for_pairs[left_index + 1 :]:
                left_terms = set(str(term) for term in left.get("key_terms", []))
                right_terms = set(str(term) for term in right.get("key_terms", []))
                shared_terms = list(left_terms & right_terms)[:6]
                left_roles = {str(role.get("role")) for role in left.get("roles", []) if isinstance(role, dict)}
                right_roles = {str(role.get("role")) for role in right.get("roles", []) if isinstance(role, dict)}

                if not left.get("covered") or not right.get("covered"):
                    relation_type = "evidence_missing"
                    relation_label = "证据缺失，暂不能建立可靠关系"
                elif consistency_intent:
                    relation_type = "consistency_check"
                    relation_label = "需要核对一致、冲突或互证关系"
                elif compare_intent:
                    relation_type = "comparison"
                    relation_label = "适合并列比较差异和共同点"
                elif shared_terms:
                    relation_type = "shared_topic"
                    relation_label = "共同关注同一主题"
                elif left_roles != right_roles:
                    relation_type = "complementary"
                    relation_label = "证据角色互补"
                elif synthesis_intent:
                    relation_type = "topic_synthesis"
                    relation_label = "可纳入同一选题综合"
                else:
                    relation_type = "parallel"
                    relation_label = "并列材料，需分别引用"

                relation_rows.append(
                    {
                        "source_document_id": str(left.get("document_id", "")),
                        "target_document_id": str(right.get("document_id", "")),
                        "source_name": str(left.get("paper_name", "")),
                        "target_name": str(right.get("paper_name", "")),
                        "relation_type": relation_type,
                        "relation_label": relation_label,
                        "shared_terms": shared_terms,
                        "source_citations": list(left.get("citation_ids", []))[:3],
                        "target_citations": list(right.get("citation_ids", []))[:3],
                        "summary": self._format_document_relation_summary(
                            left=left,
                            right=right,
                            relation_label=relation_label,
                            shared_terms=shared_terms,
                        ),
                    }
                )
                if len(relation_rows) >= 12:
                    return relation_rows
        return relation_rows

    def _format_document_relation_summary(
        self,
        *,
        left: dict[str, Any],
        right: dict[str, Any],
        relation_label: str,
        shared_terms: list[str],
    ) -> str:
        left_name = str(left.get("paper_name", "文献A"))
        right_name = str(right.get("paper_name", "文献B"))
        shared = f"共同关注：{', '.join(shared_terms[:4])}" if shared_terms else "暂无明显共同关键词"
        return f"{left_name} 与 {right_name}：{relation_label}；{shared}。"

    def _looks_like_multi_document_topic_question(self, question: str) -> bool:
        keywords = [
            "选题",
            "多篇",
            "多份",
            "几篇",
            "这些文献",
            "全部文献",
            "所有文献",
            "同时纳入",
            "一起分析",
            "综合",
            "归纳",
            "综述",
            "文献综述",
            "研究现状",
            "共同",
            "关系",
            "关联",
            "互相",
            "异同",
            "脉络",
        ]
        return any(keyword in question for keyword in keywords)

    def _route_after_evidence_judge(self, state: PaperAgentState) -> str:
        if not state.get("needs_retrieval"):
            return "answer"
        if int(state.get("retrieval_attempts", 0) or 0) >= 1:
            return "answer"
        if self._should_retry_retrieval(state):
            return "retry_retrieve"
        return "answer"

    def _should_retry_retrieval(self, state: PaperAgentState) -> bool:
        question = state["question"]
        if self._looks_like_meta_question(question):
            return False
        evidence = state.get("evidence", [])
        if not evidence:
            return True
        evidence_quality = state.get("evidence_quality") or self._evidence_quality(
            question=question,
            evidence=evidence,
            fallback_used=False,
            answer_strategy="model_answer",
        )
        if evidence_quality == "weak":
            return True
        if evidence_quality != "strong" and len(evidence) == 1 and not any(
            checker(question)
            for checker in [
                self._looks_like_field_lookup_question,
                self._looks_like_reference_question,
                self._looks_like_broad_overview_question,
            ]
        ):
            return True
        return False

    def _refine_retrieval(self, state: PaperAgentState) -> PaperAgentState:
        self._emit_status("正在调整检索策略...")
        attempts = int(state.get("retrieval_attempts", 0) or 0) + 1
        original_top_k = int(state.get("top_k", 5) or 5)
        expanded_top_k = min(max(original_top_k * 2, original_top_k + 4), 14)
        soft_intent = dict(state.get("soft_intent") or {})
        question_keywords = self._question_keywords(state["question"])[:6]
        focus = list(dict.fromkeys([*soft_intent.get("focus", []), *question_keywords]))
        roles = list(soft_intent.get("preferred_roles", []))
        if not roles:
            roles = ["purpose", "approach", "claim", "conclusion", "caveat"]
        soft_intent.update(
            {
                "focus": focus[:6],
                "preferred_roles": roles,
                "reason": "首轮证据不足，已扩大候选数量并放宽检索关注点后重试。",
                "source": soft_intent.get("source", "local"),
            }
        )
        detail = (
            f"首轮证据质量为「{state.get('evidence_quality') or 'unknown'}」，"
            f"将 top-k 从 {original_top_k} 扩大到 {expanded_top_k}，并用问题关键词补充检索关注点。"
        )
        return {
            **state,
            "top_k": expanded_top_k,
            "retrieval_attempts": attempts,
            "retrieval_strategy": "hybrid_retry",
            "soft_intent": soft_intent,
            "evidence": [],
            "evidence_quality_trace": [],
            "evidence_judgments": [],
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="retrieval_agent",
                    title="检索 Agent 调整策略",
                    detail=detail,
                ),
            ],
        }

    def _hybrid_evidence(
        self,
        *,
        question: str,
        retrieval_queries: list[str] | None = None,
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
        retrieval_strategy: str,
    ) -> tuple[list[EvidenceItem], str, str, dict[str, Any]]:
        embedding_events: list[dict[str, Any]] = []
        if not document_ids:
            return (
                [],
                "dense_vector + bm25_sparse -> rrf_fusion",
                "none",
                self._build_embedding_trace(
                    requested_model=embedding_model,
                    document_ids=document_ids,
                    query_events=embedding_events,
                ),
            )
        if self._should_balance_multi_document_retrieval(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
        ):
            ranked = self._multi_document_hybrid_evidence(
                question=question,
                retrieval_queries=retrieval_queries or [question],
                soft_intent=soft_intent,
                document_ids=document_ids,
                top_k=top_k,
                embedding_model=embedding_model,
                embedding_events=embedding_events,
            )
            return (
                self._renumber_evidence(ranked),
                "per_document_structured + per_document_multi_query_dense + per_document_multi_query_bm25 -> balanced_rrf_fusion -> evidence_expansion",
                "按文献配额的 RRF 融合排序",
                self._build_embedding_trace(
                    requested_model=embedding_model,
                    document_ids=document_ids,
                    query_events=embedding_events,
                ),
            )

        query_list = retrieval_queries or [question]
        vector_candidates_by_query = self._multi_query_vector_evidence(
            retrieval_queries=query_list,
            document_ids=document_ids,
            top_k=max(top_k * 8, 24),
            embedding_model=embedding_model,
            embedding_events=embedding_events,
            max_queries=4,
        )
        candidate_lists: list[list[EvidenceItem]] = []
        weights: list[float] = []
        used_targeted = False
        for query_index, query in enumerate(query_list):
            targeted = self._targeted_evidence_candidates(
                question=query,
                soft_intent=soft_intent,
                document_ids=document_ids,
                top_k=max(top_k, 5),
            )
            if targeted:
                used_targeted = True
            vector_candidates = vector_candidates_by_query.get(self._retrieval_query_key(query), [])
            sparse_candidates = self._bm25_sparse_evidence(
                question=query,
                soft_intent=soft_intent,
                document_ids=document_ids,
                top_k=max(top_k * 8, 24),
            )
            query_weight = 1.0 if query_index == 0 else 0.82
            candidate_lists.extend([targeted, vector_candidates, sparse_candidates])
            weights.extend([1.15 * query_weight, 1.0 * query_weight, 1.0 * query_weight])
        fused = self._rrf_fuse_evidence_candidates(
            candidate_lists=candidate_lists,
            weights=weights,
            limit=max(top_k * 8, 24),
        )
        expanded = self._expand_retrieval_candidates(
            question=question,
            candidates=fused,
            limit=max(top_k * 12, 36),
        )
        ranked = self._select_rrf_ranked_evidence(
            question=question,
            evidence=expanded,
            limit=max(top_k * 4, 8),
        )
        return (
            self._renumber_evidence(ranked),
            (
                "structured_candidates + multi_query_dense_vector + multi_query_bm25_sparse -> rrf_fusion -> evidence_expansion"
                if used_targeted
                else "multi_query_dense_vector + multi_query_bm25_sparse -> rrf_fusion -> evidence_expansion"
            ),
            "多查询 RRF 融合排序",
            self._build_embedding_trace(
                requested_model=embedding_model,
                document_ids=document_ids,
                query_events=embedding_events,
            ),
        )

    def _multi_query_vector_evidence(
        self,
        *,
        retrieval_queries: list[str],
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
        embedding_events: list[dict[str, Any]] | None = None,
        max_queries: int = 4,
    ) -> dict[str, list[EvidenceItem]]:
        results: dict[str, list[EvidenceItem]] = {}
        if not retrieval_queries or max_queries <= 0:
            return results

        seen: set[str] = set()
        for query in retrieval_queries:
            query_key = self._retrieval_query_key(query)
            if not query_key or query_key in seen:
                continue
            if len(seen) >= max_queries:
                break
            seen.add(query_key)
            results[query_key] = self._vector_similarity_evidence(
                question=query_key,
                document_ids=document_ids,
                top_k=top_k,
                embedding_model=embedding_model,
                embedding_events=embedding_events,
            )
        return results

    def _retrieval_query_key(self, query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "")).strip()

    def _expand_retrieval_candidates(
        self,
        *,
        question: str,
        candidates: list[EvidenceItem],
        limit: int,
    ) -> list[EvidenceItem]:
        if not candidates:
            return []

        limit = max(limit, len(candidates))
        expanded: list[EvidenceItem] = list(candidates)
        seen_keys = {f"{item.document_id}:{item.chunk_id}" for item in candidates}
        seen_texts = {
            self._expansion_text_key(item.text or item.quote)
            for item in candidates
            if self._expansion_text_key(item.text or item.quote)
        }
        rows_by_document: dict[str, list[dict[str, Any]]] = {}
        anchors = candidates[: min(len(candidates), 12)]
        local_limit = min(limit, max(len(candidates), int(limit * 0.65)))

        for anchor_position, anchor in enumerate(anchors):
            if not anchor.document_id or self._is_visual_evidence_item(anchor):
                continue
            rows = rows_by_document.get(anchor.document_id)
            if rows is None:
                rows = self.vector_store.get_document_chunks(anchor.document_id, limit=1000)
                rows_by_document[anchor.document_id] = rows
            if not rows:
                continue

            current_index = self._expansion_row_index(rows=rows, chunk_id=anchor.chunk_id)
            if current_index < 0:
                continue

            for row_index, proximity_bonus in self._expansion_row_indices(
                rows=rows,
                current_index=current_index,
                anchor=anchor,
            ):
                row = rows[row_index]
                for item in self._evidence_units_from_expansion_row(
                    question=question,
                    row=row,
                    fallback_document_id=anchor.document_id,
                    anchor=anchor,
                    row_index=row_index,
                    anchor_position=anchor_position,
                    proximity_bonus=proximity_bonus,
                ):
                    key = f"{item.document_id}:{item.chunk_id}"
                    text_key = self._expansion_text_key(item.text)
                    if key in seen_keys or (text_key and text_key in seen_texts):
                        continue
                    if item.score < 0.58:
                        continue
                    expanded.append(item)
                    seen_keys.add(key)
                    if text_key:
                        seen_texts.add(text_key)
                    if len(expanded) >= local_limit:
                        break
                if len(expanded) >= local_limit:
                    break
            if len(expanded) >= local_limit:
                break

        if len(expanded) < limit:
            expanded.extend(
                self._document_level_expansion_candidates(
                    question=question,
                    anchors=anchors,
                    rows_by_document=rows_by_document,
                    seen_keys=seen_keys,
                    seen_texts=seen_texts,
                    limit=limit - len(expanded),
                )
            )

        scored = sorted(
            [
                (item.score, self._expansion_granularity_bonus(item), position, item)
                for position, item in enumerate(expanded)
            ],
            key=lambda row: (row[0], row[1], -row[2]),
            reverse=True,
        )
        return [item for _, _, _, item in scored[:limit]]

    def _expansion_granularity_bonus(self, item: EvidenceItem) -> float:
        chunk_type = str(item.chunk_type or "").lower()
        score_source = str(item.score_source or "").lower()
        if chunk_type == "table_row" or score_source == "expanded_table_row":
            return 0.035
        if chunk_type == "phrase" or score_source == "expanded_phrase":
            return 0.03
        if chunk_type == "sentence" or score_source == "expanded_sentence":
            return 0.02
        return 0.0

    def _document_level_expansion_candidates(
        self,
        *,
        question: str,
        anchors: list[EvidenceItem],
        rows_by_document: dict[str, list[dict[str, Any]]],
        seen_keys: set[str],
        seen_texts: set[str],
        limit: int,
    ) -> list[EvidenceItem]:
        if limit <= 0:
            return []

        document_order: list[str] = []
        best_anchor_by_document: dict[str, EvidenceItem] = {}
        for anchor in anchors:
            if not anchor.document_id or self._is_visual_evidence_item(anchor):
                continue
            if anchor.document_id not in document_order:
                document_order.append(anchor.document_id)
            current = best_anchor_by_document.get(anchor.document_id)
            if current is None or float(anchor.score or 0.0) > float(current.score or 0.0):
                best_anchor_by_document[anchor.document_id] = anchor
        if not document_order:
            return []

        selected: list[EvidenceItem] = []
        per_document_limit = max(2, min(6, limit // max(len(document_order), 1) + 2))
        mechanism_priority = self._looks_like_mechanism_expansion_question(question)

        for document_position, document_id in enumerate(document_order):
            rows = rows_by_document.get(document_id)
            if rows is None:
                rows = self.vector_store.get_document_chunks(document_id, limit=1000)
                rows_by_document[document_id] = rows
            if not rows:
                continue

            anchor = best_anchor_by_document.get(document_id)
            row_candidates: list[tuple[float, float, float, int, int, str, str, dict[str, Any]]] = []
            for row_index, row in enumerate(rows):
                metadata = row.get("metadata") or {}
                chunk_type = str(metadata.get("chunk_type") or "").lower()
                if "image" in chunk_type:
                    continue
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                if not text.strip():
                    continue

                units: list[tuple[float, int, str, str]] = []
                if self._is_table_like_text(text) or "table" in chunk_type:
                    units.extend(
                        self._table_row_units_for_expansion(
                            question=question,
                            text=text,
                            max_units=3,
                        )
                    )
                units.extend(
                    self._sentence_units_for_expansion(
                        question=question,
                        text=text,
                        max_units=5,
                    )
                )
                quote = self._best_quote_for_question(question, text, limit=420)
                if quote:
                    quote_score = self._evidence_unit_support_score(
                        question=question,
                        text=quote,
                        anchor=None,
                        proximity_bonus=0.0,
                        unit_type="phrase",
                        section=str(metadata.get("section") or ""),
                    )
                    units.append((quote_score, 90, "phrase", quote))

                for unit_score, unit_index, unit_type, unit_text in units:
                    mechanism_bonus = self._expansion_mechanism_bonus(question=question, text=unit_text)
                    document_bonus = 0.04 if anchor is not None else 0.0
                    score = min(1.0, unit_score + mechanism_bonus * 0.45 + document_bonus)
                    if score < 0.78 and mechanism_bonus < 0.14:
                        continue
                    priority = mechanism_bonus if mechanism_priority else score
                    row_candidates.append(
                        (
                            priority,
                            score,
                            self._expansion_unit_type_bonus(unit_type),
                            -row_index,
                            unit_index,
                            unit_type,
                            unit_text,
                            row,
                        )
                    )

            row_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
            used_parent_rows: set[int] = set()
            document_selected = 0
            for _, score, _, negative_row_index, unit_index, unit_type, unit_text, row in row_candidates:
                row_index = -negative_row_index
                parent_row_key = row_index
                if parent_row_key in used_parent_rows and document_selected >= 1:
                    continue
                item = self._make_expanded_evidence_item(
                    row=row,
                    fallback_document_id=document_id,
                    unit_text=unit_text,
                    unit_type=unit_type,
                    unit_index=unit_index,
                    score=score,
                    row_index=row_index,
                    anchor_position=document_position,
                    expansion_source="document",
                )
                key = f"{item.document_id}:{item.chunk_id}"
                text_key = self._expansion_text_key(item.text)
                if key in seen_keys or (text_key and text_key in seen_texts):
                    continue
                selected.append(item)
                seen_keys.add(key)
                if text_key:
                    seen_texts.add(text_key)
                used_parent_rows.add(parent_row_key)
                document_selected += 1
                if document_selected >= per_document_limit or len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

        return selected[:limit]

    def _expansion_unit_type_bonus(self, unit_type: str) -> float:
        if unit_type == "table_row":
            return 0.03
        if unit_type == "phrase":
            return 0.02
        if unit_type == "sentence":
            return 0.01
        return 0.0

    def _looks_like_mechanism_expansion_question(self, question: str) -> bool:
        normalized = self._sanitize_evidence_text(question).lower()
        return any(
            marker in normalized
            for marker in [
                "how",
                "why",
                "pursue",
                "achieve",
                "improve",
                "approach",
                "method",
                "mechanism",
                "compare",
                "如何",
                "怎么",
                "怎样",
                "机制",
                "方法",
                "对比",
            ]
        )

    def _expansion_mechanism_bonus(self, *, question: str, text: str) -> float:
        if not self._looks_like_mechanism_expansion_question(question):
            return 0.0
        normalized = self._sanitize_evidence_text(text).lower()
        bonus = 0.0
        if any(marker in normalized for marker in ["we propose", "propose", "proposed", "method", "approach"]):
            bonus += 0.06
        if any(marker in normalized for marker in ["using", "by ", "inject", "injected", "scale", "scaling"]):
            bonus += 0.04
        if "compound" in normalized and ("coefficient" in normalized or "scaling" in normalized):
            bonus += 0.12
        if all(marker in normalized for marker in ["width", "depth", "resolution"]):
            bonus += 0.10
        if "rank decomposition" in normalized or "low-rank" in normalized or "low rank" in normalized:
            bonus += 0.12
        if any(marker in normalized for marker in ["freeze", "freezes", "frozen"]) and "weight" in normalized:
            bonus += 0.10
        if "trainable" in normalized and any(marker in normalized for marker in ["parameter", "matrix", "matrices"]):
            bonus += 0.08
        if any(marker in normalized for marker in ["efficiency", "efficient", "memory", "flops", "latency", "faster", "smaller"]):
            bonus += 0.05
        return min(0.32, bonus)

    def _expansion_row_index(self, *, rows: list[dict[str, Any]], chunk_id: str) -> int:
        for index, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            if str(row.get("id") or "") == chunk_id or str(metadata.get("chunk_id") or "") == chunk_id:
                return index
        return -1

    def _expansion_row_indices(
        self,
        *,
        rows: list[dict[str, Any]],
        current_index: int,
        anchor: EvidenceItem,
    ) -> list[tuple[int, float]]:
        selected: dict[int, float] = {current_index: 0.18}
        for offset, bonus in [(-1, 0.12), (1, 0.12), (-2, 0.08), (2, 0.08)]:
            index = current_index + offset
            if 0 <= index < len(rows):
                selected[index] = max(selected.get(index, 0.0), bonus)

        anchor_pages = {
            page
            for page in range(int(anchor.page_start or anchor.page or 0), int(anchor.page_end or anchor.page or 0) + 1)
            if page > 0
        }
        anchor_section = (anchor.section or "").strip().lower()
        for index, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            page_start = int(metadata.get("page_start", metadata.get("page", 0)) or 0)
            page_end = int(metadata.get("page_end", metadata.get("page", 0)) or 0)
            row_pages = {page for page in range(page_start, page_end + 1) if page > 0}
            if anchor_pages and row_pages and anchor_pages & row_pages:
                selected[index] = max(selected.get(index, 0.0), 0.09)

            row_section = str(metadata.get("section") or "").strip().lower()
            if anchor_section and row_section and row_section == anchor_section and abs(index - current_index) <= 6:
                selected[index] = max(selected.get(index, 0.0), 0.07)

        ranked = sorted(selected.items(), key=lambda row: (abs(row[0] - current_index), -row[1]))
        return [(index, bonus) for index, bonus in ranked[:10]]

    def _evidence_units_from_expansion_row(
        self,
        *,
        question: str,
        row: dict[str, Any],
        fallback_document_id: str,
        anchor: EvidenceItem,
        row_index: int,
        anchor_position: int,
        proximity_bonus: float,
    ) -> list[EvidenceItem]:
        metadata = row.get("metadata") or {}
        text = self._sanitize_evidence_text(str(row.get("text", "")))
        if not text.strip():
            return []

        units: list[tuple[float, int, str, str]] = []
        if self._is_table_like_text(text) or "table" in str(metadata.get("chunk_type") or "").lower():
            units.extend(
                self._table_row_units_for_expansion(
                    question=question,
                    text=text,
                    max_units=3,
                )
            )
        units.extend(
            self._sentence_units_for_expansion(
                question=question,
                text=text,
                max_units=4,
            )
        )

        quote = self._best_quote_for_question(question, text, limit=360)
        if quote and quote != self._truncate_readable_text(text, limit=360):
            quote_score = self._evidence_unit_support_score(
                question=question,
                text=quote,
                anchor=anchor,
                proximity_bonus=proximity_bonus,
                unit_type="phrase",
                section=str(metadata.get("section") or ""),
            )
            units.append((quote_score, 90, "phrase", quote))

        ranked_units = sorted(units, key=lambda row: (row[0], -row[1]), reverse=True)
        evidence: list[EvidenceItem] = []
        seen_texts: set[str] = set()
        for unit_score, unit_index, unit_type, unit_text in ranked_units:
            text_key = self._expansion_text_key(unit_text)
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            if unit_type != "phrase":
                unit_score = min(1.0, unit_score + min(float(anchor.score or 0.0), 1.0) * 0.18 + proximity_bonus)
            if unit_score < 0.58:
                continue
            evidence.append(
                self._make_expanded_evidence_item(
                    row=row,
                    fallback_document_id=fallback_document_id,
                    unit_text=unit_text,
                    unit_type=unit_type,
                    unit_index=unit_index,
                    score=unit_score,
                    row_index=row_index,
                    anchor_position=anchor_position,
                )
            )
            if len(evidence) >= 4:
                break
        return evidence

    def _sentence_units_for_expansion(
        self,
        *,
        question: str,
        text: str,
        max_units: int,
    ) -> list[tuple[float, int, str, str]]:
        sentences = self._split_quote_sentences(text)
        units: list[tuple[float, int, str, str]] = []
        for index, sentence in enumerate(sentences):
            if len(sentence) < 24:
                continue
            unit_text = sentence
            if len(unit_text) < 90 and index + 1 < len(sentences):
                unit_text = f"{unit_text} {sentences[index + 1]}"
            score = self._evidence_unit_support_score(
                question=question,
                text=unit_text,
                anchor=None,
                proximity_bonus=0.0,
                unit_type="sentence",
                section="",
            )
            units.append((score, index, "sentence", self._truncate_readable_text(unit_text, limit=520)))
        units.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return units[:max_units]

    def _table_row_units_for_expansion(
        self,
        *,
        question: str,
        text: str,
        max_units: int,
    ) -> list[tuple[float, int, str, str]]:
        rows = self._split_table_rows(text)
        if not rows:
            return []
        header = self._table_header_row(rows)
        focus_terms = self._quote_focus_terms(question, self._question_keywords(question), [])
        units: list[tuple[float, int, str, str]] = []
        for index, row_text in enumerate(rows):
            if header and row_text == header:
                continue
            normalized = row_text.lower()
            term_hits = sum(1 for term in focus_terms if self._quote_term_present(term, normalized))
            number_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|times|m|b|k|million|billion)?\b", normalized))
            metric_hits = sum(
                1
                for term in ["accuracy", "error", "score", "metric", "result", "auc", "f1", "memory", "parameter", "%"]
                if term in normalized
            )
            if term_hits <= 0 and number_hits <= 0 and metric_hits <= 0:
                continue
            unit_text = f"{header} | {row_text}" if header and header not in row_text else row_text
            score = min(1.0, 0.58 + term_hits * 0.08 + min(number_hits, 4) * 0.08 + metric_hits * 0.07)
            units.append((score, index, "table_row", self._truncate_readable_text(unit_text, limit=620)))
        units.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return units[:max_units]

    def _evidence_unit_support_score(
        self,
        *,
        question: str,
        text: str,
        anchor: EvidenceItem | None,
        proximity_bonus: float,
        unit_type: str,
        section: str,
    ) -> float:
        normalized = self._sanitize_evidence_text(text).lower()
        relevance = self._question_relevance_score(question, normalized)
        keywords = self._question_keywords(question)[:18]
        keyphrase_hits = sum(1 for phrase in self._question_keyphrases(question)[:8] if phrase in normalized)
        keyword_hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
        numeric_bonus = 0.0
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|times|m|b|k|million|billion)?\b", normalized):
            numeric_bonus += 0.10
            if self._looks_like_metric_result_question(question) or self._looks_like_efficiency_or_resource_question(question):
                numeric_bonus += 0.14
        unit_bonus = {
            "table_row": 0.18,
            "phrase": 0.12,
            "sentence": 0.08,
        }.get(unit_type, 0.0)
        mechanism_bonus = self._expansion_mechanism_bonus(question=question, text=normalized)
        section_bonus = 0.06 if any(marker in section.lower() for marker in ["result", "experiment", "abstract", "conclusion"]) else 0.0
        anchor_score = min(float(anchor.score or 0.0), 1.0) * 0.28 if anchor is not None else 0.0
        score = (
            0.42
            + anchor_score
            + relevance * 0.55
            + min(keyword_hits, 5) * 0.035
            + min(keyphrase_hits, 3) * 0.08
            + numeric_bonus
            + unit_bonus
            + mechanism_bonus
            + section_bonus
            + proximity_bonus
        )
        return max(0.0, min(1.0, score))

    def _make_expanded_evidence_item(
        self,
        *,
        row: dict[str, Any],
        fallback_document_id: str,
        unit_text: str,
        unit_type: str,
        unit_index: int,
        score: float,
        row_index: int,
        anchor_position: int,
        expansion_source: str = "neighbor",
    ) -> EvidenceItem:
        metadata = row.get("metadata") or {}
        base_chunk_id = str(metadata.get("chunk_id") or row.get("id") or f"row_{row_index:05d}")
        page = int(metadata.get("page", metadata.get("page_start", 0)) or 0)
        page_start = int(metadata.get("page_start", page) or page)
        page_end = int(metadata.get("page_end", page) or page)
        quote = self._best_quote_for_question("", unit_text, limit=360)
        return EvidenceItem(
            citation_id="",
            chunk_id=f"{base_chunk_id}:{unit_type}:{unit_index:02d}",
            document_id=str(metadata.get("document_id", fallback_document_id)),
            paper_name=str(metadata.get("paper_name", "")),
            page=page,
            page_start=page_start,
            page_end=page_end,
            section=str(metadata.get("section") or ""),
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=score,
            rule_score=score,
            final_score=score,
            score_source=f"expanded_{unit_type}",
            text=unit_text,
            quote=quote or self._truncate_readable_text(unit_text, limit=360),
            char_start=int(metadata.get("char_start", 0) or 0),
            char_end=int(metadata.get("char_end", 0) or 0),
            token_count=None,
            chunk_type=unit_type,
            parent_id=base_chunk_id,
            image_id=str(metadata.get("image_id") or "") or None,
            image_path=str(metadata.get("image_path") or "") or None,
            bbox_json=str(metadata.get("bbox_json") or "") or None,
            quality_reasons=[
                f"expansion_source:{expansion_source}",
                f"expanded_from_chunk:{base_chunk_id}",
                f"row_index:{row_index}",
                f"anchor_rank:{anchor_position + 1}",
            ],
        )

    def _expansion_text_key(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", self._sanitize_evidence_text(text).lower()).strip()
        return normalized[:260]

    def _should_balance_multi_document_retrieval(
        self,
        *,
        question: str,
        soft_intent: dict[str, Any],
        document_ids: list[str],
    ) -> bool:
        if len(document_ids) < 2:
            return False
        intent = str(soft_intent.get("intent") or "")
        scope = str(soft_intent.get("scope") or "")
        return (
            intent in {"compare_question", "document_wide_question"}
            or scope in {"multi_document", "whole_document"}
            or self._looks_like_compare_question(question)
            or self._looks_like_document_wide_question(question)
            or self._looks_like_multi_document_topic_question(question)
        )

    def _multi_document_hybrid_evidence(
        self,
        *,
        question: str,
        retrieval_queries: list[str],
        soft_intent: dict[str, Any],
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
        embedding_events: list[dict[str, Any]] | None = None,
    ) -> list[EvidenceItem]:
        per_document_limit = max(2, min(4, top_k))
        candidate_lists: list[list[EvidenceItem]] = []
        weights: list[float] = []
        per_document_selected: list[EvidenceItem] = []

        for document_id in document_ids:
            scoped_document_ids = [document_id]
            scoped_lists: list[list[EvidenceItem]] = []
            scoped_weights: list[float] = []
            vector_candidates_by_query = self._multi_query_vector_evidence(
                retrieval_queries=retrieval_queries or [question],
                document_ids=scoped_document_ids,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
                embedding_events=embedding_events,
                max_queries=3,
            )
            for query_index, query in enumerate(retrieval_queries or [question]):
                targeted = self._targeted_evidence_candidates(
                    question=query,
                    soft_intent=soft_intent,
                    document_ids=scoped_document_ids,
                    top_k=max(top_k, per_document_limit),
                )
                vector_candidates = vector_candidates_by_query.get(self._retrieval_query_key(query), [])
                sparse_candidates = self._bm25_sparse_evidence(
                    question=query,
                    soft_intent=soft_intent,
                    document_ids=scoped_document_ids,
                    top_k=max(top_k * 4, 12),
                )
                query_weight = 1.0 if query_index == 0 else 0.82
                scoped_lists.extend([targeted, vector_candidates, sparse_candidates])
                scoped_weights.extend([1.15 * query_weight, 1.0 * query_weight, 1.0 * query_weight])
            scoped_fused = self._rrf_fuse_evidence_candidates(
                candidate_lists=scoped_lists,
                weights=scoped_weights,
                limit=max(per_document_limit * 4, 8),
            )
            scoped_ranked = self._select_rrf_ranked_evidence(
                question=question,
                evidence=scoped_fused,
                limit=per_document_limit,
            )
            per_document_selected.extend(scoped_ranked)
            candidate_lists.extend(scoped_lists)
            weights.extend(scoped_weights)

        if per_document_selected:
            candidate_lists.insert(0, per_document_selected)
            weights.insert(0, 1.35)

        fused = self._rrf_fuse_evidence_candidates(
            candidate_lists=candidate_lists,
            weights=weights,
            limit=max(top_k * 8, len(document_ids) * per_document_limit),
        )
        expanded = self._expand_retrieval_candidates(
            question=question,
            candidates=fused,
            limit=max(top_k * 12, len(document_ids) * per_document_limit * 4),
        )
        return self._select_balanced_multi_document_evidence(
            question=question,
            evidence=expanded,
            target_document_ids=document_ids,
            limit=max(top_k * 4, len(document_ids) * 2),
        )

    def _select_balanced_multi_document_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        target_document_ids: list[str],
        limit: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        ranked = sorted(
            [(item.score, position, item) for position, item in enumerate(evidence)],
            key=lambda row: (row[0], -row[1]),
            reverse=True,
        )
        selected: list[EvidenceItem] = []
        selected_chunks: set[str] = set()

        for document_id in target_document_ids:
            for _, _, item in ranked:
                if item.document_id != document_id or item.chunk_id in selected_chunks:
                    continue
                selected.append(item)
                selected_chunks.add(item.chunk_id)
                break

        per_document_cap = max(2, limit // max(len(target_document_ids), 1) + 1)
        per_document_counts: dict[str, int] = Counter(item.document_id for item in selected if item.document_id)
        for _, _, item in ranked:
            if item.chunk_id in selected_chunks:
                continue
            count = per_document_counts.get(item.document_id, 0)
            if count >= per_document_cap:
                continue
            selected.append(item)
            selected_chunks.add(item.chunk_id)
            per_document_counts[item.document_id] = count + 1
            if len(selected) >= limit:
                break

        for item in selected:
            item.quote = self._best_quote_for_question(question, item.text)
        return selected[:limit]

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
        if intent == "compare_question" or scope == "multi_document" or self._looks_like_multi_document_topic_question(question):
            extend("compare", self._comparison_evidence(document_ids=document_ids, top_k=top_k), 0.2)
        structure_roles = [
            role
            for role in self._paper_structure_roles_for_question(question, soft_intent=soft_intent)
            if role not in {"field", "reference"}
        ]
        if structure_roles:
            extend(
                "paper_structure",
                self._paper_structure_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    roles=structure_roles,
                ),
                0.28,
            )
            extend(
                "paper_salient",
                self._paper_salient_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    roles=structure_roles,
                ),
                0.18,
            )
        if self._looks_like_framework_function_question(question):
            extend(
                "framework_functions",
                self._framework_function_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                ),
                0.55,
            )
        if self._looks_like_trustworthy_characteristics_question(question):
            extend(
                "trustworthy_characteristics",
                self._keyword_rule_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    terms=[
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
                    bonus_phrases=[
                        "characteristics of trustworthy ai systems include",
                        "Fig. 4. Characteristics of trustworthy AI systems",
                    ],
                    min_hits=3,
                    score_source="trustworthy_characteristics_rule",
                ),
                0.5,
            )
        if self._looks_like_pretraining_objective_question(question):
            extend(
                "pretraining_objectives",
                self._keyword_rule_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    terms=[
                        "pre-training objective",
                        "training objective",
                        "objective function",
                        "self-supervised",
                        "supervised objective",
                        "pretext task",
                        "loss function",
                    ],
                    bonus_phrases=["pre-training objective", "training objective", "loss function"],
                    min_hits=2,
                    score_source="pretraining_objective_rule",
                ),
                0.5,
            )
        if self._looks_like_dataset_or_scale_question(question):
            extend(
                "dataset_or_scale",
                self._keyword_rule_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    terms=[
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
                    bonus_phrases=[
                        "training set",
                        "pre-training dataset",
                        "data collection",
                        "image-text pairs",
                    ],
                    min_hits=2,
                    score_source="dataset_scale_rule",
                ),
                0.5,
            )
        if self._looks_like_metric_result_question(question):
            extend(
                "metric_results",
                self._keyword_rule_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    terms=[
                        "Table 1",
                        "Table 2",
                        "Table 3",
                        "benchmark",
                        "metric",
                        "score",
                        "result",
                        "accuracy",
                        "error",
                        "F1",
                        "AUC",
                        "zero-shot",
                        "perplexity",
                        "state-of-the-art",
                    ],
                    bonus_phrases=[
                        "state-of-the-art",
                        "outperforms",
                        "results are shown",
                        "experimental results",
                        "benchmark results",
                    ],
                    min_hits=2,
                    score_source="metric_result_rule",
                ),
                0.45,
            )
        if self._looks_like_visual_retrieval_question(question):
            extend(
                "visual",
                self._modality_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    include_images=True,
                    include_tables=False,
                ),
                0.45,
            )
        if self._looks_like_table_question(question):
            extend(
                "table",
                self._modality_evidence(
                    question=question,
                    document_ids=document_ids,
                    top_k=top_k,
                    include_images=False,
                    include_tables=True,
                ),
                0.45,
            )
        if (
            intent == "document_wide_question"
            or scope == "whole_document"
            or self._looks_like_document_wide_question(question)
            or self._looks_like_broad_overview_question(question)
            or self._looks_like_multi_document_topic_question(question)
        ):
            extend(
                "overview",
                self._overview_evidence(question=question, document_ids=document_ids, top_k=top_k),
                0.2,
            )
        return candidates

    def _looks_like_framework_function_question(self, question: str) -> bool:
        normalized = question.lower()
        has_function_intent = any(
            keyword in normalized
            for keyword in ["核心功能", "功能集合", "功能", "functions", "function", "core functions", "core function"]
        )
        has_framework_scope = any(
            keyword in normalized
            for keyword in ["framework", "框架", "model", "system", "method"]
        )
        asks_component_role = any(keyword in normalized for keyword in ["作用", "角色", "role", "component", "stage"])
        return has_framework_scope and (has_function_intent or asks_component_role)

    def _looks_like_trustworthy_characteristics_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(keyword in normalized for keyword in ["可信", "trustworthy"]) and any(
            keyword in normalized for keyword in ["特征", "特点", "characteristic", "characteristics"]
        )

    def _looks_like_pretraining_objective_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(keyword in normalized for keyword in ["预训练目标", "pre-training objective", "pretraining objective"]) or (
            any(keyword in normalized for keyword in ["预训练", "pre-training", "pretraining"])
            and any(keyword in normalized for keyword in ["目标", "objective", "goal", "loss", "task"])
        )

    def _looks_like_dataset_or_scale_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            keyword in normalized
            for keyword in [
                "数据集",
                "训练数据",
                "数据规模",
                "规模",
                "dataset",
                "corpus",
                "training set",
                "data scale",
                "samples",
                "多少",
                "构建",
            ]
        )

    def _looks_like_metric_result_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            keyword in normalized
            for keyword in [
                "结果",
                "result",
                "results",
                "table",
                "表格",
                "指标",
                "metric",
                "score",
                "accuracy",
                "error",
                "benchmark",
            ]
        )

    def _keyword_rule_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        terms: list[str],
        bonus_phrases: list[str],
        min_hits: int,
        score_source: str,
    ) -> list[EvidenceItem]:
        normalized_terms = [term.lower() for term in terms]
        normalized_bonus = [phrase.lower() for phrase in bonus_phrases]
        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            for row in self.vector_store.get_document_chunks(document_id, limit=1000):
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                normalized_text = text.lower()
                hits = [
                    term
                    for term in normalized_terms
                    if term and term in normalized_text
                ]
                if len(hits) < min_hits:
                    continue
                metadata = row.get("metadata") or {}
                score = 0.72 + min(0.28, len(set(hits)) * 0.05)
                score += self._question_relevance_score(question, f"{metadata.get('paper_name', '')}\n{text}") * 0.35
                page = int(metadata.get("page", 0) or 0)
                if page and page <= 2 and (
                    self._looks_like_broad_overview_question(question)
                    or self._looks_like_metric_result_question(question)
                    or any(term in question for term in ["概括", "核心贡献", "主要贡献"])
                ):
                    score += 0.18
                for phrase in normalized_bonus:
                    if phrase and phrase in normalized_text:
                        score += 0.16
                chunk_type = str(metadata.get("chunk_type") or "").lower()
                if "table" in chunk_type:
                    score += 0.1
                if any(marker in chunk_type for marker in ["image", "figure", "chart"]):
                    score += 0.08
                item = self._evidence_from_row(
                    row,
                    document_id,
                    score=min(1.0, score),
                    rule_score=min(1.0, score),
                    final_score=min(1.0, score),
                    score_source=score_source,
                )
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[: max(top_k, 1)]]

    def _paper_structure_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        roles: list[str],
    ) -> list[EvidenceItem]:
        normalized_roles = [
            self._paper_structure_role_alias(role)
            for role in roles
            if self._paper_structure_role_alias(role) in self._paper_structure_signal_definitions()
        ]
        if not normalized_roles:
            return []

        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            total = max(len(rows), 1)
            for index, row in enumerate(rows):
                metadata = row.get("metadata") or {}
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                if not text.strip():
                    continue
                if self._looks_like_reference_section_text(text) and "reference" not in normalized_roles:
                    continue
                if self._looks_like_front_matter_noise(text) and "field" not in normalized_roles:
                    continue
                section = str(metadata.get("section") or "")
                role_scores = self._semantic_role_scores(
                    text=text,
                    section=section,
                    index=index,
                    total=total,
                )
                best_role = max(
                    normalized_roles,
                    key=lambda role: role_scores.get(role, 0.0),
                )
                best_role_score = role_scores.get(best_role, 0.0)
                relevance = self._question_relevance_score(question, f"{metadata.get('paper_name', '')}\n{section}\n{text}")
                if best_role_score < 0.35 and relevance < 0.14:
                    continue

                chunk_type = str(metadata.get("chunk_type") or "").lower()
                score = 0.58 + min(0.26, best_role_score * 0.12) + relevance * 0.32
                score += self._readable_text_score(text) * 0.08
                if "table" in normalized_roles and ("table" in chunk_type or self._is_table_like_text(text)):
                    score += 0.12
                if "visual" in normalized_roles and any(marker in chunk_type for marker in ["image", "figure", "chart"]):
                    score += 0.14
                if best_role in {"purpose", "conclusion"} and index <= max(2, int(total * 0.08)):
                    score += 0.08

                item = self._evidence_from_row(
                    row,
                    document_id,
                    score=min(1.0, score),
                    rule_score=min(1.0, score),
                    final_score=min(1.0, score),
                    score_source=f"paper_structure_{best_role}",
                )
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[: max(top_k, 1)]]

    def _paper_salient_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        roles: list[str],
    ) -> list[EvidenceItem]:
        normalized_question = " ".join(self._sanitize_evidence_text(question).lower().split())
        phrases = self._question_keyphrases(question)
        keywords = self._question_keywords(question)
        normalized_roles = [
            self._paper_structure_role_alias(role)
            for role in roles
            if self._paper_structure_role_alias(role) in self._paper_structure_signal_definitions()
        ]
        if not normalized_roles:
            normalized_roles = ["purpose", "approach", "claim"]

        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            total = max(len(rows), 1)
            for index, row in enumerate(rows):
                metadata = row.get("metadata") or {}
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                if not text.strip():
                    continue
                if self._looks_like_reference_section_text(text):
                    continue
                if self._looks_like_front_matter_noise(text):
                    continue

                section = str(metadata.get("section") or "")
                normalized_text = " ".join(text.lower().split())
                phrase_text = normalized_text.replace("-", " ")
                phrase_hits = sum(1 for phrase in phrases if phrase in phrase_text)
                keyword_hits = sum(1 for term in keywords[:16] if term.lower() in normalized_text)
                role_scores = self._semantic_role_scores(
                    text=text,
                    section=section,
                    index=index,
                    total=total,
                )
                role_score = max((role_scores.get(role, 0.0) for role in normalized_roles), default=0.0)
                relevance = self._question_relevance_score(question, f"{metadata.get('paper_name', '')}\n{section}\n{text}")
                salient_bonus = self._paper_salient_signal_bonus(
                    question=normalized_question,
                    text=normalized_text,
                    section=section,
                    index=index,
                    total=total,
                )
                score = (
                    relevance * 1.25
                    + min(phrase_hits, 3) * 0.38
                    + min(keyword_hits, 5) * 0.08
                    + role_score * 0.42
                    + salient_bonus
                    + self._readable_text_score(text) * 0.12
                )
                if score < 0.72:
                    continue

                item = self._evidence_from_row(
                    row,
                    document_id,
                    score=min(1.0, 0.56 + score * 0.18),
                    rule_score=min(1.0, 0.56 + score * 0.18),
                    final_score=min(1.0, 0.56 + score * 0.18),
                    score_source="paper_salient",
                )
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[: max(top_k, 1)]]

    def _paper_salient_signal_bonus(
        self,
        *,
        question: str,
        text: str,
        section: str,
        index: int,
        total: int,
    ) -> float:
        score = 0.0
        early = index <= max(2, int(max(total, 1) * 0.08))
        section_normalized = section.lower()
        if "abstract" in question and (early or "abstract" in section_normalized or "abstract" in text[:120]):
            score += 0.45
        if any(term in question for term in ["main idea", "idea", "method", "approach", "mechanism", "adaptation"]):
            if any(
                phrase in text
                for phrase in [
                    "we propose",
                    "we introduce",
                    "we present",
                    "we define",
                    "is designed to",
                    "allows us",
                    "consists of",
                    "based on",
                ]
            ):
                score += 0.48
        if any(term in question for term in ["result", "results", "benchmark", "performance", "report", "reported"]):
            if any(
                phrase in text
                for phrase in [
                    "results show",
                    "we show",
                    "we demonstrate",
                    "achieves",
                    "outperforms",
                    "state-of-the-art",
                    "accuracy",
                    "error",
                ]
            ):
                score += 0.42
        if any(term in question for term in ["efficiency", "efficient", "benefit", "benefits", "faster", "smaller", "memory", "parameter"]):
            if any(term in text for term in ["efficient", "efficiency", "faster", "smaller", "memory", "parameters", "compute", "storage"]):
                score += 0.42
            if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|times|m|b|k|million|billion)?\b", text):
                score += 0.2
        return score

    def _framework_function_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        function_terms = [
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
        expected_terms = self._expected_framework_function_terms(question)
        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            for row in self.vector_store.get_document_chunks(document_id, limit=1000):
                metadata = row.get("metadata") or {}
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                normalized_text = text.lower()
                hits = [
                    term
                    for term in function_terms
                    if re.search(rf"\b{re.escape(term)}\b", normalized_text)
                ]
                if len(hits) < 2:
                    continue
                if expected_terms and not expected_terms.issubset(set(hits)):
                    continue

                score = 0.72 + min(0.22, len(set(hits)) * 0.04)
                if any(marker in normalized_text for marker in ["core", "function", "functions", "component", "components", "categories"]):
                    score += 0.16
                if any(marker in normalized_text for marker in ["composed of", "consists of", "organized into", "functions organize"]):
                    score += 0.24
                chunk_type = str(metadata.get("chunk_type") or "").lower()
                if any(marker in chunk_type for marker in ["image", "figure", "table"]):
                    score += 0.08 if self._question_requires_visual_evidence(question) else -0.16
                score += self._question_relevance_score(question, f"{metadata.get('paper_name', '')}\n{text}") * 0.25

                item = self._evidence_from_row(
                    row,
                    document_id,
                    score=min(1.0, score),
                    rule_score=min(1.0, score),
                    final_score=min(1.0, score),
                    score_source="framework_function_rule",
                )
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[: max(top_k, 1)]]

    def _modality_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        include_images: bool,
        include_tables: bool,
    ) -> list[EvidenceItem]:
        scored: list[tuple[float, int, EvidenceItem]] = []
        position = 0
        for document_id in document_ids:
            for row in self.vector_store.get_document_chunks(document_id, limit=1000):
                metadata = row.get("metadata") or {}
                text = self._sanitize_evidence_text(str(row.get("text", "")))
                chunk_type = str(metadata.get("chunk_type") or "").lower()
                is_image = bool(metadata.get("image_id")) or any(
                    marker in chunk_type for marker in ["image", "figure", "chart"]
                )
                is_table = "table" in chunk_type or self._is_table_like_text(text)
                if (include_images and not is_image) or (include_tables and not is_table):
                    continue
                item = self._evidence_from_row(
                    row,
                    document_id,
                    score=0.85,
                    rule_score=0.85,
                    final_score=0.85,
                    score_source="modality_rule",
                )
                relevance = self._question_relevance_score(question, f"{item.paper_name}\n{text}")
                score = 0.85 + relevance * 0.5
                if include_tables and any(marker in text for marker in ["实验分数构成", "实验过程", "实验结果", "实验总分"]):
                    score += 0.25
                if include_images and any(marker in text for marker in ["运行结果", "截图", "Visual Studio", "开放", "回显"]):
                    score += 0.25
                item.score = min(1.0, score)
                item.rule_score = item.score
                item.final_score = item.score
                item.quote = self._best_quote_for_question(question, item.text)
                scored.append((score, position, item))
                position += 1

        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[: max(top_k, 1)]]

    def _looks_like_visual_retrieval_question(self, question: str) -> bool:
        return self._looks_like_visual_question_text(question)

    def _vector_similarity_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        embedding_model: str,
        embedding_events: list[dict[str, Any]] | None = None,
    ) -> list[EvidenceItem]:
        try:
            resolved_model = self._resolve_query_embedding_model(
                requested_model=embedding_model,
                document_ids=document_ids,
            )
            query_embedding = self.model_clients.embed_query_with_info(question, model=resolved_model)
            if embedding_events is not None:
                embedding_events.append(
                    {
                        "requested_model": embedding_model,
                        "resolved_model": resolved_model,
                        "provider": query_embedding.provider,
                        "used_fallback": query_embedding.used_fallback,
                        "fallback_reason": query_embedding.fallback_reason,
                        "document_ids": document_ids,
                        "query": question,
                    }
                )
            if query_embedding.used_fallback and resolved_model != LOCAL_FALLBACK_EMBEDDING_PROVIDER:
                return []
            return self.vector_store.query(
                query_embedding=query_embedding.vector,
                top_k=top_k,
                document_ids=document_ids,
            )
        except RuntimeError:
            return []

    def _boost_evidence_scores(self, evidence: list[EvidenceItem], boost: float) -> list[EvidenceItem]:
        boosted: list[EvidenceItem] = []
        for item in evidence:
            next_score = min(1.0, max(0.0, item.score + boost))
            boosted.append(
                item.model_copy(
                    update={
                        "score": next_score,
                        "rule_score": next_score if item.rule_score is not None else next_score,
                        "final_score": next_score,
                        "score_source": item.score_source or "rule_boost",
                    }
                )
            )
        return boosted

    def _build_embedding_trace(
        self,
        *,
        requested_model: str,
        document_ids: list[str],
        query_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        document_providers: dict[str, str] = {}
        document_fallback_count = 0
        for document_id in document_ids:
            document = self.store.get_document(document_id)
            provider = str(getattr(document, "embedding_model", "") or "")
            if provider:
                document_providers[document_id] = provider
            if provider == LOCAL_FALLBACK_EMBEDDING_PROVIDER:
                document_fallback_count += 1

        provider = requested_model
        fallback_reason = ""
        query_used_fallback = False
        if query_events:
            provider = str(query_events[-1].get("provider") or provider)
            query_used_fallback = any(bool(event.get("used_fallback")) for event in query_events)
            reasons = [
                str(event.get("fallback_reason") or "").strip()
                for event in query_events
                if str(event.get("fallback_reason") or "").strip()
            ]
            fallback_reason = "; ".join(dict.fromkeys(reasons))[:500]

        if document_fallback_count and not fallback_reason:
            fallback_reason = "至少一份目标文档使用本地备用检索索引，查询向量同步使用本地备用检索。"

        return {
            "embedding_requested_model": requested_model,
            "embedding_provider": provider,
            "embedding_used_fallback": query_used_fallback or document_fallback_count > 0,
            "embedding_fallback_reason": fallback_reason,
            "embedding_document_fallback_count": document_fallback_count,
            "embedding_document_providers": document_providers,
        }

    def _resolve_query_embedding_model(
        self,
        *,
        requested_model: str,
        document_ids: list[str],
    ) -> str:
        indexed_models: list[str] = []
        for document_id in document_ids:
            document = self.store.get_document(document_id)
            provider = str(getattr(document, "embedding_model", "") or "").strip()
            if provider:
                indexed_models.append(provider)
            if provider == LOCAL_FALLBACK_EMBEDDING_PROVIDER:
                return LOCAL_FALLBACK_EMBEDDING_PROVIDER
        unique_models = sorted(set(indexed_models))
        if len(unique_models) == 1:
            return unique_models[0]
        if len(unique_models) > 1:
            raise RuntimeError(
                "目标文档使用了不同 embedding 索引，不能在同一次 dense 检索中混用向量空间。"
            )
        return requested_model
