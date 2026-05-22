from __future__ import annotations

from typing import Any

from backend.app.agent_parts.state import PaperAgentState
from backend.app.agent_parts.retrieval_engines import AgentRetrievalEngineMixin
from backend.app.agent_parts.retrieval_evidence import AgentRetrievalEvidenceMixin
from backend.app.agent_parts.retrieval_filters import AgentRetrievalFilterMixin
from backend.app.agent_parts.retrieval_rules import AgentRetrievalRuleMixin
from backend.app.agent_parts.retrieval_scoring import AgentRetrievalScoringMixin
from backend.app.models import EvidenceItem, RuntimeStep


class AgentRetrievalMixin(
    AgentRetrievalEngineMixin,
    AgentRetrievalEvidenceMixin,
    AgentRetrievalScoringMixin,
    AgentRetrievalRuleMixin,
    AgentRetrievalFilterMixin,
):
    def _retrieve(self, state: PaperAgentState) -> PaperAgentState:
        attempt = int(state.get("retrieval_attempts", 0) or 0)
        if attempt:
            self._emit_status("证据还不够稳，正在扩大范围重新检索...")
        else:
            self._emit_status("正在从当前文档里查找证据...")
        document_ids = self._resolve_document_ids(state.get("document_ids"))
        strategy = state.get("retrieval_strategy") or "hybrid_soft"
        evidence, retrieval_pipeline, ranking_method = self._hybrid_evidence(
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
            "retrieval_pipeline": retrieval_pipeline,
            "ranking_method": ranking_method,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="retriever",
                    title="检索证据",
                    detail=(
                        f"实际使用「{self._friendly_retrieval_strategy(strategy)}」，"
                        f"管线为 {retrieval_pipeline}，排序方式为 {ranking_method}；"
                        f"检索范围 {len(document_ids)} 篇文档，top-k 为 {state['top_k']}，"
                        f"这是第 {attempt + 1} 次检索，返回 {len(evidence)} 条证据。"
                    ),
                ),
            ],
        }

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
            "evidence_judgments": [],
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="retrieval_refiner",
                    title="调整检索",
                    detail=detail,
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
    ) -> tuple[list[EvidenceItem], str, str]:
        if not document_ids:
            return [], "dense_vector + bm25_sparse -> rrf_fusion", "none"

        targeted = self._targeted_evidence_candidates(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
            top_k=top_k,
        )
        vector_candidates = self._vector_similarity_evidence(
            question=question,
            document_ids=document_ids,
            top_k=max(top_k * 8, 24),
            embedding_model=embedding_model,
        )
        sparse_candidates = self._bm25_sparse_evidence(
            question=question,
            soft_intent=soft_intent,
            document_ids=document_ids,
            top_k=max(top_k * 8, 24),
        )
        fused = self._rrf_fuse_evidence_candidates(
            candidate_lists=[
                targeted,
                vector_candidates,
                sparse_candidates,
            ],
            weights=[1.15, 1.0, 1.0],
            limit=max(top_k * 8, 24),
        )
        ranked = self._select_rrf_ranked_evidence(
            question=question,
            evidence=fused,
            limit=max(top_k * 4, 8),
        )
        return (
            self._renumber_evidence(ranked),
            (
                "structured_candidates + dense_vector + bm25_sparse -> rrf_fusion"
                if targeted
                else "dense_vector + bm25_sparse -> rrf_fusion"
            ),
            "RRF 融合排序",
        )

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

