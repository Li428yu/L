from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.app.agent_parts.events import AgentEvent, final_event, status_event, token_event
from backend.app.agent_parts.answering import AgentAnsweringMixin
from backend.app.agent_parts.evidence_coverage import AgentEvidenceCoverageMixin
from backend.app.agent_parts.planning import AgentPlanningMixin
from backend.app.agent_parts.retrieval import AgentRetrievalMixin
from backend.app.agent_parts.state import (
    DocumentProfile,
    PaperAgentState,
    ParsedTask,
    ReadingTaskContract,
)
from backend.app.agent_parts.text_utils import AgentTextUtilityMixin
from backend.app.agent_parts.verification import AgentVerificationMixin
from backend.app.config import Settings
from backend.app.llm_clients import ModelClients
from backend.app.memory import MemoryManager
from backend.app.models import AskRequest, AskResponse, EvidenceItem, RagTrace, RelatedImageInfo, RuntimeStep
from backend.app.observability import ObservabilityClient
from backend.app.storage import MetadataStore
from backend.app.vector_store import ChromaPaperStore


__all__ = [
    "DocumentProfile",
    "PaperAgentService",
    "PaperAgentState",
    "ParsedTask",
    "ReadingTaskContract",
]


class PaperAgentService(
    AgentPlanningMixin,
    AgentRetrievalMixin,
    AgentVerificationMixin,
    AgentEvidenceCoverageMixin,
    AgentAnsweringMixin,
    AgentTextUtilityMixin,
):
    def __init__(
        self,
        *,
        settings: Settings,
        store: MetadataStore,
        vector_store: ChromaPaperStore,
        model_clients: ModelClients,
        memory: MemoryManager,
        observer: ObservabilityClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.vector_store = vector_store
        self.model_clients = model_clients
        self.memory = memory
        self.observer = observer
        self.graph = self._build_graph()

    def ask(self, request: AskRequest) -> AskResponse:
        final_response: AskResponse | None = None
        for event in self.stream(request):
            if event.type == "final" and isinstance(event.payload, AskResponse):
                final_response = event.payload
            elif event.type == "error":
                raise RuntimeError(str(event.payload))
        if final_response is None:
            raise RuntimeError("回答没有完整生成，请稍后重试。")
        return final_response

    def stream(self, request: AskRequest) -> Iterator[AgentEvent]:
        started = time.perf_counter()
        conversation = self.store.ensure_conversation(
            request.conversation_id,
            title=request.question,
        )
        preset = self.settings.resolve_model_preset(request.model_preset)
        chat_model = self._resolve_chat_model(request.chat_model, preset.chat_model)
        embedding_model = request.embedding_model or preset.embedding_model
        top_k = request.top_k or preset.top_k
        # Capture self-descriptions before the graph reads memory, so facts stated
        # in the current question can influence the current answer and trace.
        self.memory.remember_from_user_text(conversation.id, request.question)

        state: PaperAgentState = {
            "question": request.question,
            "conversation_id": conversation.id,
            "document_ids": request.document_ids,
            "top_k": top_k,
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "runtime": [],
            "evidence": [],
            "final_prompt_evidence": [],
            "retrieval_attempts": 0,
            "retrieval_pipeline": "",
            "ranking_method": "",
            "embedding_trace": {},
        }

        final_state = state
        for mode, payload in self.graph.stream(state, stream_mode=["custom", "updates"]):
            if mode == "custom":
                event = self._agent_event_from_graph_payload(payload)
                if event is not None:
                    yield event
                continue
            if mode != "updates" or not isinstance(payload, dict):
                continue
            for node_update in payload.values():
                if isinstance(node_update, dict):
                    final_state = {**final_state, **node_update}

        response = self._build_response_from_state(
            request=request,
            state=final_state,
            model_profile=preset.label,
            top_k=int(final_state.get("top_k", top_k) or top_k),
        )
        if self.observer is not None:
            self.observer.record_rag_run(
                request=request,
                response=response,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        yield final_event(response)

    def _build_response_from_state(
        self,
        *,
        request: AskRequest,
        state: PaperAgentState,
        model_profile: str,
        top_k: int,
    ) -> AskResponse:
        evidence = state.get("evidence", [])
        answer = state.get("answer", "本轮没有生成可用回答。")
        visible_evidence = self.attach_related_images(
            self._visible_evidence_for_answer(
                answer,
                evidence,
                question=request.question,
            ),
            question=request.question,
        )
        runtime = state.get("runtime", [])
        final_prompt_evidence = state.get("final_prompt_evidence", [])
        evidence_quality_trace = state.get("evidence_quality_trace", [])
        evidence_judgments = state.get("evidence_judgments", [])
        verification = state.get("verification", {})
        multi_document_cards = state.get("multi_document_cards", [])
        document_relation_map = state.get("document_relation_map", [])
        multi_document_coverage = state.get("multi_document_coverage", {})
        compound_tasks = state.get("compound_tasks") or []
        task_parse_reason = state.get("task_parse_reason") or ""
        intent = state.get("intent") or self._classify_question_intent(request.question)
        retrieval_strategy = state.get("retrieval_strategy") or self._retrieval_strategy_for_question(
            request.question
        )
        retrieval_pipeline = state.get("retrieval_pipeline", "")
        ranking_method = state.get("ranking_method", "")
        embedding_trace = state.get("embedding_trace", {}) or {}
        answer_strategy = state.get("answer_strategy") or "model_answer"
        fallback_used = bool(state.get("fallback_used", False))
        evidence_coverage = state.get("evidence_coverage", {}) or {}
        evidence_quality = state.get("evidence_quality") or self._evidence_quality(
            question=request.question,
            evidence=evidence,
            fallback_used=fallback_used,
            answer_strategy=answer_strategy,
        )
        diagnosis = state.get("diagnosis") or self._build_trace_diagnosis(
            intent=intent,
            retrieval_strategy=retrieval_strategy,
            answer_strategy=answer_strategy,
            evidence_quality=evidence_quality,
            evidence=evidence,
            fallback_used=fallback_used,
            verification=verification,
            embedding_trace=embedding_trace,
        )
        retrieval_debug = self._build_retrieval_debug(
            question=request.question,
            evidence=evidence,
            retrieval_strategy=retrieval_strategy,
            answer=answer,
            final_prompt_evidence=final_prompt_evidence,
        )

        conversation_id = state["conversation_id"]
        user_message_id = self.store.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.question,
        )
        self.store.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            evidence=[item.model_dump() for item in visible_evidence],
        )
        self.memory.remember_from_user_text(
            conversation_id,
            request.question,
            source_message_id=user_message_id,
        )
        self.memory.refresh_conversation_summary(conversation_id)

        retrieval_document_ids = state.get("retrieval_document_ids") or request.document_ids
        visual_ocr_warnings = state.get("visual_ocr_warnings") or self._build_visual_ocr_warnings(
            question=request.question,
            evidence=visible_evidence,
            document_ids=retrieval_document_ids,
        )

        return AskResponse(
            answer=answer,
            conversation_id=conversation_id,
            evidence=visible_evidence,
            runtime=runtime,
            rag_trace=RagTrace(
                model_profile=model_profile,
                vector_store=self.vector_store.name,
                vector_record_count=self.vector_store.count(),
                top_k=top_k,
                filter_document_ids=retrieval_document_ids,
                retrieved_count=len(evidence),
                final_prompt_evidence=final_prompt_evidence,
                intent=intent,
                retrieval_strategy=retrieval_strategy,
                retrieval_pipeline=retrieval_pipeline,
                ranking_method=ranking_method,
                embedding_requested_model=str(embedding_trace.get("embedding_requested_model") or ""),
                embedding_provider=str(embedding_trace.get("embedding_provider") or ""),
                embedding_used_fallback=bool(embedding_trace.get("embedding_used_fallback", False)),
                embedding_fallback_reason=str(embedding_trace.get("embedding_fallback_reason") or ""),
                embedding_document_fallback_count=int(embedding_trace.get("embedding_document_fallback_count") or 0),
                embedding_document_providers=dict(embedding_trace.get("embedding_document_providers") or {}),
                answer_strategy=answer_strategy,
                fallback_used=fallback_used,
                evidence_quality=evidence_quality,
                evidence_coverage=evidence_coverage,
                diagnosis=diagnosis,
                retrieval_debug=retrieval_debug,
                evidence_quality_trace=evidence_quality_trace,
                compound_tasks=compound_tasks,
                task_parse_reason=task_parse_reason,
                evidence_judgments=evidence_judgments,
                verification=verification,
                multi_document_cards=multi_document_cards,
                document_relation_map=document_relation_map,
                multi_document_coverage=multi_document_coverage,
                visual_ocr_warnings=visual_ocr_warnings,
            ),
            memory_used=self.memory.build_memory_context(conversation_id, question=request.question),
        )

    def _visible_evidence_for_answer(
        self,
        answer: str,
        evidence: list[EvidenceItem],
        *,
        question: str = "",
        limit: int = 6,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        limit = max(1, limit)
        cited_ids = set(self._citation_ids_from_answer(answer))
        visible: list[EvidenceItem] = [
            item for item in evidence if item.citation_id in cited_ids
        ]
        visible_chunks = {f"{item.document_id}:{item.chunk_id}" for item in visible}

        direct_support_scorer = getattr(self, "_final_evidence_direct_support_score", None)
        if not callable(direct_support_scorer):
            return visible[:limit]

        visual_checker = getattr(self, "_question_requires_visual_evidence", None)
        visual_question = bool(callable(visual_checker) and visual_checker(question))
        supplemental: list[tuple[float, int, EvidenceItem]] = []
        for position, item in enumerate(evidence):
            key = f"{item.document_id}:{item.chunk_id}"
            if key in visible_chunks:
                continue
            score = float(direct_support_scorer(question=question, item=item) or 0.0)
            if self._is_visual_evidence_item(item):
                if visual_question:
                    score += 0.35
                else:
                    score -= 0.45
            threshold = 1.35 if visual_question and self._is_visual_evidence_item(item) else 2.0
            if score < threshold:
                continue
            supplemental.append((score, position, item))

        supplemental.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        room = max(0, min(2, limit - len(visible)))
        for _, _, item in supplemental[:room]:
            visible.append(item)
        if not visible and supplemental:
            visible.append(supplemental[0][2])
        return visible[:limit]

    def attach_related_images(
        self,
        evidence: list[EvidenceItem],
        *,
        question: str = "",
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        image_cache: dict[str, list[dict[str, Any]]] = {}
        enriched: list[EvidenceItem] = []
        for item in evidence:
            document_id = item.document_id
            if not document_id:
                enriched.append(item)
                continue
            if document_id not in image_cache:
                image_cache[document_id] = self.store.list_document_images(document_id)
            related = self._related_images_for_evidence(item, image_cache[document_id], question=question)
            enriched.append(item.model_copy(update={"related_images": related}))
        return enriched

    def _related_images_for_evidence(
        self,
        item: EvidenceItem,
        images: list[dict[str, Any]],
        *,
        question: str = "",
        limit: int = 2,
    ) -> list[RelatedImageInfo]:
        if item.image_id or "image" in (item.chunk_type or "").lower() or "figure" in (item.chunk_type or "").lower():
            return []
        page_start = int(item.page_start or item.page or 0)
        page_end = int(item.page_end or page_start)
        if not page_start:
            return []

        scored_related: list[tuple[float, RelatedImageInfo]] = []
        for image in images:
            image_id = str(image.get("id", ""))
            if item.image_id and image_id == item.image_id:
                continue
            image_start = int(image.get("page_start", 0) or 0)
            image_end = int(image.get("page_end", image_start) or image_start)
            if image_end < page_start or image_start > page_end:
                continue
            relevance = self._related_image_relevance_score(
                question=question,
                evidence=item,
                image=image,
            )
            if relevance < 0.28:
                continue
            scored_related.append((
                relevance,
                RelatedImageInfo(
                    id=image_id,
                    document_id=str(image.get("document_id", item.document_id)),
                    page_start=image_start,
                    page_end=image_end,
                    kind=str(image.get("kind") or "image"),
                    caption_text=str(image.get("caption_text") or ""),
                    ocr_text=str(image.get("ocr_text") or ""),
                    ocr_status=str(image.get("ocr_status") or ""),
                    ocr_error=str(image.get("ocr_error") or ""),
                    vision_summary=str(image.get("vision_summary") or ""),
                    vision_error=str(image.get("vision_error") or ""),
                    status=str(image.get("status") or ""),
                )
            ))
        scored_related.sort(key=lambda row: row[0], reverse=True)
        return [image for _, image in scored_related[:limit]]

    def _build_visual_ocr_warnings(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        document_ids: list[str],
    ) -> list[dict[str, Any]]:
        expects_visual = self._looks_like_visual_retrieval_question(question)
        expects_ocr = self._looks_like_ocr_question(question)
        if not expects_visual and not expects_ocr:
            return []

        warnings: list[dict[str, Any]] = []
        evidence_document_ids = [item.document_id for item in evidence if item.document_id]
        target_document_ids = list(dict.fromkeys(evidence_document_ids or document_ids))
        if expects_visual and not any(self._evidence_has_image_signal(item) for item in evidence):
            warnings.append(
                {
                    "type": "visual_expected_but_not_cited",
                    "severity": "warn",
                    "message": "问题需要图片/视觉证据，但最终引用里没有图片型证据。",
                }
            )

        for document_id in target_document_ids:
            if not document_id:
                continue
            images = self.store.list_document_images(document_id)
            document = self.store.get_document(document_id)
            paper_name = document.file_name if document else document_id
            if not images:
                warnings.append(
                    {
                        "type": "visual_expected_no_extracted_images",
                        "severity": "warn",
                        "document_id": document_id,
                        "paper_name": paper_name,
                        "message": "问题需要图片/OCR 证据，但该文档没有已抽取图片记录。",
                    }
                )
                continue

            status_counts: dict[str, int] = {}
            ocr_status_counts: dict[str, int] = {}
            for image in images:
                status = str(image.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                ocr_status = str(image.get("ocr_status") or "unknown")
                ocr_status_counts[ocr_status] = ocr_status_counts.get(ocr_status, 0) + 1
            vision_ready_count = sum(1 for image in images if str(image.get("status") or "") == "vision_ready")
            ocr_text_count = sum(1 for image in images if str(image.get("ocr_text") or "").strip())
            ocr_failed_count = ocr_status_counts.get("ocr_failed", 0)
            ocr_empty_count = ocr_status_counts.get("ocr_empty", 0)
            ocr_skipped_count = ocr_status_counts.get("ocr_skipped", 0)
            unfinished_count = sum(
                count
                for status, count in status_counts.items()
                if status not in {"vision_ready", "ready"}
            )

            if expects_visual and vision_ready_count == 0:
                warnings.append(
                    {
                        "type": "visual_expected_without_vision_ready_images",
                        "severity": "warn",
                        "document_id": document_id,
                        "paper_name": paper_name,
                        "image_count": len(images),
                        "status_counts": status_counts,
                        "ocr_status_counts": ocr_status_counts,
                        "message": "问题需要视觉理解，但该文档图片尚无 vision_ready 记录，回答可能只依赖题注/OCR 或普通文本。",
                    }
                )
            if expects_ocr and ocr_text_count == 0:
                severity = "info" if vision_ready_count else "warn"
                warnings.append(
                    {
                        "type": "ocr_expected_without_ocr_text",
                        "severity": severity,
                        "document_id": document_id,
                        "paper_name": paper_name,
                        "image_count": len(images),
                        "vision_ready_count": vision_ready_count,
                        "status_counts": status_counts,
                        "ocr_status_counts": ocr_status_counts,
                        "message": "问题涉及 OCR/扫描文本，但图片记录中没有 OCR 文本；当前会退化为视觉摘要或图片证据。",
                    }
                )
            if expects_ocr and (ocr_failed_count or ocr_empty_count or ocr_skipped_count):
                warnings.append(
                    {
                        "type": "ocr_processing_not_fully_ready",
                        "severity": "info",
                        "document_id": document_id,
                        "paper_name": paper_name,
                        "image_count": len(images),
                        "ocr_failed_count": ocr_failed_count,
                        "ocr_empty_count": ocr_empty_count,
                        "ocr_skipped_count": ocr_skipped_count,
                        "ocr_status_counts": ocr_status_counts,
                        "message": "OCR 证据不完整；信任图片文字覆盖率前应先检查 ocr_status_counts。",
                    }
                )
            if unfinished_count and (expects_visual or expects_ocr):
                warnings.append(
                    {
                        "type": "image_processing_incomplete",
                        "severity": "info",
                        "document_id": document_id,
                        "paper_name": paper_name,
                        "image_count": len(images),
                        "unfinished_count": unfinished_count,
                        "status_counts": status_counts,
                        "ocr_status_counts": ocr_status_counts,
                        "message": "该文档仍有图片处于未完全视觉化/OCR 化状态，必要时应重新索引或提高视觉处理上限。",
                    }
                )
        return warnings

    def _evidence_has_image_signal(self, item: EvidenceItem) -> bool:
        chunk_type = (item.chunk_type or "").lower()
        return bool(item.image_id or item.image_path or item.related_images) or any(
            marker in chunk_type for marker in ["image", "figure", "chart"]
        )

    def _looks_like_ocr_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            keyword in normalized
            for keyword in ["ocr", "扫描", "文字识别", "图片里的文字", "图中文字", "纯文字", "text layer", "文本层"]
        )

    def _related_image_relevance_score(
        self,
        *,
        question: str,
        evidence: EvidenceItem,
        image: dict[str, Any],
    ) -> float:
        image_text = self._related_image_text(image)
        if not image_text:
            return 0.0
        evidence_focus = self._truncate_readable_text(
            " ".join(part for part in [evidence.quote, evidence.text] if part),
            limit=520,
        )
        question_score = self._question_relevance_score(question, image_text) if question else 0.0
        evidence_score = self._question_relevance_score(evidence_focus, image_text)
        shared_score = self._shared_visual_keyword_score(
            question=question,
            evidence_focus=evidence_focus,
            image_text=image_text,
        )
        content_bonus = 0.06 if str(image.get("vision_summary") or "").strip() else 0.0
        content_bonus += 0.04 if str(image.get("ocr_text") or "").strip() else 0.0
        return min(1.0, question_score * 0.35 + evidence_score * 0.45 + shared_score * 0.35 + content_bonus)

    def _related_image_text(self, image: dict[str, Any]) -> str:
        return self._sanitize_evidence_text(
            " ".join(
                str(image.get(field) or "")
                for field in ["caption_text", "ocr_text", "vision_summary", "kind"]
            )
        ).strip()

    def _shared_visual_keyword_score(
        self,
        *,
        question: str,
        evidence_focus: str,
        image_text: str,
    ) -> float:
        terms = [
            term
            for term in [*self._question_keywords(question), *self._question_keywords(evidence_focus)]
            if len(term.strip()) >= 2
        ]
        unique_terms = list(dict.fromkeys(terms))[:24]
        if not unique_terms:
            return 0.0
        normalized_image = image_text.lower()
        hits = sum(1 for term in unique_terms if term in image_text or term.lower() in normalized_image)
        return min(1.0, hits / max(min(len(unique_terms), 8), 1))

    def _resolve_chat_model(self, requested_model: str | None, preset_model: str) -> str:
        options = set(self.settings.chat_model_options)
        if requested_model and requested_model in options:
            return requested_model
        return preset_model or self.settings.default_chat_model

    def _build_graph(self):
        builder = StateGraph(PaperAgentState)
        builder.add_node("memory", self._load_memory)
        builder.add_node("planner", self._plan)
        builder.add_node("retriever", self._retrieve)
        builder.add_node("evidence_judge", self._judge_evidence)
        builder.add_node("retrieval_refiner", self._refine_retrieval)
        builder.add_node("answer", self._answer)
        builder.add_node("verifier", self._verify_answer)
        builder.add_node("memory_writer", self._write_memory)
        builder.add_edge(START, "memory")
        builder.add_edge("memory", "planner")
        builder.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"retrieve": "retriever", "answer": "answer"},
        )
        builder.add_edge("retriever", "evidence_judge")
        builder.add_conditional_edges(
            "evidence_judge",
            self._route_after_evidence_judge,
            {"retry_retrieve": "retrieval_refiner", "answer": "answer"},
        )
        builder.add_edge("retrieval_refiner", "retriever")
        builder.add_conditional_edges(
            "answer",
            self._route_after_answer,
            {"verify": "verifier", "write_memory": "memory_writer"},
        )
        builder.add_edge("verifier", "memory_writer")
        builder.add_edge("memory_writer", END)
        return builder.compile()

    def _route_after_answer(self, state: PaperAgentState) -> str:
        if not state.get("needs_retrieval"):
            return "write_memory"
        answer_strategy = str(state.get("answer_strategy") or "")
        if answer_strategy in {"model_unavailable", "missing_evidence_refusal"}:
            return "write_memory"
        answer = state.get("answer", "")
        if state.get("evidence") or self._citation_ids_from_answer(answer):
            return "verify"
        return "write_memory"

    def _load_memory(self, state: PaperAgentState) -> PaperAgentState:
        self._emit_status("正在读取记忆...")
        facts = self.memory.build_memory_context(
            state["conversation_id"],
            question=state["question"],
        )
        recent_messages = self.store.get_recent_messages(state["conversation_id"], limit=10)
        return {
            **state,
            "memory_facts": facts,
            "memory_prompt": self.memory.render_memory_prompt(facts),
            "recent_messages": recent_messages,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="memory",
                    title="读取记忆",
                    detail=f"读取到 {len(facts)} 条长期画像/偏好，{len(recent_messages)} 条短期历史。",
                ),
            ],
        }

    def _write_memory(self, state: PaperAgentState) -> PaperAgentState:
        self._emit_status("正在更新记忆...")
        remembered = self.memory.remember_from_user_text(state["conversation_id"], state["question"])
        facts = self.memory.build_memory_context(
            state["conversation_id"],
            question=state["question"],
        )
        detail = (
            f"本轮识别并写入 {len(remembered)} 条新画像/偏好；长期记忆当前共有 {len(facts)} 条。"
            if remembered
            else f"本轮没有识别到新的画像/偏好；长期记忆当前共有 {len(facts)} 条。"
        )
        return {
            **state,
            "memory_facts": facts,
            "memory_prompt": self.memory.render_memory_prompt(facts),
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="memory_writer",
                    title="更新记忆",
                    detail=detail,
                ),
            ],
        }

    def _resolve_document_ids(self, requested_ids: list[str] | None) -> list[str]:
        if requested_ids:
            return [
                document_id
                for document_id in requested_ids
                if (document := self.store.get_document(document_id)) and document.status == "ready"
            ]
        return [document.id for document in self.store.list_documents() if document.status == "ready"]

    def _emit_status(self, message: str) -> None:
        self._emit_graph_event("status", message)

    def _emit_token(self, text: str) -> None:
        if text:
            self._emit_graph_event("token", text)

    def _emit_graph_event(self, event_type: str, payload: object) -> None:
        try:
            get_stream_writer()({"type": event_type, "payload": payload})
        except Exception:
            # The same node methods are still callable in unit tests or direct
            # invocations outside a LangGraph stream, where no stream writer is
            # installed.
            return

    def _agent_event_from_graph_payload(self, payload: Any) -> AgentEvent | None:
        if isinstance(payload, AgentEvent):
            return payload
        if not isinstance(payload, dict):
            return status_event(str(payload))
        event_type = str(payload.get("type") or "")
        event_payload = payload.get("payload", "")
        if event_type == "status":
            return status_event(str(event_payload))
        if event_type in {"token", "chunk"}:
            return token_event(str(event_payload))
        if event_type == "error":
            return AgentEvent(type="error", payload=str(event_payload))
        if event_type == "final":
            return final_event(event_payload)
        return status_event(str(event_payload or payload))
