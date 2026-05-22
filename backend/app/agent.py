from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.app.agent_parts.events import AgentEvent, final_event, status_event, token_event
from backend.app.agent_parts.answering import AgentAnsweringMixin
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
from backend.app.models import AskRequest, AskResponse, EvidenceItem, RagTrace, RuntimeStep
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
    ) -> None:
        self.settings = settings
        self.store = store
        self.vector_store = vector_store
        self.model_clients = model_clients
        self.memory = memory
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
        visible_evidence = self._visible_evidence_for_answer(answer, evidence)
        runtime = state.get("runtime", [])
        final_prompt_evidence = state.get("final_prompt_evidence", [])
        evidence_judgments = state.get("evidence_judgments", [])
        verification = state.get("verification", {})
        compound_tasks = state.get("compound_tasks") or []
        task_parse_reason = state.get("task_parse_reason") or ""
        intent = state.get("intent") or self._classify_question_intent(request.question)
        retrieval_strategy = state.get("retrieval_strategy") or self._retrieval_strategy_for_question(
            request.question
        )
        retrieval_pipeline = state.get("retrieval_pipeline", "")
        ranking_method = state.get("ranking_method", "")
        answer_strategy = state.get("answer_strategy") or "model_answer"
        fallback_used = bool(state.get("fallback_used", False))
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
        )
        retrieval_debug = self._build_retrieval_debug(
            question=request.question,
            evidence=evidence,
            retrieval_strategy=retrieval_strategy,
            answer=answer,
            final_prompt_evidence=final_prompt_evidence,
        )

        conversation_id = state["conversation_id"]
        self.store.save_message(
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
                filter_document_ids=request.document_ids,
                retrieved_count=len(evidence),
                final_prompt_evidence=final_prompt_evidence,
                intent=intent,
                retrieval_strategy=retrieval_strategy,
                retrieval_pipeline=retrieval_pipeline,
                ranking_method=ranking_method,
                answer_strategy=answer_strategy,
                fallback_used=fallback_used,
                evidence_quality=evidence_quality,
                diagnosis=diagnosis,
                retrieval_debug=retrieval_debug,
                compound_tasks=compound_tasks,
                task_parse_reason=task_parse_reason,
                evidence_judgments=evidence_judgments,
                verification=verification,
            ),
            memory_used=self.memory.build_memory_context(conversation_id),
        )

    def _visible_evidence_for_answer(
        self,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        cited_ids = set(self._citation_ids_from_answer(answer))
        if not cited_ids:
            return []
        return [item for item in evidence if item.citation_id in cited_ids]

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
        facts = self.memory.build_memory_context(state["conversation_id"])
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
        facts = self.memory.build_memory_context(state["conversation_id"])
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
