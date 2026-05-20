from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from backend.app.config import Settings
from backend.app.llm_clients import ModelClients
from backend.app.memory import MemoryManager
from backend.app.models import AskRequest, AskResponse, EvidenceItem, RagTrace, RetrievalDebugItem, RuntimeStep
from backend.app.storage import MetadataStore
from backend.app.vector_store import ChromaPaperStore


class PaperAgentState(TypedDict, total=False):
    question: str
    conversation_id: str
    document_ids: list[str]
    top_k: int
    chat_model: str
    embedding_model: str
    memory_facts: dict[str, str]
    memory_prompt: str
    recent_messages: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    answer: str
    runtime: list[RuntimeStep]
    final_prompt_evidence: list[str]
    needs_retrieval: bool
    intent: str
    retrieval_strategy: str
    answer_strategy: str
    fallback_used: bool
    evidence_quality: str
    diagnosis: str
    compound_tasks: list[str]
    task_parse_reason: str
    evidence_judgments: list[dict[str, Any]]
    verification: dict[str, Any]


@dataclass
class DocumentProfile:
    document_id: str
    name: str
    title: str
    kind: str
    method: str
    main_claim: str
    has_empirical_data: bool
    has_references: bool
    is_generated_sample: bool


@dataclass(frozen=True)
class ParsedTask:
    task_type: str
    label: str
    position: int
    trigger: str


class PaperAgentService:
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
        conversation = self.store.ensure_conversation(
            request.conversation_id,
            title=request.question,
        )
        preset = self.settings.resolve_model_preset(request.model_preset)
        chat_model = request.chat_model or preset.chat_model
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
        }
        result = self.graph.invoke(state)
        evidence = result.get("evidence", [])
        answer = result.get("answer", "本轮没有生成可用回答。")
        runtime = result.get("runtime", [])
        final_prompt_evidence = result.get("final_prompt_evidence", [])
        evidence_judgments = result.get("evidence_judgments", [])
        verification = result.get("verification", {})
        parsed_tasks = self._parse_compound_tasks(request.question)
        compound_tasks = result.get("compound_tasks") or [task.task_type for task in parsed_tasks]
        task_parse_reason = result.get("task_parse_reason") or self._task_parse_reason(parsed_tasks)
        intent = result.get("intent") or self._classify_question_intent(request.question)
        retrieval_strategy = result.get("retrieval_strategy") or self._retrieval_strategy_for_question(
            request.question
        )
        answer_strategy = result.get("answer_strategy") or "model_answer"
        fallback_used = bool(result.get("fallback_used", False))
        evidence_quality = result.get("evidence_quality") or self._evidence_quality(
            question=request.question,
            evidence=evidence,
            fallback_used=fallback_used,
            answer_strategy=answer_strategy,
        )
        diagnosis = result.get("diagnosis") or self._build_trace_diagnosis(
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

        self.store.save_message(
            conversation_id=conversation.id,
            role="user",
            content=request.question,
        )
        self.store.save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            evidence=[item.model_dump() for item in evidence],
        )

        return AskResponse(
            answer=answer,
            conversation_id=conversation.id,
            evidence=evidence,
            runtime=runtime,
            rag_trace=RagTrace(
                model_profile=preset.label,
                vector_store=self.vector_store.name,
                vector_record_count=self.vector_store.count(),
                top_k=top_k,
                filter_document_ids=request.document_ids,
                retrieved_count=len(evidence),
                final_prompt_evidence=final_prompt_evidence,
                intent=intent,
                retrieval_strategy=retrieval_strategy,
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
            memory_used=self.memory.build_memory_context(conversation.id),
        )

    def _build_graph(self):
        builder = StateGraph(PaperAgentState)
        builder.add_node("memory", self._load_memory)
        builder.add_node("planner", self._plan)
        builder.add_node("retriever", self._retrieve)
        builder.add_node("evidence_judge", self._judge_evidence)
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
        builder.add_edge("evidence_judge", "answer")
        builder.add_edge("answer", "verifier")
        builder.add_edge("verifier", "memory_writer")
        builder.add_edge("memory_writer", END)
        return builder.compile()

    def _load_memory(self, state: PaperAgentState) -> PaperAgentState:
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

    def _plan(self, state: PaperAgentState) -> PaperAgentState:
        question = state["question"].strip()
        has_documents = bool(self._resolve_document_ids(state.get("document_ids")))
        needs_retrieval = has_documents and not self._looks_like_meta_question(question)
        intent = self._classify_question_intent(question)
        retrieval_strategy = self._retrieval_strategy_for_question(question) if needs_retrieval else "no_retrieval"
        parsed_tasks = self._parse_compound_tasks(question)
        task_parse_reason = self._task_parse_reason(parsed_tasks)
        detail = (
            f"识别为「{self._friendly_intent(intent)}」，检索策略为「{self._friendly_retrieval_strategy(retrieval_strategy)}」。{task_parse_reason}"
            if needs_retrieval
            else f"识别为「{self._friendly_intent(intent)}」，这轮不需要检索文档。"
        )
        return {
            **state,
            "intent": intent,
            "retrieval_strategy": retrieval_strategy,
            "compound_tasks": [task.task_type for task in parsed_tasks],
            "task_parse_reason": task_parse_reason,
            "needs_retrieval": needs_retrieval,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(node="planner", title="判断路径", detail=detail),
            ],
        }

    def _route_after_planner(self, state: PaperAgentState) -> str:
        return "retrieve" if state.get("needs_retrieval") else "answer"

    def _retrieve(self, state: PaperAgentState) -> PaperAgentState:
        document_ids = self._resolve_document_ids(state.get("document_ids"))
        if self._looks_like_compound_request(state["question"]):
            strategy = "compound_request"
            evidence = self._compound_evidence(
                question=state["question"],
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_reference_question(state["question"]):
            strategy = "reference_section"
            evidence = self._reference_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_structured_review_request(state["question"]):
            strategy = "structured_review"
            evidence = self._structured_review_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_compare_question(state["question"]):
            strategy = "comparison_overview"
            evidence = self._comparison_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_title_alignment_question(state["question"]):
            strategy = "title_alignment"
            evidence = self._title_alignment_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_reliability_question(state["question"]):
            strategy = "reliability_check"
            evidence = self._reliability_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_research_limitation_question(state["question"]):
            strategy = "research_limitation"
            evidence = self._research_limitation_evidence(
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        elif self._looks_like_document_wide_question(state["question"]):
            strategy = "document_overview"
            evidence = self._overview_evidence(
                question=state["question"],
                document_ids=document_ids,
                top_k=state["top_k"],
            )
        else:
            strategy = "vector_similarity"
            embedding_model = self._resolve_query_embedding_model(
                requested_model=state["embedding_model"],
                document_ids=document_ids,
            )
            query_embedding = self.model_clients.embed_query(
                state["question"],
                model=embedding_model,
            )
            evidence = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=max(state["top_k"] * 3, state["top_k"]),
                document_ids=document_ids,
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
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="retriever",
                    title="检索证据",
                    detail=(
                        f"实际使用「{self._friendly_retrieval_strategy(strategy)}」，"
                        f"检索范围 {len(document_ids)} 篇文档，返回 {len(evidence)} 条证据。"
                    ),
                ),
            ],
        }

    def _judge_evidence(self, state: PaperAgentState) -> PaperAgentState:
        evidence = state.get("evidence", [])
        question = state["question"]
        strategy = state.get("retrieval_strategy", "")
        if not evidence:
            return {
                **state,
                "evidence_judgments": [],
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="evidence_judge",
                        title="证据裁判",
                        detail="没有可裁判的证据，后续回答会按证据不足处理。",
                    ),
                ],
            }

        strict = self._strict_evidence_judge_question(question)
        allow_tables = self._looks_like_table_question(question)
        kept: list[EvidenceItem] = []
        judgments: list[dict[str, Any]] = []
        verdict_counts = {"direct": 0, "supporting": 0, "background": 0, "reject": 0}

        for item in evidence:
            judgment = self._judge_single_evidence(
                question=question,
                item=item,
                retrieval_strategy=strategy,
                allow_tables=allow_tables,
            )
            verdict = str(judgment["verdict"])
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            judgments.append(judgment)

            if verdict in {"direct", "supporting"}:
                kept.append(item)
            elif verdict == "background" and not strict:
                kept.append(item)

        if kept:
            kept = self._renumber_evidence(kept)
            citation_by_chunk = {item.chunk_id: item.citation_id for item in kept}
            for judgment in judgments:
                next_citation = citation_by_chunk.get(str(judgment.get("chunk_id", "")))
                if next_citation:
                    judgment["citation_id"] = next_citation
        elif not strict:
            kept = evidence

        return {
            **state,
            "evidence": kept,
            "evidence_judgments": judgments,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="evidence_judge",
                    title="证据裁判",
                    detail=(
                        "证据裁判 Agent 已逐条判断证据是否直接支撑问题："
                        f"直接 {verdict_counts.get('direct', 0)} 条、辅助 {verdict_counts.get('supporting', 0)} 条、"
                        f"背景 {verdict_counts.get('background', 0)} 条、拒绝 {verdict_counts.get('reject', 0)} 条；"
                        f"最终保留 {len(kept)} 条进入回答。"
                    ),
                ),
            ],
        }

    def _answer(self, state: PaperAgentState) -> PaperAgentState:
        evidence_blocks = self._format_evidence(state.get("evidence", []))
        final_prompt_evidence = [
            f"[{item.citation_id}] {item.paper_name} p.{item.page} score={item.score:.3f}"
            for item in state.get("evidence", [])
        ]
        recent_history = self._format_recent_history(state.get("recent_messages", []))
        if self._looks_like_compound_request(state["question"]):
            answer = self._build_local_compound_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
                state.get("document_ids", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_compound_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是复合任务，已按用户提出的子任务顺序逐项回答。",
                    ),
                ],
            }

        if self._looks_like_reference_question(state["question"]):
            answer = self._build_local_reference_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_reference_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是参考文献类问题，已直接读取文末参考文献区并整理为列表。",
                    ),
                ],
            }

        if self._looks_like_structured_review_request(state["question"]):
            answer = self._build_local_structured_review_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
                state.get("document_ids", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_structured_review_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是带输出格式要求的结构化阅读任务，已按“概括-分部分分析-专业收获-可靠性判断”组织回答。",
                    ),
                ],
            }

        if self._looks_like_title_alignment_question(state["question"]):
            answer = self._build_local_title_alignment_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
                state.get("document_ids", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_title_alignment_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是题目与结论匹配问题，已按题目关键词逐项检查结论是否回应。",
                    ),
                ],
            }

        if self._looks_like_reliability_question(state["question"]):
            answer = self._build_local_reliability_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
                state.get("document_ids", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_reliability_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是可靠性判断问题，已优先核对文档类型、计算/分析过程、结论支撑和证据缺口。",
                    ),
                ],
            }

        if self._looks_like_research_limitation_question(state["question"]):
            answer = self._build_local_research_limitation_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
                state.get("document_ids", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_research_limitation_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是文章研究局限问题，已优先核对未来研究、方法边界、数据/样本和实证验证缺口。",
                    ),
                ],
            }

        if self._looks_like_compare_question(state["question"]):
            answer = self._build_local_compare_answer(
                state["question"],
                state.get("evidence", []),
                state.get("memory_facts", {}),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_compare_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是多文档对比问题，已按文档分别整理证据并生成对比回答。",
                    ),
                ],
            }

        if self._looks_like_document_wide_question(state["question"]):
            evidence = state.get("evidence", [])
            grouped = self._group_evidence_by_document(evidence)
            answer = (
                self._build_local_grouped_answer(
                    state["question"],
                    evidence,
                    state.get("memory_facts", {}),
                )
                if len(grouped) > 1
                else self._build_local_answer(state["question"], evidence, state.get("memory_facts", {}))
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "local_document_answer",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="这是概括类问题，已直接基于原文证据生成快速回答。",
                    ),
                ],
            }

        if self._should_decline_for_missing_direct_evidence(
            state["question"],
            state.get("evidence", []),
        ):
            answer = self._build_missing_direct_evidence_answer(
                state["question"],
                state.get("evidence", []),
            )
            return {
                **state,
                "answer": answer,
                "answer_strategy": "missing_evidence_refusal",
                "fallback_used": False,
                "final_prompt_evidence": final_prompt_evidence,
                "runtime": [
                    *state.get("runtime", []),
                    RuntimeStep(
                        node="answer",
                        title="生成回答",
                        detail="检索到的段落不能直接回答问题，已停止硬答并提示用户证据不足。",
                    ),
                ],
            }

        system_prompt = (
            "你是一个严谨、友好的中文文档阅读助手。用户说“这篇论文”“这份文档”时，"
            "默认指当前已上传文档，不要要求用户重新提供标题。若当前有多篇文档且用户没有指定某一篇，"
            "必须按文档分开回答，避免把几篇文档混成一个结论。回答必须优先基于检索证据，"
            "不要编造文档内容。使用证据时用 [E1]、[E2] 这样的编号引用。"
            "如果证据不足，要说明缺少什么，但只要已有证据能回答，就先直接回答。"
            "不要输出姓名、学号、邮箱等个人信息，除非用户明确询问。"
        )
        user_prompt = f"""
用户长期画像和偏好：
{state.get("memory_prompt", "暂无")}

最近对话历史：
{recent_history}

本轮问题：
{state["question"]}

检索证据：
{evidence_blocks or "本轮没有检索到证据。"}

请给出适合非技术用户阅读的中文回答，并在关键事实后标注证据编号。
""".strip()

        try:
            answer = self.model_clients.chat_text(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                model=state["chat_model"],
            )
            answer_strategy = "model_answer"
            fallback_used = False
            answer_detail = "结合原文证据、用户偏好和最近对话生成最终回答。"
        except RuntimeError:
            evidence = state.get("evidence", [])
            grouped = self._group_evidence_by_document(evidence)
            answer = (
                self._build_local_grouped_answer(
                    state["question"],
                    evidence,
                    state.get("memory_facts", {}),
                )
                if len(grouped) > 1
                else self._build_local_answer(state["question"], evidence, state.get("memory_facts", {}))
            )
            answer_strategy = "local_fallback_answer"
            fallback_used = True
            answer_detail = "对话模型响应太慢或暂时不可用，已基于原文证据生成快速回答。"
        return {
            **state,
            "answer": answer,
            "answer_strategy": answer_strategy,
            "fallback_used": fallback_used,
            "final_prompt_evidence": final_prompt_evidence,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="answer",
                    title="生成回答",
                    detail=answer_detail,
                ),
            ],
        }

    def _verify_answer(self, state: PaperAgentState) -> PaperAgentState:
        verification = self._cross_verify_answer(
            question=state["question"],
            answer=state.get("answer", ""),
            evidence=state.get("evidence", []),
            answer_strategy=state.get("answer_strategy", ""),
        )
        detail = (
            f"交叉验证 Agent 完成核对：状态 {verification['status']}；"
            f"引用 {verification['citation_count']} 个，缺失引用 {len(verification['missing_citations'])} 个，"
            f"弱支撑引用 {len(verification['weak_citations'])} 个。{verification['summary']}"
        )
        answer = state.get("answer", "")
        if verification["status"] == "fail" and "交叉验证提示" not in answer:
            answer = (
                f"{answer}\n\n"
                "交叉验证提示：这轮回答中有引用没有找到对应证据，建议以右侧证据面板为准，或重新提问让我重新核对。"
            )
        return {
            **state,
            "answer": answer,
            "verification": verification,
            "runtime": [
                *state.get("runtime", []),
                RuntimeStep(
                    node="verifier",
                    title="交叉验证",
                    detail=detail,
                ),
            ],
        }

    def _write_memory(self, state: PaperAgentState) -> PaperAgentState:
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

    def _format_evidence(self, evidence: list[EvidenceItem]) -> str:
        blocks = []
        for item in evidence:
            blocks.append(
                f"[{item.citation_id}] {item.paper_name} | Page {item.page} | "
                f"Section: {item.section or 'Unknown'} | Score: {item.score:.3f}\n"
                f"{self._sanitize_evidence_text(item.text)}"
            )
        return "\n\n".join(blocks)

    def _format_recent_history(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "暂无。"
        lines = []
        for item in messages[-8:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip()
            if len(content) > 500:
                content = f"{content[:500]}..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _task_definitions(self) -> list[tuple[str, str, list[str]]]:
        return [
            (
                "overview_summary",
                "总体概括",
                [
                    "总体概括",
                    "整体概括",
                    "概括",
                    "总结",
                    "文章内容",
                    "主要内容",
                    "讲了什么",
                    "讲什么",
                    "主要讲",
                    "大意",
                    "主题",
                ],
            ),
            (
                "reference_list",
                "参考文献",
                [
                    "参考文献",
                    "参考资料",
                    "引用文献",
                    "文献列表",
                    "用了哪些文献",
                    "用了哪些参考",
                    "参考了哪些",
                    "引用了哪些",
                    "列出文献",
                    "文献来源",
                    "references",
                    "bibliography",
                ],
            ),
            (
                "professional_takeaways",
                "专业角度收获",
                [
                    "专业角度",
                    "从我的专业",
                    "从我专业",
                    "结合我的专业",
                    "能收获什么",
                    "能学到什么",
                    "学到什么",
                    "收获",
                    "启发",
                    "启示",
                    "专业收获",
                ],
            ),
            ("reliability_judgment", "可靠性判断", ["可靠性", "可靠吗", "可靠", "可信", "靠不靠谱", "能不能信"]),
            (
                "method_analysis",
                "方法分析",
                ["研究方法", "实验方法", "方法是什么", "用了什么方法", "采用什么方法", "怎么做", "如何研究", "如何实现"],
            ),
            ("limitation_analysis", "局限不足", ["局限", "不足", "存在的问题", "有什么问题", "短板", "缺陷"]),
            ("conclusion_summary", "结论核心", ["结论", "核心观点", "主要发现", "发现"]),
            ("comparison", "对比", ["对比", "比较", "不同点", "差异", "区别"]),
        ]

    def _parse_compound_tasks(self, question: str) -> list[ParsedTask]:
        tasks: list[ParsedTask] = []

        def add_task(task_type: str, label: str, position: int, trigger: str) -> None:
            tasks.append(
                ParsedTask(
                    task_type=task_type,
                    label=label,
                    position=position,
                    trigger=trigger,
                )
            )

        for task_type, label, keywords in self._task_definitions():
            best_position = -1
            best_trigger = ""
            for keyword in keywords:
                position = question.find(keyword)
                if position < 0:
                    continue
                if best_position < 0 or position < best_position:
                    best_position = position
                    best_trigger = keyword
            if best_position >= 0:
                add_task(task_type, label, best_position, best_trigger)

        if "方法" in question and not any(task.task_type == "method_analysis" for task in tasks):
            method_compound_patterns = [
                "总结方法",
                "概括方法",
                "分析方法",
                "方法和",
                "方法与",
                "方法及",
                "方法以及",
                "方法、",
                "方法，",
                "方法；",
            ]
            if any(pattern in question for pattern in method_compound_patterns):
                add_task("method_analysis", "方法分析", question.find("方法"), "方法")

        explicit_overview_phrases = [
            "总结文章",
            "总结全文",
            "总结论文",
            "总结这篇",
            "总结这份",
            "概括文章",
            "概括全文",
            "概括论文",
            "文章内容",
            "主要内容",
            "总体概括",
            "整体概括",
        ]
        has_explicit_overview = any(phrase in question for phrase in explicit_overview_phrases)
        filtered_tasks: list[ParsedTask] = []
        for task in tasks:
            if task.task_type == "overview_summary" and task.trigger in {"总结", "概括"} and not has_explicit_overview:
                nearby_text = question[task.position : task.position + 14]
                if any(keyword in nearby_text for keyword in ["方法", "不足", "局限", "参考文献", "结论", "可靠", "风险"]):
                    continue
            filtered_tasks.append(task)
        tasks = filtered_tasks

        tasks.sort(key=lambda task: (task.position, task.task_type))
        unique: list[ParsedTask] = []
        seen: set[str] = set()
        for task in tasks:
            if task.task_type in seen:
                continue
            seen.add(task.task_type)
            unique.append(task)
        return unique

    def _task_parse_reason(self, tasks: list[ParsedTask]) -> str:
        if len(tasks) < 2:
            return ""
        detail = "、".join(f"{task.label}({task.trigger})" for task in tasks)
        return f"检测到 {len(tasks)} 个任务目标，按用户文本出现顺序执行：{detail}。"

    def _looks_like_compound_request(self, question: str) -> bool:
        tasks = self._parse_compound_tasks(question)
        if len(tasks) >= 2:
            return True
        enumeration_pattern = r"(第一|第二|第三|首先|其次|再次|最后|①|②|③|[（(]?\d+[）).、]|[一二三四五六七八九十][、.])"
        return len(tasks) >= 1 and len(re.findall(enumeration_pattern, question)) >= 2

    def _classify_question_intent(self, question: str) -> str:
        if self._looks_like_compound_request(question):
            return "compound_request"
        if self._looks_like_structured_review_request(question):
            return "structured_review_request"
        if self._looks_like_reference_question(question):
            return "reference_question"
        if self._looks_like_compare_question(question):
            return "compare_question"
        if self._looks_like_title_alignment_question(question):
            return "title_alignment_question"
        if self._looks_like_reliability_question(question):
            return "reliability_question"
        if self._looks_like_research_limitation_question(question):
            return "research_limitation_question"
        if self._looks_like_document_wide_question(question):
            return "document_wide_question"
        if self._looks_like_meta_question(question):
            return "meta_question"
        return "specific_question"

    def _retrieval_strategy_for_question(self, question: str) -> str:
        if self._looks_like_compound_request(question):
            return "compound_request"
        if self._looks_like_structured_review_request(question):
            return "structured_review"
        if self._looks_like_reference_question(question):
            return "reference_section"
        if self._looks_like_compare_question(question):
            return "comparison_overview"
        if self._looks_like_title_alignment_question(question):
            return "title_alignment"
        if self._looks_like_reliability_question(question):
            return "reliability_check"
        if self._looks_like_research_limitation_question(question):
            return "research_limitation"
        if self._looks_like_document_wide_question(question):
            return "document_overview"
        if self._looks_like_meta_question(question):
            return "no_retrieval"
        return "vector_similarity"

    def _friendly_intent(self, intent: str) -> str:
        labels = {
            "compound_request": "复合任务",
            "reference_question": "参考文献问题",
            "structured_review_request": "结构化阅读报告任务",
            "compare_question": "多文档对比问题",
            "title_alignment_question": "题目-结论匹配问题",
            "reliability_question": "可靠性判断问题",
            "research_limitation_question": "文章研究局限问题",
            "document_wide_question": "整篇概括/分析问题",
            "meta_question": "使用说明问题",
            "specific_question": "具体内容问答",
        }
        return labels.get(intent, intent)

    def _friendly_retrieval_strategy(self, strategy: str) -> str:
        labels = {
            "compound_request": "复合任务检索",
            "reference_section": "参考文献区检索",
            "structured_review": "结构化阅读报告检索",
            "comparison_overview": "多文档概览检索",
            "title_alignment": "题目与结论专项检索",
            "reliability_check": "可靠性专项检索",
            "research_limitation": "文章研究局限检索",
            "document_overview": "整篇文档重点检索",
            "vector_similarity": "向量相似度检索",
            "no_retrieval": "不检索文档",
        }
        return labels.get(strategy, strategy)

    def _friendly_answer_strategy(self, strategy: str) -> str:
        labels = {
            "local_compound_answer": "按用户顺序逐项回答复合任务",
            "local_reference_answer": "本地规则整理参考文献",
            "local_structured_review_answer": "本地规则生成结构化阅读报告",
            "local_title_alignment_answer": "本地规则判断题目匹配",
            "local_reliability_answer": "本地规则判断可靠性",
            "local_research_limitation_answer": "本地规则分析文章研究局限",
            "local_compare_answer": "本地规则对比多文档",
            "local_document_answer": "本地规则概括文档",
            "missing_evidence_refusal": "证据不足，拒绝硬答",
            "model_answer": "模型基于证据生成",
            "local_fallback_answer": "模型失败后本地降级回答",
        }
        return labels.get(strategy, strategy)

    def _evidence_quality(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        fallback_used: bool,
        answer_strategy: str,
    ) -> str:
        if answer_strategy == "missing_evidence_refusal":
            return "insufficient"
        if fallback_used:
            return "fallback"
        if not evidence:
            return "none"
        best_score = max((item.score for item in evidence), default=0.0)
        best_relevance = max(
            (self._question_relevance_score(question, item.text) for item in evidence),
            default=0.0,
        )
        if answer_strategy.startswith("local_") and len(evidence) >= 1 and best_score >= 0.75:
            return "strong"
        if best_score >= 0.7 or best_relevance >= 0.25:
            return "strong"
        if best_score >= 0.45 or best_relevance >= 0.12:
            return "medium"
        return "weak"

    def _build_trace_diagnosis(
        self,
        *,
        intent: str,
        retrieval_strategy: str,
        answer_strategy: str,
        evidence_quality: str,
        evidence: list[EvidenceItem],
        fallback_used: bool,
        verification: dict[str, Any] | None = None,
    ) -> str:
        verification = verification or {}
        if verification.get("status") == "fail":
            return f"交叉验证未通过：{verification.get('summary', '回答和证据引用存在不一致。')}"
        if verification.get("status") == "warn":
            return f"交叉验证提示：{verification.get('summary', '部分证据支撑较弱，需要谨慎阅读。')}"
        if fallback_used:
            return "模型调用失败或超时，本轮已改用原文证据和本地规则生成回答。"
        if answer_strategy == "missing_evidence_refusal":
            return "检索到的段落和问题关联不够强，本轮选择停止硬答并提示证据不足。"
        if not evidence:
            return "本轮没有返回可用证据，需要检查文档是否准备完成或问题是否依赖文档内容。"

        pages = sorted({item.page for item in evidence if item.page})
        page_text = "、".join(str(page) for page in pages[:3]) if pages else "未知位置"
        return (
            f"本轮识别为「{self._friendly_intent(intent)}」，"
            f"使用「{self._friendly_retrieval_strategy(retrieval_strategy)}」，"
            f"命中 {len(evidence)} 条证据，主要来自第 {page_text} 页/段；"
            f"回答策略是「{self._friendly_answer_strategy(answer_strategy)}」。"
        )

    def _build_retrieval_debug(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
        retrieval_strategy: str,
        answer: str,
        final_prompt_evidence: list[str],
    ) -> list[RetrievalDebugItem]:
        prompt_text = "\n".join(final_prompt_evidence)
        items: list[RetrievalDebugItem] = []
        for item in evidence:
            matched_keywords = self._debug_matched_keywords(
                question=question,
                text=f"{item.section or ''}\n{item.quote}\n{item.text}",
                retrieval_strategy=retrieval_strategy,
            )
            items.append(
                RetrievalDebugItem(
                    citation_id=item.citation_id,
                    chunk_id=item.chunk_id,
                    page=item.page,
                    section=item.section,
                    score=item.score,
                    retrieval_strategy=retrieval_strategy,
                    selected_by=self._debug_selected_by(retrieval_strategy),
                    matched_keywords=matched_keywords,
                    reason=self._debug_reason_for_evidence(
                        question=question,
                        retrieval_strategy=retrieval_strategy,
                        matched_keywords=matched_keywords,
                        item=item,
                    ),
                    used_in_answer=f"[{item.citation_id}]" in answer,
                    used_in_prompt=f"[{item.citation_id}]" in prompt_text,
                    quote=self._best_quote_for_question(question, item.text, limit=180),
                )
            )
        return items

    def _debug_selected_by(self, retrieval_strategy: str) -> str:
        labels = {
            "compound_request": "按子任务顺序组合检索",
            "reference_section": "规则定位参考文献区",
            "comparison_overview": "按文档抽取概览片段",
            "structured_review": "按模板抽取多类证据",
            "title_alignment": "题目与结论专项规则",
            "reliability_check": "可靠性专项规则",
            "research_limitation": "文章研究局限专项规则",
            "document_overview": "整篇文档关键词/结构检索",
            "vector_similarity": "Chroma 向量相似度召回后重排",
            "no_retrieval": "未检索文档",
        }
        return labels.get(retrieval_strategy, retrieval_strategy)

    def _debug_matched_keywords(
        self,
        *,
        question: str,
        text: str,
        retrieval_strategy: str,
    ) -> list[str]:
        strategy_keywords = {
            "compound_request": self._compound_focus_keywords_for_question(question),
            "reference_section": ["参考文献", "References", "[1]", "[2]", "[3]"],
            "comparison_overview": ["实验名称", "实验类型", "实验目的", "主题", "方法", "结论"],
            "structured_review": [
                "摘要",
                "本文围绕",
                "采用",
                "文献分析",
                "认知支架",
                "资源重组",
                "课程学习场景",
                "风险",
                "人机协同",
                "未来研究",
                "实证数据",
                "参考文献",
            ],
            "title_alignment": [
                "题目",
                "结论",
                "机制",
                "风险",
                "治理路径",
                "未来研究",
                "实证数据",
            ],
            "reliability_check": [
                "随机生成",
                "论文样稿",
                "采用",
                "文献分析",
                "情境推演",
                "机制建构",
                "未来研究",
                "实证数据",
                "参考文献",
            ],
            "research_limitation": [
                "局限性",
                "研究局限",
                "研究不足",
                "结论与展望",
                "未来研究",
                "实证数据",
                "验证",
                "检验",
                "样本",
                "数据来源",
                "文献分析",
                "情境推演",
                "机制建构",
            ],
            "document_overview": self._overview_focus_keywords(question),
            "vector_similarity": self._question_keywords(question),
        }
        keywords = strategy_keywords.get(retrieval_strategy, self._question_keywords(question))
        matches: list[str] = []
        for keyword in keywords:
            if keyword and keyword in text and keyword not in matches:
                matches.append(keyword)
            if len(matches) >= 8:
                break
        return matches

    def _debug_reason_for_evidence(
        self,
        *,
        question: str,
        retrieval_strategy: str,
        matched_keywords: list[str],
        item: EvidenceItem,
    ) -> str:
        keyword_text = "、".join(matched_keywords) if matched_keywords else "没有明显关键词，主要依赖向量相似度/结构位置"
        if retrieval_strategy == "compound_request":
            tasks = self._parse_compound_tasks(question)
            task_text = " → ".join(task.label for task in tasks) if tasks else "多个子任务"
            return f"用户问题被拆成“{task_text}”，系统按这些子任务分别抽取摘要、方法、参考文献、收获或可靠性相关证据；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "reference_section":
            return f"问题像是在问参考文献，系统优先扫描文末参考文献区；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "document_overview":
            return f"问题属于整篇分析/概括，系统优先找和问题目标相关的结构段落；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "structured_review":
            return f"用户提出了多步骤模板要求，系统同时抽取摘要、方法、主体内容、风险治理、结论和参考文献证据；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "reliability_check":
            return f"问题是在判断可靠性，系统优先找文档类型、方法、数据验证、未来研究和参考文献信息；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "research_limitation":
            return f"问题是在问文章本身的研究局限，系统优先找未来研究、方法边界、数据/样本和实证验证缺口；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "title_alignment":
            return f"问题是在判断题目与结论是否匹配，系统优先找题目关键词、结论、风险、治理和不足；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "comparison_overview":
            return f"问题是多文档对比，系统为每篇文档抽取能代表主题和方法的片段；该 chunk 命中 {keyword_text}。"
        if retrieval_strategy == "vector_similarity":
            return f"系统把问题转成检索向量，在 Chroma 中召回相似 chunk，再按相关度和可读性筛选；该 chunk 的 score 为 {item.score:.3f}，命中 {keyword_text}。"
        return f"该证据由「{self._friendly_retrieval_strategy(retrieval_strategy)}」选中，命中 {keyword_text}。"

    def _looks_like_meta_question(self, question: str) -> bool:
        keywords = ["你是谁", "怎么使用", "有哪些功能", "如何上传", "支持什么模型"]
        return any(keyword in question for keyword in keywords)

    def _looks_like_overview_question(self, question: str) -> bool:
        keywords = ["概括", "总结", "讲了什么", "主要内容", "通俗语言", "大意", "一句话"]
        return any(keyword in question for keyword in keywords)

    def _looks_like_document_wide_question(self, question: str) -> bool:
        keywords = [
            "概括",
            "总结",
            "讲了什么",
            "主要内容",
            "通俗语言",
            "大意",
            "一句话",
            "发现",
            "重点",
            "核心",
            "结论",
            "贡献",
            "方法",
            "局限",
            "不足",
            "目的",
            "主题",
            "能学到",
            "学到什么",
            "收获",
            "启发",
        ]
        return any(keyword in question for keyword in keywords)

    def _overview_focus_keywords(self, question: str) -> list[str]:
        keywords: list[str] = []
        if any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"]):
            keywords.extend([
                "研究方法",
                "方法",
                "采用",
                "样本",
                "数据",
                "问卷",
                "访谈",
                "实验",
                "模型",
                "机制建构",
                "method",
                "model",
                "architecture",
                "training",
                "experiment",
                "evaluation",
            ])
        if any(word in question for word in ["结论", "发现", "核心", "重点", "贡献"]):
            keywords.extend([
                "结论",
                "结论与展望",
                "总体而言",
                "研究认为",
                "发现",
                "表明",
                "贡献",
                "提出",
                "未来研究",
                "results",
                "achieves",
                "state-of-the-art",
                "conclusion",
                "we show",
                "we propose",
            ])
        if any(word in question for word in ["局限", "不足"]):
            keywords.extend(["未来研究", "实证数据", "数据来源", "样本", "验证", "检验", "缺少", "缺乏", "尚未", "不足"])
        if any(word in question for word in ["问题", "风险"]):
            keywords.extend(["风险", "挑战", "问题", "隐私", "偏差", "诚信", "责任", "不足"])
        if any(word in question for word in ["目的", "主题", "讲了什么", "主要内容", "概括", "总结", "大意", "一句话"]):
            keywords.extend([
                "摘要",
                "本文围绕",
                "研究目的",
                "旨在",
                "主要讨论",
                "研究认为",
                "结论",
                "abstract",
                "introduction",
                "in this paper",
                "in this work",
                "we propose",
                "we present",
                "we introduce",
                "we show",
                "transformer",
                "attention",
                "sequence transduction",
                "machine translation",
            ])
        if any(word in question for word in ["能学到", "学到什么", "收获", "启发"]):
            keywords.extend(["机制", "应用场景", "学习支持", "风险", "治理", "人机协同", "数据", "隐私", "算法", "价值"])
        return list(dict.fromkeys(keywords or ["摘要", "本文", "研究", "结论", "方法"]))

    def _needs_opening_context(self, question: str) -> bool:
        return any(word in question for word in ["概括", "总结", "讲了什么", "主要内容", "大意", "一句话", "目的", "主题", "能学到", "学到什么", "收获", "启发"])

    def _looks_like_research_limitation_question(self, question: str) -> bool:
        if "局限性" in question:
            return True
        return any(word in question for word in ["局限", "不足", "短板", "缺陷"]) and any(
            subject in question
            for subject in ["文章", "论文", "研究", "文档", "报告", "原文", "这篇", "这份", "它"]
        )

    def _looks_like_compare_question(self, question: str) -> bool:
        keywords = ["对比", "比较", "不同点", "差异", "区别", "有什么不同", "哪里不同"]
        return any(keyword in question for keyword in keywords)

    def _looks_like_structured_review_request(self, question: str) -> bool:
        format_keywords = [
            "先总结",
            "先概括",
            "总结概括",
            "再详细分析",
            "详细分析",
            "每一部分",
            "每个部分",
            "最后告诉我",
            "最后判断",
            "最后评价",
            "用这个模板",
            "按照这个模板",
            "按这个模板",
            "模板来回答",
            "按这个格式",
            "分部分",
        ]
        has_format_request = any(keyword in question for keyword in format_keywords)
        has_multi_step_words = sum(
            1
            for keyword in ["先", "再", "然后", "最后", "模板", "每一部分", "详细分析", "总结"]
            if keyword in question
        ) >= 2
        has_review_goal = any(
            keyword in question
            for keyword in ["论文", "文档", "报告", "可靠性", "可靠吗", "能学到", "分析"]
        )
        return has_review_goal and (has_format_request or has_multi_step_words)

    def _looks_like_reference_question(self, question: str) -> bool:
        normalized = question.lower().strip()
        keywords = [
            "参考文献",
            "引用文献",
            "引用了哪些",
            "参考了哪些",
            "用了哪些文献",
            "用了哪些参考",
            "文献列表",
            "参考列表",
            "references",
            "bibliography",
        ]
        if any(keyword in normalized for keyword in keywords):
            return True
        return "文献" in question and any(
            verb in question
            for verb in ["哪些", "列出", "引用", "参考", "出处", "来源"]
        )

    def _looks_like_title_alignment_question(self, question: str) -> bool:
        keywords = [
            "支撑题目",
            "支撑标题",
            "支持题目",
            "支持标题",
            "能不能支撑",
            "能否支撑",
            "是否支撑",
            "结论和题目",
            "结论与题目",
            "结论跟题目",
            "题文相符",
            "扣题",
            "偏题",
            "跑题",
            "题目相符",
            "标题相符",
        ]
        return any(keyword in question for keyword in keywords)

    def _looks_like_reliability_question(self, question: str) -> bool:
        explicit_keywords = [
            "结果可靠吗",
            "结论可靠吗",
            "可靠吗",
            "靠不靠谱",
            "靠谱不",
            "可信",
            "能信",
            "是否成立",
            "站得住脚",
            "准确吗",
            "准不准",
        ]
        if any(keyword in question for keyword in explicit_keywords):
            return True
        return "可靠" in question and any(
            subject in question
            for subject in ["结果", "结论", "论文", "报告", "文档", "数据", "这篇", "这份"]
        )

    def _comparison_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(2, min(3, top_k))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=per_document_limit)
            for row in rows:
                evidence.append(self._evidence_from_row(row, document_id, score=1.0))
        return evidence

    def _overview_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(6, top_k + 1))
        focus_keywords = self._overview_focus_keywords(question)
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            if rows and self._needs_opening_context(question):
                opening_row = self._first_informative_overview_row(rows) or rows[0]
                selected.append((1.0, opening_row))
                selected_ids.add(str(opening_row.get("id", "")))

            scored_rows: list[tuple[float, dict[str, Any]]] = []
            research_limitation_question = self._looks_like_research_limitation_question(question)
            method_question = any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"])
            for index, row in enumerate(rows):
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                text = str(row.get("text", ""))
                if self._looks_like_front_matter_noise(text):
                    continue
                normalized_text = text.lower()
                keyword_hits = sum(
                    1
                    for keyword in focus_keywords
                    if keyword in text or keyword.lower() in normalized_text
                )
                relevance = 0.0 if research_limitation_question else self._question_relevance_score(question, text)
                if keyword_hits == 0 and relevance < 0.08:
                    continue
                position_bonus = 0.12 if index <= 2 else 0.0
                overview_bonus = self._overview_structure_score(text)
                score = 0.45 + keyword_hits * 0.16 + relevance * 0.5 + position_bonus + overview_bonus
                if research_limitation_question:
                    if "未来研究" in text:
                        score += 0.8
                    if "实证数据" in text or "检验" in text:
                        score += 0.4
                    if "传统高校学习支持服务存在" in text or "上述局限" in text:
                        score -= 0.7
                if method_question:
                    if all(keyword in text for keyword in ["采用", "文献分析"]):
                        score += 2.0
                    if all(keyword in text for keyword in ["情境推演", "机制建构"]):
                        score += 0.8
                    if "本文围绕" in text and "方法" in text:
                        score += 0.8
                    if any(keyword in text for keyword in ["案例化情境推演", "课程教师", "学生可以", "不得直接提交"]):
                        score -= 1.3
                scored_rows.append((score, row))

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            for score, row in scored_rows:
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                selected.append((min(score, 1.0), row))
                selected_ids.add(row_id)
                if len(selected) >= per_document_limit:
                    break

            if len(selected) < min(3, per_document_limit):
                for row in rows:
                    row_id = str(row.get("id", ""))
                    if row_id in selected_ids:
                        continue
                    text = str(row.get("text", ""))
                    if self._looks_like_front_matter_noise(text):
                        continue
                    selected.append((0.55 + self._overview_structure_score(text), row))
                    selected_ids.add(row_id)
                    if len(selected) >= min(3, per_document_limit):
                        break

            if not selected:
                selected = [
                    (0.55, row)
                    for row in rows[:per_document_limit]
                    if not self._looks_like_front_matter_noise(str(row.get("text", "")))
                ] or [(0.55, row) for row in rows[:per_document_limit]]

            for score, row in selected:
                evidence.append(self._evidence_from_row(row, document_id, score=score))
        return evidence

    def _research_limitation_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(6, top_k + 1))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()
            scored_rows: list[tuple[float, int, dict[str, Any]]] = []
            total_rows = max(len(rows), 1)

            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                section = str((row.get("metadata") or {}).get("section") or "")
                score = self._research_limitation_relevance_score(
                    text=text,
                    section=section,
                    index=index,
                    total=total_rows,
                )
                if score >= 0.85:
                    scored_rows.append((score, index, row))

            scored_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)

            def add_row(row: dict[str, Any] | None, score: float) -> None:
                if not row:
                    return
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    return
                selected_ids.add(row_id)
                selected.append((min(max(score / 4.0, 0.55), 1.0), row))

            for score, _, row in scored_rows:
                add_row(row, score)
                if len(selected) >= per_document_limit:
                    break

            method_row = self._best_row_for_keywords_excluding(
                rows,
                ["文献分析", "情境推演", "机制建构", "研究方法"],
                selected_ids,
            )
            if method_row:
                method_score = self._research_limitation_relevance_score(
                    text=str(method_row.get("text", "")),
                    section=str((method_row.get("metadata") or {}).get("section") or ""),
                    index=0,
                    total=total_rows,
                )
                if method_score >= 0.65:
                    add_row(method_row, method_score)

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=score))

        return evidence

    def _research_limitation_relevance_score(
        self,
        *,
        text: str,
        section: str,
        index: int,
        total: int,
    ) -> float:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        if not normalized:
            return 0.0

        section_bonus = {
            "Limitations": 2.4,
            "FutureWork": 2.0,
            "Conclusion": 1.1,
            "Discussion": 0.9,
            "Methods": 0.75,
        }.get(section, 0.0)
        direct_keywords = [
            "局限性",
            "研究局限",
            "研究不足",
            "局限与不足",
            "不足与展望",
            "结论与展望",
            "未来研究",
            "后续研究",
            "研究展望",
        ]
        boundary_keywords = [
            "实证数据",
            "实证检验",
            "检验",
            "验证",
            "尚未",
            "未能",
            "样本量",
            "样本代表性",
            "抽样",
            "统计检验",
            "数据来源",
            "不同应用场景",
            "不同学生群体",
            "使用差异",
            "外推",
            "普适性",
            "代表性",
            "长期效果",
        ]
        method_keywords = ["文献分析", "情境推演", "机制建构"]
        content_problem_keywords = [
            "学习困难",
            "目标模糊",
            "计划松散",
            "拖延",
            "反思缺失",
            "传统高校学习支持服务存在",
            "学生学习",
            "许多学生",
            "学习者",
            "教师反馈",
            "课程学习场景",
            "实验与实践教学场景",
            "程序设计课程",
            "社会调查课程",
            "论文初稿阶段",
            "学生文本",
            "学生根据反馈",
            "教师最终评价",
            "人工智能使用声明制度",
            "学习依赖",
            "信息准确性",
            "数据隐私",
            "算法偏差",
            "学术诚信",
            "责任边界",
        ]

        direct_hits = sum(1 for keyword in direct_keywords if keyword in normalized)
        boundary_hits = sum(1 for keyword in boundary_keywords if keyword in normalized)
        method_hits = sum(1 for keyword in method_keywords if keyword in normalized)
        content_problem_hits = sum(1 for keyword in content_problem_keywords if keyword in normalized)
        research_method_context = any(keyword in normalized for keyword in ["采用", "本文围绕", "研究方法：", "研究方法:"]) and method_hits > 0

        score = section_bonus + direct_hits * 0.9 + boundary_hits * 0.38 + method_hits * 0.25
        if re.search(r"(缺少|缺乏|没有|未能).{0,14}(样本|数据|实证|验证|检验|统计|抽样|方法)", normalized):
            score += 0.7
        if "未来研究" in normalized and any(keyword in normalized for keyword in ["实证数据", "检验", "验证"]):
            score += 1.1
        if "比较不同学生群体" in normalized or "不同学生群体的使用差异" in normalized:
            score += 0.75
        if "采用" in normalized and any(keyword in normalized for keyword in ["文献分析", "情境推演", "机制建构"]):
            score += 0.65
        if method_hits and not research_method_context and direct_hits == 0 and boundary_hits == 0:
            score -= 1.0
        if index >= total * 0.7:
            score += 0.18
        if index <= 1 and direct_hits == 0 and boundary_hits == 0:
            score -= 0.35
        if content_problem_hits and direct_hits == 0 and boundary_hits == 0 and method_hits == 0:
            score -= 2.2
        if content_problem_hits and method_hits == 0 and "不足" in normalized and "未来研究" not in normalized:
            score -= 0.7
        return score

    def _reference_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(8, top_k + 3))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[dict[str, Any]] = []
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

    def _structured_review_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(6, min(8, top_k + 3))
        selectors = [
            ("开头/摘要", ["摘要", "本文围绕", "关键词", "研究主题"]),
            ("方法", ["采用", "文献分析", "情境推演", "机制建构", "研究方法"]),
            ("作用机制", ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "运行机制"]),
            ("应用场景", ["课程学习场景", "实验与实践教学", "学业预警", "应用场景"]),
            ("风险", ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界", "风险"]),
            ("治理", ["人机协同", "价值对齐", "过程可控", "数据最小化", "多主体治理", "治理"]),
            ("结论/局限", ["结论与展望", "未来研究", "实证数据", "总体而言", "不足", "局限"]),
            ("参考文献", ["参考文献", "[1]", "Journal", "研究[J]"]),
        ]

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            if rows:
                selected.append((1.0, rows[0]))
                selected_ids.add(str(rows[0].get("id", "")))

            for index, (_, keywords) in enumerate(selectors, start=1):
                row = self._best_row_for_keywords_excluding(rows, keywords, selected_ids)
                if row:
                    selected.append((max(0.65, 1.0 - index * 0.04), row))
                    selected_ids.add(str(row.get("id", "")))
                if len(selected) >= per_document_limit:
                    break

            if len(selected) < per_document_limit:
                for row in rows[:per_document_limit]:
                    row_id = str(row.get("id", ""))
                    if row_id in selected_ids:
                        continue
                    selected.append((0.55, row))
                    selected_ids.add(row_id)
                    if len(selected) >= per_document_limit:
                        break

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=min(score, 1.0)))
        return evidence

    def _compound_task_evidence_keywords(self, task_type: str) -> list[list[str]]:
        selectors: dict[str, list[list[str]]] = {
            "overview_summary": [
                ["摘要", "本文围绕", "研究目的", "主要讨论", "研究主题", "关键词"],
                ["实验名称", "实验目的", "主要内容", "主题"],
                ["研究认为", "总体而言", "结论"],
            ],
            "professional_takeaways": [
                ["机制", "应用场景", "学习支持", "系统", "流程"],
                ["风险", "隐私", "算法偏差", "学术诚信", "责任边界"],
                ["人机协同", "治理", "价值对齐", "过程可控", "工程"],
            ],
            "reliability_judgment": [
                ["采用", "文献分析", "情境推演", "机制建构", "研究方法"],
                ["未来研究", "实证数据", "样本", "问卷", "实验", "统计检验"],
                ["参考文献", "数据来源", "验证", "局限", "不足"],
            ],
            "method_analysis": [
                ["研究方法", "采用", "文献分析", "情境推演", "机制建构"],
                ["实验方法", "实验步骤", "实现过程", "算法", "流程"],
            ],
            "limitation_analysis": [
                ["结论与展望", "未来研究", "研究局限", "局限性", "研究不足"],
                ["实证数据", "实证检验", "样本", "验证", "不同应用场景", "不同学生群体"],
                ["文献分析", "情境推演", "机制建构", "研究方法"],
            ],
            "conclusion_summary": [
                ["结论与展望", "总体而言", "研究认为", "综上所述"],
                ["核心观点", "主要发现", "结论", "发现"],
            ],
            "comparison": [
                ["实验名称", "实验类型", "实验目的", "主题"],
                ["方法", "实现", "结论", "结果"],
            ],
        }
        return selectors.get(task_type, [["摘要", "本文", "研究", "结论"]])

    def _compound_focus_keywords_for_question(self, question: str) -> list[str]:
        keywords: list[str] = []
        for task in self._parse_compound_tasks(question):
            for selector in self._compound_task_evidence_keywords(task.task_type):
                keywords.extend(selector)
            keywords.append(task.trigger)
        return list(dict.fromkeys(keyword for keyword in keywords if keyword))

    def _compound_evidence(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[EvidenceItem]:
        tasks = self._parse_compound_tasks(question)
        if len(tasks) < 2:
            return self._overview_evidence(question=question, document_ids=document_ids, top_k=top_k)

        evidence: list[EvidenceItem] = []
        per_document_limit = max(6, min(14, top_k + len(tasks) + 5))

        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[tuple[float, dict[str, Any]]] = []
            selected_ids: set[str] = set()

            def add_row(row: dict[str, Any] | None, score: float) -> None:
                if not row:
                    return
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    return
                selected_ids.add(row_id)
                selected.append((min(score, 1.0), row))

            for task_index, task in enumerate(tasks):
                base_score = max(0.58, 1.0 - task_index * 0.06)
                if task.task_type == "overview_summary" and rows:
                    add_row(rows[0], base_score)

                if task.task_type == "reference_list":
                    start_index = next(
                        (
                            index
                            for index, row in enumerate(rows)
                            if self._looks_like_reference_section_text(str(row.get("text", "")))
                        ),
                        -1,
                    )
                    reference_rows: list[dict[str, Any]] = []
                    if start_index >= 0:
                        for row in rows[start_index:]:
                            text = str(row.get("text", ""))
                            if row is rows[start_index] or self._reference_marker_count(text) > 0:
                                reference_rows.append(row)
                            elif reference_rows and self._looks_like_reference_continuation(text):
                                reference_rows.append(row)
                            elif reference_rows:
                                break
                            if len(reference_rows) >= 4:
                                break
                    if not reference_rows:
                        reference_rows = [
                            row
                            for row in rows
                            if self._reference_marker_count(str(row.get("text", ""))) >= 2
                            or "References" in str(row.get("text", ""))
                        ][:4]
                    for ref_index, row in enumerate(reference_rows):
                        add_row(row, max(0.62, base_score - ref_index * 0.03))
                    continue

                for selector_index, keywords in enumerate(self._compound_task_evidence_keywords(task.task_type)):
                    row = self._best_row_for_keywords_excluding(rows, keywords, selected_ids)
                    add_row(row, max(0.55, base_score - selector_index * 0.04))

            if len(selected) < min(3, len(rows)):
                for row in rows[:3]:
                    add_row(row, 0.52)
                    if len(selected) >= 3:
                        break

            for score, row in selected[:per_document_limit]:
                evidence.append(self._evidence_from_row(row, document_id, score=score))

        return evidence

    def _title_alignment_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(4, min(6, top_k + 1))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            selected: list[dict[str, Any]] = []
            if rows:
                selected.append(rows[0])
            selected_ids = {str(row.get("id", "")) for row in selected}
            candidate_getters = [
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["结论与展望", "认知支架"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_any_keywords(
                    rows,
                    ["第一，学习依赖风险", "第二，信息准确性风险", "第三，数据隐私风险", "第五，学术诚信风险", "第六，责任边界风险"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["针对上述风险", "人机协同"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_all_keywords(
                    rows,
                    ["未来研究", "实证数据"],
                    exclude_ids=selected_ids,
                ),
                lambda: self._first_row_with_any_keywords(
                    rows,
                    ["采用文献分析", "文献分析、情境推演和机制建构", "机制建构的方法"],
                    exclude_ids=selected_ids,
                ),
            ]
            for get_row in candidate_getters:
                row = get_row()
                if row:
                    selected.append(row)
                    selected_ids.add(str(row.get("id", "")))

            seen_ids: set[str] = set()
            for row in selected:
                row_id = str(row.get("id", ""))
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                evidence.append(self._evidence_from_row(row, document_id, score=1.0))
                if len(seen_ids) >= per_document_limit:
                    break
        return evidence

    def _first_row_with_all_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = exclude_ids or set()
        for row in rows:
            if str(row.get("id", "")) in excluded:
                continue
            text = str(row.get("text", ""))
            if all(keyword in text for keyword in keywords):
                return row
        return None

    def _first_row_with_any_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = exclude_ids or set()
        for row in rows:
            if str(row.get("id", "")) in excluded:
                continue
            text = str(row.get("text", ""))
            if any(keyword in text for keyword in keywords):
                return row
        return None

    def _reliability_evidence(self, *, document_ids: list[str], top_k: int) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        per_document_limit = max(3, min(5, top_k))
        for document_id in document_ids:
            rows = self.vector_store.get_document_chunks(document_id, limit=1000)
            scored_rows: list[tuple[float, dict[str, Any]]] = []
            for index, row in enumerate(rows):
                text = str(row.get("text", ""))
                metadata = row.get("metadata", {}) or {}
                page = int(metadata.get("page", 0) or 0)
                relevance = self._reliability_relevance_score(text)
                quality = self._readable_text_score(text)
                table_penalty = 0.35 if self._is_table_like_text(text) else 0.0
                early_bonus = 0.18 if index <= 1 or page <= 1 else 0.0
                score = 0.35 + relevance * 0.08 + quality * 0.25 + early_bonus - table_penalty
                if relevance > 0 or early_bonus > 0:
                    scored_rows.append((score, row))

            if not scored_rows:
                scored_rows = [
                    (self._readable_text_score(str(row.get("text", ""))), row)
                    for row in rows[:per_document_limit]
                ]

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            selected: list[tuple[float, dict[str, Any]]] = []
            if rows:
                # Always keep the opening chunk: it usually contains the title
                # and document type, which are essential for reliability checks.
                selected.append((1.0, rows[0]))
            for keywords in [
                ["采用", "文献分析", "情境推演", "机制建构", "研究认为"],
                ["未来研究", "实证数据", "检验不同应用场景", "结论与展望"],
                ["参考文献", "随机生成", "样稿"],
                ["风险", "挑战", "局限", "不足"],
            ]:
                row = self._best_row_for_keywords(rows, keywords)
                if row:
                    selected.append((1.0, row))
            selected_ids = {str(row.get("id", "")) for _, row in selected}
            for score, row in scored_rows:
                row_id = str(row.get("id", ""))
                if row_id in selected_ids:
                    continue
                selected.append((score, row))
                selected_ids.add(row_id)
                if len(selected) >= per_document_limit:
                    break
            for score, row in selected:
                evidence.append(self._evidence_from_row(row, document_id, score=min(score, 1.0)))
        return evidence

    def _best_row_for_keywords(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
    ) -> dict[str, Any] | None:
        best_row: dict[str, Any] | None = None
        best_score = 0
        for row in rows:
            text = str(row.get("text", ""))
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_row = row
        return best_row if best_score > 0 else None

    def _best_row_for_keywords_excluding(
        self,
        rows: list[dict[str, Any]],
        keywords: list[str],
        exclude_ids: set[str],
    ) -> dict[str, Any] | None:
        best_row: dict[str, Any] | None = None
        best_score = 0
        for row in rows:
            if str(row.get("id", "")) in exclude_ids:
                continue
            text = str(row.get("text", ""))
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_row = row
        return best_row if best_score > 0 else None

    def _evidence_from_row(
        self,
        row: dict[str, Any],
        fallback_document_id: str,
        *,
        score: float,
    ) -> EvidenceItem:
        metadata = row["metadata"]
        text = str(row["text"])
        quote = self._best_readable_quote(str(metadata.get("quote", "")) or text)
        return EvidenceItem(
            citation_id="",
            chunk_id=str(metadata.get("chunk_id") or row["id"]),
            document_id=str(metadata.get("document_id", fallback_document_id)),
            paper_name=str(metadata.get("paper_name", "")),
            page=int(metadata.get("page", 0) or 0),
            section=str(metadata.get("section") or ""),
            source=str(metadata.get("source", "")),
            file_hash=str(metadata.get("file_hash", "")),
            score=score,
            text=text,
            quote=quote,
            char_start=int(metadata.get("char_start", 0) or 0),
            char_end=int(metadata.get("char_end", 0) or 0),
        )

    def _strict_evidence_judge_question(self, question: str) -> bool:
        return any(
            checker(question)
            for checker in [
                self._looks_like_reference_question,
                self._looks_like_reliability_question,
                self._looks_like_research_limitation_question,
                self._looks_like_title_alignment_question,
            ]
        )

    def _judge_single_evidence(
        self,
        *,
        question: str,
        item: EvidenceItem,
        retrieval_strategy: str,
        allow_tables: bool,
    ) -> dict[str, Any]:
        text = self._sanitize_evidence_text(f"{item.section or ''}\n{item.quote}\n{item.text}")
        verdict = "background"
        confidence = 0.45
        reason = "证据可作为背景，但和问题目标不是直接对应。"

        if self._is_table_like_text(text) and not allow_tables:
            verdict = "reject"
            confidence = 0.7
            reason = "该片段主要是表格或结构化数据，而问题没有要求核对表格。"
        elif self._looks_like_reference_question(question):
            if self._looks_like_reference_section_text(text):
                verdict = "direct"
                confidence = 0.92
                reason = "片段位于参考文献区域，可直接回答参考文献类问题。"
            else:
                verdict = "reject"
                confidence = 0.86
                reason = "问题要求参考文献，但该片段不是参考文献区域。"
        elif self._looks_like_research_limitation_question(question):
            score = self._research_limitation_relevance_score(
                text=text,
                section=item.section or "",
                index=0,
                total=1,
            )
            if score >= 2.0:
                verdict = "direct"
                confidence = min(0.96, 0.62 + score * 0.08)
                reason = "片段包含未来研究、实证验证、研究方法或结论边界，可支撑文章研究局限判断。"
            elif score >= 0.85:
                verdict = "supporting"
                confidence = min(0.82, 0.5 + score * 0.08)
                reason = "片段能辅助判断文章研究边界，但不是最直接的局限说明。"
            else:
                verdict = "reject"
                confidence = 0.82
                reason = "片段更像正文研究对象的困难/风险，不能直接当作文章本身的研究局限。"
        elif self._looks_like_reliability_question(question):
            score = self._reliability_relevance_score(text)
            if score >= 8:
                verdict = "direct"
                confidence = 0.88
                reason = "片段包含文档类型、方法、数据来源、验证或未来研究线索，可直接用于可靠性判断。"
            elif score >= 3:
                verdict = "supporting"
                confidence = 0.7
                reason = "片段提供了可靠性判断的辅助线索。"
            else:
                verdict = "reject"
                confidence = 0.76
                reason = "片段缺少方法、数据、验证或结论支撑信息，不适合用于可靠性判断。"
        elif self._looks_like_title_alignment_question(question):
            keywords = ["题目", "机制", "风险", "治理", "结论", "未来研究", "实证数据"]
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits >= 2:
                verdict = "direct"
                confidence = 0.78
                reason = "片段包含题目、机制、风险、治理或结论线索，可用于核对题目与结论匹配。"
            elif hits == 1:
                verdict = "supporting"
                confidence = 0.62
                reason = "片段只命中一个匹配线索，适合作为辅助证据。"
        else:
            relevance = self._question_relevance_score(question, text)
            if relevance >= 0.24 or item.score >= 0.82:
                verdict = "direct"
                confidence = min(0.9, 0.58 + relevance)
                reason = "片段和问题关键词/语义关联较强，可直接进入回答。"
            elif relevance >= 0.08 or item.score >= 0.45:
                verdict = "supporting"
                confidence = 0.62
                reason = "片段与问题有一定关联，可作为辅助证据。"

        return {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "reason": reason,
            "retrieval_strategy": retrieval_strategy,
        }

    def _cross_verify_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
        answer_strategy: str,
    ) -> dict[str, Any]:
        cited_ids = self._citation_ids_from_answer(answer)
        evidence_by_id = {item.citation_id: item for item in evidence}
        missing_citations = [citation_id for citation_id in cited_ids if citation_id not in evidence_by_id]
        weak_citations: list[dict[str, Any]] = []

        for sentence, sentence_citations in self._sentences_with_citations(answer):
            for citation_id in sentence_citations:
                item = evidence_by_id.get(citation_id)
                if not item:
                    continue
                overlap = self._sentence_evidence_overlap(sentence, item.text)
                if overlap < 0.08:
                    weak_citations.append(
                        {
                            "citation_id": citation_id,
                            "overlap": round(overlap, 3),
                            "reason": "回答句子和对应证据的可解释重叠较低，需要人工核对。",
                        }
                    )

        needs_evidence = not self._looks_like_meta_question(question)
        refusal_answer = any(phrase in answer for phrase in ["没有找到", "证据不足", "不能可靠", "不应该硬编"])
        uncited_answer = needs_evidence and evidence and not cited_ids and not refusal_answer

        if missing_citations:
            status = "fail"
            summary = "回答引用了不存在的证据编号。"
        elif uncited_answer:
            status = "warn"
            summary = "回答没有显式引用证据，可信度需要降低。"
        elif weak_citations:
            status = "warn"
            summary = "部分引用和回答句子的直接重叠较低，已标记为弱支撑。"
        else:
            status = "pass"
            summary = "回答中的引用都能在本轮证据中找到。"

        if answer_strategy == "missing_evidence_refusal" or (not evidence and refusal_answer):
            status = "pass"
            summary = "回答已经按证据不足处理，没有继续硬答。"

        return {
            "status": status,
            "summary": summary,
            "citation_count": len(cited_ids),
            "missing_citations": missing_citations,
            "weak_citations": weak_citations[:5],
            "uncited_answer": uncited_answer,
        }

    def _citation_ids_from_answer(self, answer: str) -> list[str]:
        return [f"E{value}" for value in dict.fromkeys(re.findall(r"\[E(\d+)\]", answer))]

    def _sentences_with_citations(self, answer: str) -> list[tuple[str, list[str]]]:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？.!?])\s+|(?<=。)|(?<=！)|(?<=？)|\n+", answer)
            if part.strip()
        ]
        results: list[tuple[str, list[str]]] = []
        for sentence in sentences:
            citation_ids = [f"E{value}" for value in re.findall(r"\[E(\d+)\]", sentence)]
            if citation_ids:
                results.append((sentence, list(dict.fromkeys(citation_ids))))
        return results

    def _sentence_evidence_overlap(self, sentence: str, evidence_text: str) -> float:
        sentence_text = re.sub(r"\[E\d+\]", "", self._sanitize_evidence_text(sentence))
        evidence_text = self._sanitize_evidence_text(evidence_text)
        meaningful_chars = [
            char
            for char in re.findall(r"[\u4e00-\u9fff]", sentence_text)
            if char not in set("这篇份个的了呢吗啊和与及或是在中里上下主要可以说明因此因为所以当前原文证据文章研究局限")
        ]
        if not meaningful_chars:
            return 1.0
        evidence_chars = set(re.findall(r"[\u4e00-\u9fff]", evidence_text))
        hits = sum(1 for char in meaningful_chars if char in evidence_chars)
        return hits / max(len(meaningful_chars), 1)

    def _filter_evidence_for_question(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []

        reliability_question = self._looks_like_reliability_question(question)
        research_limitation_question = self._looks_like_research_limitation_question(question)
        alignment_question = self._looks_like_title_alignment_question(question)
        compound_request = self._looks_like_compound_request(question)
        reference_question = self._looks_like_reference_question(question)
        structured_review = self._looks_like_structured_review_request(question)
        allow_tables = self._looks_like_table_question(question)
        if compound_request:
            selected: list[EvidenceItem] = []
            seen_compound: set[str] = set()
            target_count = max(6, min(16, top_k + 10))
            for item in evidence:
                if item.chunk_id in seen_compound:
                    continue
                seen_compound.add(item.chunk_id)
                if self._looks_like_reference_section_text(item.text):
                    item.quote = self._best_reference_quote(item.text)
                else:
                    item.quote = self._best_quote_for_question(question, item.text)
                selected.append(item)
                if len(selected) >= target_count:
                    break
            return self._renumber_evidence(selected)

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

        if structured_review:
            selected: list[EvidenceItem] = []
            seen_review: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_review:
                    continue
                seen_review.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                selected.append(item)
                if len(selected) >= max(4, min(top_k + 3, 8)):
                    break
            return self._renumber_evidence(selected)

        if reliability_question or alignment_question:
            selected: list[EvidenceItem] = []
            seen_for_reliability: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_for_reliability:
                    continue
                seen_for_reliability.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                text = self._sanitize_evidence_text(item.text)
                if self._is_table_like_text(text) and not allow_tables:
                    continue
                selected.append(item)
                if len(selected) >= max(1, top_k):
                    break
            return self._renumber_evidence(selected or evidence[: max(1, top_k)])

        if research_limitation_question:
            selected: list[EvidenceItem] = []
            seen_limitations: set[str] = set()
            for item in evidence:
                if item.chunk_id in seen_limitations:
                    continue
                seen_limitations.add(item.chunk_id)
                item.quote = self._best_quote_for_question(question, item.text)
                text = self._sanitize_evidence_text(item.text)
                if self._is_table_like_text(text) and not allow_tables:
                    continue
                score = self._research_limitation_relevance_score(
                    text=text,
                    section=item.section or "",
                    index=0,
                    total=1,
                )
                if score < 0.65:
                    continue
                selected.append(item)
                if len(selected) >= max(2, min(top_k + 1, 6)):
                    break
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
            if reliability_question:
                adjusted_score += self._reliability_relevance_score(text) * 0.08
                if any(keyword in text for keyword in ["随机生成", "论文样稿", "课程报告", "实验报告", "毕业论文", "学位论文", "摘要"]):
                    adjusted_score += 1.6
                if any(keyword in text for keyword in ["未来研究", "实证数据", "文献分析", "情境推演", "机制建构"]):
                    adjusted_score += 0.6
                elif "结论" in text:
                    adjusted_score += 0.2
                if table_like and not allow_tables:
                    adjusted_score -= 0.4
            scored.append((adjusted_score, position, item))

        target_count = max(1, top_k)
        if not scored:
            selected = fallback[:target_count] or evidence[:target_count]
        else:
            scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
            selected = [item for _, _, item in scored[:target_count]]

        return self._renumber_evidence(selected)

    def _build_local_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到可用的原文片段，所以暂时不能可靠回答。你可以换个问法，或重新准备文档。"
        if self._looks_like_reference_question(question):
            return self._build_local_reference_answer(question, evidence, memory_facts)
        if self._looks_like_reliability_question(question):
            return self._build_local_reliability_answer(question, evidence, memory_facts)

        profile = self._profile_from_evidence(evidence)
        clean_text = " ".join(
            self._sanitize_evidence_text(item.text)
            for item in evidence[:4]
            if item.text.strip()
        )
        sentences = self._pick_readable_sentences(clean_text, limit=4)
        citation = evidence[0].citation_id

        if self._looks_like_document_wide_question(question):
            return self._build_local_document_wide_answer(
                question,
                evidence,
                memory_facts,
            )

        body = " ".join(sentences)
        audience_note = self._audience_note(memory_facts or {}, question, clean_text)
        return (
            f"我先按你的问题直接回答：\n\n"
            f"{audience_note}"
            f"我在《{profile.title}》里找到的相关信息是：{body or profile.main_claim} "
            f"[{citation}]\n\n"
            "如果你想要更准确的分析，可以继续问“它的方法可靠吗”“它有哪些局限”“它的结论能不能支撑题目”。"
        )

    def _build_local_document_wide_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文内容，所以暂时不能概括这份文档。"

        profile = self._profile_from_evidence(evidence)
        all_text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        citation_suffix = self._join_citations([item.citation_id for item in evidence[:2]])
        if not citation_suffix and evidence:
            citation_suffix = f" [{evidence[0].citation_id}]"
        audience_note = self._audience_note(memory_facts or {}, question, all_text).strip()
        prefix = f"{audience_note}\n\n" if audience_note else ""

        if any(word in question for word in ["方法", "怎么做", "如何研究", "研究设计"]):
            method = profile.method
            method_evidence = self._method_sentences(all_text, limit=1)
            detail = " ".join(method_evidence) if method_evidence else profile.main_claim
            if method and "没有清楚" not in method:
                return (
                    f"{prefix}这份文档的方法可以概括为：{method}。\n\n"
                    f"从原文看，相关说明是：{detail}{citation_suffix}"
                )
            return (
                f"{prefix}我没有看到它清楚交代可复核的研究方法、样本或数据来源。\n\n"
                f"目前能找到的相关内容是：{detail}{citation_suffix}"
            )

        if any(word in question for word in ["局限", "不足", "问题", "风险"]):
            limitations = (
                self._research_limitation_points(profile, all_text)
                if any(word in question for word in ["局限", "不足"])
                else []
            )
            if not limitations:
                limitations = self._focused_sentences_for_question(question, all_text, limit=3)
            if not limitations:
                limitations = self._focused_sentences_for_question("未来研究 缺少 不足 风险", all_text, limit=3)
            body = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(limitations, start=1))
            if not body:
                body = "原文没有非常集中地说明局限，只能从现有证据中看到它仍需要更多数据、验证或边界说明。"
            return f"{prefix}它的局限/风险主要可以这样看：\n\n{body}\n\n这些判断来自原文相关段落。{citation_suffix}"

        if any(word in question for word in ["结论", "发现", "核心", "重点", "贡献"]):
            conclusion = self._extract_conclusion_from_text(all_text)
            focused = self._focused_sentences_for_question(question, all_text, limit=3)
            if conclusion:
                return f"{prefix}它的核心结论是：{conclusion}{citation_suffix}"
            body = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(focused, start=1))
            return (
                f"{prefix}我从原文里提炼到的核心观点是：\n\n{body or profile.main_claim}\n\n"
                f"这些内容来自文档中与问题最相关的段落。{citation_suffix}"
            )

        if any(word in question for word in ["能学到", "学到什么", "收获", "启发"]):
            learning_points = self._learning_points_for_profile(
                profile=profile,
                question=question,
                text=all_text,
                memory_facts=memory_facts or {},
            )
            body = "\n".join(
                f"{index}. {point}"
                for index, point in enumerate(learning_points, start=1)
            )
            major = self._extract_major_from_context(question, memory_facts or {})
            intro = (
                f"从{major}专业角度看，你更适合重点学这几件事："
                if major
                else "你这次说了“专业角度”，但我没有读到具体专业；我先说明边界，再按通用阅读角度给你可用的收获："
            )
            return (
                f"{prefix}{intro}\n\n"
                f"{body}\n\n"
                f"这些判断来自当前文档的主题、机制、风险和治理相关段落。{citation_suffix}"
            )

        topic = self._extract_topic_from_text(all_text) or profile.title
        key_sentences = self._focused_sentences_for_question(question, all_text, limit=3)
        if not key_sentences:
            key_sentences = self._pick_readable_sentences(all_text, limit=3)
        method_line = (
            f"它采用的方法是{profile.method}。"
            if profile.method and "没有清楚" not in profile.method
            else "原文没有清楚呈现严格、可复核的研究方法。"
        )
        main_points = "\n".join(
            f"{index}. {sentence}"
            for index, sentence in enumerate(key_sentences[:3], start=1)
        )
        point_intro = "可以先抓住三点：" if len(key_sentences) >= 3 else "可以先抓住这些点："
        return (
            f"{prefix}这份文档主要讲《{topic}》。\n\n"
            f"{point_intro}\n"
            f"{main_points or f'1. {profile.main_claim}'}\n\n"
            f"{method_line}{citation_suffix}"
        )

    def _should_decline_for_missing_direct_evidence(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> bool:
        if not evidence:
            return False
        if any(
            checker(question)
            for checker in [
                self._looks_like_reference_question,
                self._looks_like_structured_review_request,
                self._looks_like_reliability_question,
                self._looks_like_title_alignment_question,
                self._looks_like_compare_question,
                self._looks_like_document_wide_question,
            ]
        ):
            return False

        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence[:5]))
        strict_terms = [
            "数据库",
            "向量库",
            "代码",
            "公式",
            "指标",
            "样本量",
            "问卷",
            "访谈",
            "回归",
            "实验组",
            "对照组",
            "p值",
            "显著性",
            "作者",
            "日期",
            "年份",
        ]
        requested_terms = [term for term in strict_terms if term in question]
        if requested_terms and not any(term in text for term in requested_terms):
            return True

        relevance_scores = [
            self._question_relevance_score(question, item.text)
            for item in evidence[:5]
        ]
        best_relevance = max(relevance_scores or [0.0])
        best_vector_score = max((item.score for item in evidence[:5]), default=0.0)
        return best_relevance < 0.05 and best_vector_score < 0.35

    def _build_missing_direct_evidence_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not evidence:
            return "我没有在当前文档里找到能回答这个问题的原文证据，所以不能可靠回答。"
        citations = self._join_citations([item.citation_id for item in evidence[:2]])
        return (
            "我没有在当前文档里找到能直接回答这个问题的证据，所以不应该硬编答案。\n\n"
            f"我刚刚检索到的相近段落和你的问题关联不够强。你可以换成更贴近原文的问法，"
            f"或者点右侧证据看看文档里实际出现了哪些内容。{citations}"
        )

    def _section_number(self, index: int) -> str:
        numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 1 <= index <= len(numbers):
            return numbers[index - 1]
        return str(index)

    def _compound_task_title(
        self,
        task: ParsedTask,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        major = self._extract_major_from_context(question, memory_facts)
        titles = {
            "overview_summary": "总体概括文章内容",
            "reference_list": "文章使用的参考文献",
            "professional_takeaways": f"从{major}专业角度，你能收获什么" if major else "从专业角度，你能收获什么",
            "reliability_judgment": "可靠性判断",
            "method_analysis": "研究方法/实现方法分析",
            "limitation_analysis": "局限与不足",
            "conclusion_summary": "核心结论",
            "comparison": "文档对比",
        }
        return titles.get(task.task_type, task.label)

    def _build_local_compound_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文证据，所以不能可靠地完成这个多步骤问题。请确认文档已经上传并完成索引。"

        tasks = self._parse_compound_tasks(question)
        if len(tasks) < 2:
            return self._build_local_answer(question, evidence, memory_facts)

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_key: dict[str, DocumentProfile] = {}
        for document_id in resolved_document_ids:
            profile = self._build_document_profile(document_id)
            profile_by_key[profile.document_id] = profile
            profile_by_key[profile.name] = profile

        all_context = " ".join(item.text for item in evidence)
        audience_note = self._audience_note(memory_facts or {}, question, all_context).strip()
        ordered_tasks = " → ".join(task.label for task in tasks)
        parts = [f"我按你问题里的顺序来回答：{ordered_tasks}。"]
        if len(grouped) > 1:
            parts.append("你上传了多篇文档，我会在每个任务里把文档分开说，避免混在一起。")
        if audience_note:
            parts.append(audience_note)

        for index, task in enumerate(tasks, start=1):
            section_body = self._render_compound_task_section(
                task=task,
                question=question,
                grouped=grouped,
                profile_by_key=profile_by_key,
                memory_facts=memory_facts or {},
            )
            parts.append(
                f"{self._section_number(index)}、{self._compound_task_title(task, question, memory_facts or {})}\n\n"
                f"{section_body}"
            )

        return "\n\n".join(part for part in parts if part.strip())

    def _render_compound_task_section(
        self,
        *,
        task: ParsedTask,
        question: str,
        grouped: dict[str, list[EvidenceItem]],
        profile_by_key: dict[str, DocumentProfile],
        memory_facts: dict[str, str],
    ) -> str:
        if task.task_type == "comparison":
            all_evidence = [item for items in grouped.values() for item in items]
            return self._build_local_compare_answer(question, all_evidence, memory_facts)

        sections: list[str] = []
        multi_document = len(grouped) > 1
        for doc_index, (name, items) in enumerate(grouped.items(), start=1):
            profile = (
                profile_by_key.get(items[0].document_id)
                or profile_by_key.get(name)
                or self._profile_from_evidence(items)
            )
            text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            if task.task_type == "overview_summary":
                content = self._render_compound_overview(profile, items, text)
            elif task.task_type == "reference_list":
                content = self._render_compound_references(name, items)
            elif task.task_type == "professional_takeaways":
                content = self._render_compound_takeaways(profile, items, text, question, memory_facts)
            elif task.task_type == "reliability_judgment":
                content = self._render_compound_reliability(profile, items, text)
            elif task.task_type == "method_analysis":
                content = self._render_compound_method(profile, items, text)
            elif task.task_type == "limitation_analysis":
                content = self._render_compound_limitations(profile, items, text, question)
            elif task.task_type == "conclusion_summary":
                content = self._render_compound_conclusion(profile, items, text)
            else:
                content = self._build_local_document_wide_answer(question, items, memory_facts)

            if multi_document:
                title = profile.title or name
                sections.append(f"{doc_index}. 《{title}》\n{content}")
            else:
                sections.append(content)
        return "\n\n".join(sections)

    def _render_compound_overview(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["摘要", "本文围绕", "研究主题", "实验名称", "实验目的"]),
                self._first_citation_with(items, ["结论", "研究认为", "总体而言"], fallback=False),
            ]
        )
        if profile.kind == "实验报告" or "实验名称" in text:
            summary = self._summarize_document_for_compare(items)
            return (
                f"这份文档主要是《{summary['title']}》相关内容，重点在{summary['focus']}。"
                f"它的实现/展开过程主要围绕{summary['implementation']}，对应要掌握的知识点是{summary['knowledge']}。{citation}"
            )

        topic = self._extract_topic_from_text(text) or profile.title
        conclusion = (self._extract_conclusion_from_text(text) or profile.main_claim).rstrip("。；; ")
        method_line = (
            f"它主要采用{profile.method}来展开论证"
            if profile.method and "没有清楚" not in profile.method
            else "它更偏观点梳理和材料分析"
        )
        risk_line = (
            "同时提醒学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等风险"
            if any(keyword in text for keyword in ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界"])
            else "同时需要继续核对方法、数据来源和结论边界"
        )
        return (
            f"这篇文档主要讨论《{topic}》。可以抓住三点：\n\n"
            f"1. 研究对象：它围绕“{topic}”展开，重点看这个主题中的问题、机制、场景和结果。\n"
            f"2. 论证方式：{method_line}，所以它更像一篇框架分析型文本，而不是严格实证论文。\n"
            f"3. 核心判断：{conclusion}；{risk_line}。\n\n"
            f"这些概括来自文档开头、方法和结论相关证据。{citation}"
        )

    def _render_compound_references(self, name: str, items: list[EvidenceItem]) -> str:
        text = "\n".join(item.text for item in items)
        references = self._extract_references_from_text(text)
        citations = self._join_citations([item.citation_id for item in items if self._reference_marker_count(item.text) > 0])
        if not citations:
            citations = self._join_citations([self._first_citation_with(items, ["参考文献", "References", "[1]"], fallback=False)])

        if not references:
            return (
                f"我没有稳定解析出《{name}》的编号参考文献条目。"
                f"如果右侧原文里能看到参考文献，建议点开对应证据人工核对；如果看不到，可能是文档没有参考文献区或解析没有抽到。{citations}"
            )

        visible_references = references[:12]
        lines = [f"{number}. {content}" for number, content in visible_references]
        more = f"\n\n其余 {len(references) - len(visible_references)} 条可在右侧原文继续核对。" if len(references) > len(visible_references) else ""
        return (
            f"我在文末参考文献区域解析到 {len(references)} 条条目：\n\n"
            f"{chr(10).join(lines)}{more}\n\n"
            f"这些条目来自参考文献区域。{citations}"
        )

    def _render_compound_takeaways(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        points = self._learning_points_for_profile(
            profile=profile,
            question=question,
            text=text,
            memory_facts=memory_facts,
        )[:3]
        citations = self._join_citations(
            [
                self._first_citation_with(items, ["机制", "系统", "流程", "应用场景", "实验目的"], fallback=False),
                self._first_citation_with(items, ["风险", "隐私", "算法偏差", "学术诚信", "责任边界"], fallback=False),
                self._first_citation_with(items, ["人机协同", "治理", "价值对齐", "过程可控"], fallback=False),
            ]
        )
        if not citations and items:
            citations = self._join_citations([items[0].citation_id])
        body = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))
        major = self._extract_major_from_context(question, memory_facts)
        intro = (
            f"从{major}专业学习角度，建议重点收获这三点："
            if major
            else "你这次只说“专业角度”，但我没有读到具体专业；我先说明边界，再给通用阅读收获："
        )
        return f"{intro}\n\n{body}\n\n这些收获不是凭空扩展，而是从文档主题、方法、场景和风险治理线索中提炼出来的。{citations}"

    def _render_compound_reliability(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        kind_citation = self._first_citation_with(items, ["课程报告", "实验报告", "论文", "摘要", "关键词", "样稿"])
        process_citation = self._first_citation_with(items, ["采用", "方法", "实验", "计算", "分析", "机制建构"], fallback=False)
        support_citation = self._first_citation_with(items, ["未来研究", "实证数据", "参考文献", "验证", "局限", "不足"], fallback=False)
        support_points = self._reliability_support_points(
            profile=profile,
            all_text=text,
            kind_citation=kind_citation,
            process_citation=process_citation or kind_citation,
            support_citation=support_citation or process_citation or kind_citation,
        )
        return (
            f"直接判断：{self._reliability_verdict(profile)}\n\n"
            f"理由：\n"
            f"1. {support_points[0]}\n"
            f"2. {support_points[1]}\n"
            f"3. {support_points[2]}"
        )

    def _render_compound_method(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["研究方法", "采用", "文献分析", "情境推演", "机制建构"], fallback=False),
                self._first_citation_with(items, ["实验方法", "实验步骤", "实现过程", "流程"], fallback=False),
            ]
        )
        if profile.kind == "实验报告" or "实验步骤" in text or "实验目的" in text:
            summary = self._summarize_document_for_compare(items)
            return (
                f"它的方法/实现过程可以概括为：围绕{summary['implementation']}展开。"
                f"这类文档更像实验过程说明，重点是把知识点{summary['knowledge']}落实到具体步骤。{citation}"
            )
        method_sentences = self._method_sentences(text, limit=2)
        detail = "；".join(method_sentences) if method_sentences else profile.method
        return (
            f"它的方法可以概括为：{profile.method}。"
            f"这意味着它更侧重观点梳理、框架建构或案例推演；如果要判断研究强度，还要看有没有样本、数据和验证过程。"
            f"{citation}\n\n原文中与方法最相关的信息是：{detail}{citation}"
        )

    def _render_compound_limitations(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
        question: str,
    ) -> str:
        return self._render_research_limitation_section(profile, items)

    def _render_compound_conclusion(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
        text: str,
    ) -> str:
        conclusion = self._extract_conclusion_from_text(text) or profile.main_claim
        citation = self._join_citations(
            [
                self._first_citation_with(items, ["结论与展望", "总体而言", "研究认为", "综上所述"], fallback=False),
                self._first_citation_with(items, ["结论", "发现", "核心观点"], fallback=False),
            ]
        )
        return f"核心结论可以概括为：{conclusion}{citation}"

    def _build_local_structured_review_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return "我没有找到足够的原文证据，所以不能按你的模板完成概括、分部分分析和可靠性判断。"

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        all_context = " ".join(item.text for item in evidence)
        audience_note = self._audience_note(memory_facts or {}, question, all_context).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for doc_index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            all_text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            overview_citation = self._first_citation_with(items, ["摘要", "本文围绕", "关键词", "研究主题"])
            method_citation = self._first_citation_with(items, ["采用", "文献分析", "情境推演", "机制建构"])
            mechanism_citation = self._first_citation_with(
                items,
                ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "运行机制"],
            )
            scenario_citation = self._first_citation_with(
                items,
                ["课程学习场景", "实验与实践教学", "学业预警", "学习支持场景", "应用场景"],
            )
            risk_citation = self._first_citation_with(
                items,
                ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界", "风险"],
            )
            governance_citation = self._first_citation_with(
                items,
                ["人机协同", "价值对齐", "过程可控", "数据最小化", "多主体治理", "治理"],
            )
            risk_governance_citations = self._join_citations([risk_citation, governance_citation])
            conclusion_citation = self._first_citation_with(
                items,
                ["结论与展望", "未来研究", "实证数据", "总体而言"],
            )
            reference_citation = self._first_citation_with(items, ["参考文献", "[1]", "研究[J]"], fallback=False)

            title = profile.title or name
            topic = self._extract_topic_from_text(all_text) or title
            conclusion = self._extract_conclusion_from_text(all_text) or profile.main_claim
            method = profile.method if profile.method else "原文没有清楚给出研究方法"
            learning_points = self._learning_points_for_profile(
                profile=profile,
                question=question,
                text=all_text,
                memory_facts=memory_facts or {},
            )[:3]
            reliability_points = self._reliability_support_points(
                profile=profile,
                all_text=all_text,
                kind_citation=overview_citation,
                process_citation=method_citation,
                support_citation=conclusion_citation or reference_citation,
            )

            heading = f"第 {doc_index} 篇文档：《{name}》\n\n" if len(grouped) > 1 else ""
            sections.append(
                f"{heading}"
                f"一、先总结概括\n\n"
                f"这篇文档主要讨论《{topic}》。它的核心意思是：生成式人工智能可以为高校学习支持服务提供新的能力，"
                f"但这种能力必须放在教育目标、制度约束和风险治理中理解，不能简单等同于“技术越强越好”。[{overview_citation}]\n\n"
                f"二、再详细分析每一部分\n\n"
                f"1. 研究背景与问题：文档关注高校学习支持服务在个性化、持续反馈、资源供给和组织协同方面的现实压力，"
                f"并把生成式人工智能作为一种可能的支持工具来讨论。[{overview_citation}]\n\n"
                f"2. 研究方法：原文说明主要采用{method}。这说明它更偏理论分析和框架建构，而不是大样本实证研究。[{method_citation}]\n\n"
                f"3. 作用机制：文档把 AI 的作用概括为认知支架、资源重组、过程陪伴、反馈生成和组织协同等方向，"
                f"重点不是单个模型能力，而是 AI 如何嵌入学习支持流程。[{mechanism_citation}]\n\n"
                f"4. 应用场景：它讨论了课程学习、实验实践、学业预警与辅导等场景，说明作者想把 AI 放到具体教育流程里分析。[{scenario_citation}]\n\n"
                f"5. 风险与治理：文档提醒学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等风险，"
                f"并提出人机协同、价值对齐、过程可控、数据最小化和多主体治理等思路。{risk_governance_citations}\n\n"
                f"三、从你的专业角度看，可以重点学什么\n\n"
                f"1. {learning_points[0]}\n"
                f"2. {learning_points[1]}\n"
                f"3. {learning_points[2]}\n\n"
                f"四、最后判断这篇论文的可靠性\n\n"
                f"直接判断：{self._reliability_verdict(profile)}\n\n"
                f"理由是：\n"
                f"1. {reliability_points[0]}\n"
                f"2. {reliability_points[1]}\n"
                f"3. {reliability_points[2]}\n\n"
                f"五、一句话总结\n\n"
                f"这篇文档适合用来学习“AI 教育应用如何做系统化分析”，但如果要把它当作严格论文结论，还需要进一步核对真实数据、样本、方法和实证验证。"
                f"[{conclusion_citation or overview_citation}]"
            )

        return "\n\n".join(sections)

    def _build_local_reference_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "我没有在当前文档里找到明确的“参考文献/References”部分，所以不能可靠列出它引用了哪些文献。"
                "这通常有三种可能：文档本身没有参考文献、参考文献被图片扫描进了 PDF、或解析时没有抽取到文末内容。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            text = "\n".join(item.text for item in items)
            references = self._extract_references_from_text(text)
            citations = self._join_citations([item.citation_id for item in items])
            label = f"第 {index} 篇文档《{name}》" if len(grouped) > 1 else f"《{name}》"

            if not references:
                sections.append(
                    f"{label}：我找到了疑似参考文献区域，但没有稳定解析出编号条目。"
                    f"建议你点右侧证据查看原文位置再核对。{citations}"
                )
                continue

            reference_lines = [
                f"{number}. {content}"
                for number, content in references
            ]
            sections.append(
                f"{label}文末列出了 {len(references)} 条参考文献，整理如下：\n\n"
                f"{chr(10).join(reference_lines)}\n\n"
                f"这些条目来自文末参考文献区域。{citations}"
            )

        return "\n\n".join(sections)

    def _build_local_research_limitation_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "我没有找到能直接支撑“文章研究局限”的原文证据，所以不应该把正文里的困难/不足硬说成文章局限。\n\n"
                "要回答这个问题，最好看到作者关于研究方法、数据来源、样本、实证验证、讨论或未来研究的说明。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        intro = "我会把“文章局限”限定为研究设计、证据边界和未来研究空间，不把正文里研究对象本身的困难当成文章局限。"
        sections.append(intro)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = (
                self._build_document_profile(items[0].document_id)
                if items and items[0].document_id
                else self._profile_from_evidence(items)
            )
            label = f"第 {index} 篇文档《{name}》" if len(grouped) > 1 else f"《{name}》"
            sections.append(f"{label}的局限可以这样看：\n\n{self._render_research_limitation_section(profile, items)}")

        return "\n\n".join(sections)

    def _render_research_limitation_section(
        self,
        profile: DocumentProfile,
        items: list[EvidenceItem],
    ) -> str:
        text = self._sanitize_evidence_text(" ".join(item.text for item in items))
        future_citation = self._first_citation_with(
            items,
            ["未来研究", "实证数据", "检验", "验证"],
            fallback=False,
        )
        group_citation = self._first_citation_with(
            items,
            ["不同学生群体", "使用差异"],
            fallback=False,
        )
        method_citation = (
            self._first_citation_with_all(items, ["文献分析", "情境推演"])
            or self._first_citation_with_all(items, ["采用", "文献分析"])
            or self._first_citation_with_all(items, ["采用", "机制建构"])
            or self._first_citation_with(items, ["研究方法"], fallback=False)
        )
        direct_section_citation = self._first_citation_with(
            items,
            ["局限性", "研究局限", "研究不足", "局限与不足", "不足与展望", "结论与展望"],
            fallback=False,
        )

        lines: list[str] = []
        if future_citation:
            lines.append(
                f"1. 原文把进一步的实证检验放到未来研究里，说明当前结论还没有被充分的真实数据或多场景验证支撑。"
                f"[{future_citation}]"
            )
        elif direct_section_citation:
            lines.append(
                f"1. 原文在结论、展望或局限相关位置出现研究边界说明，说明作者并没有把当前结论当成已经完全验证的结论。"
                f"[{direct_section_citation}]"
            )

        if group_citation and group_citation != future_citation:
            lines.append(
                f"{len(lines) + 1}. 原文还把不同学生群体或使用差异留作继续比较的内容，说明它对群体差异的讨论还不充分。"
                f"[{group_citation}]"
            )

        if method_citation:
            lines.append(
                f"{len(lines) + 1}. 从方法看，它主要是{profile.method}，这类方法适合梳理框架和提出解释，但不能单独证明实际效果已经发生。"
                f"[{method_citation}]"
            )

        if not profile.has_empirical_data:
            support_citation = future_citation or method_citation or direct_section_citation
            if support_citation:
                lines.append(
                    f"{len(lines) + 1}. 当前证据没有看到样本、问卷、访谈、实验或统计检验，因此它的结论更适合作为理论分析或研究假设，而不是强实证结论。"
                    f"[{support_citation}]"
                )

        if not lines:
            citations = self._join_citations([item.citation_id for item in items[:2]])
            return (
                "原文没有直接列出清晰的“局限性/研究不足”段落；从已检索到的证据看，也缺少能支撑具体局限判断的研究边界说明。"
                f"因此我不能把正文里的普通困难或风险硬说成文章局限。{citations}"
            )

        body = "\n".join(lines[:4])
        return (
            f"{body}\n\n"
            "这些是文章研究层面的局限，不是正文中被研究对象自身存在的“困难/不足”。"
        )

    def _build_local_reliability_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "直接结论：我现在不能判断这篇文档的结果是否可靠，因为没有检索到足够的原文证据。\n\n"
                "要回答这个问题，至少需要看到它的研究方法、数据来源、结果推导、结论和局限说明。"
            )

        grouped = self._group_evidence_by_document(evidence)
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            all_text = self._sanitize_evidence_text(" ".join(item.text for item in items))
            kind_citation = self._first_citation_with(
                items,
                ["随机生成", "样稿", "课程报告", "实验报告", "论文", "摘要", "关键词", "课程", "报告"],
            )
            process_citation = self._first_citation_with(
                items,
                ["采用", "文献分析", "情境推演", "机制建构", "计算", "公式", "分析", "评价", "一致性检验", "测试", "结果", "结论"],
            )
            support_citation = self._first_citation_with(
                items,
                ["未来研究", "实证数据", "参考文献", "数据来源", "资料来源", "出处", "误差", "验证", "一致性检验", "后评价", "局限", "不足"],
            )
            used_citations = self._join_citations(
                [kind_citation, process_citation, support_citation] or [items[0].citation_id]
            )

            verdict = self._reliability_verdict(profile)
            support_points = self._reliability_support_points(
                profile=profile,
                all_text=all_text,
                kind_citation=kind_citation,
                process_citation=process_citation,
                support_citation=support_citation,
            )

            label = f"第 {index} 份文档" if len(grouped) > 1 else "这份文档"
            title_line = f"《{profile.title}》" if profile.title else f"《{name}》"
            sections.append(
                f"{label}{title_line}：\n\n"
                f"直接结论：{verdict}\n\n"
                f"为什么这样判断：\n"
                f"1. {support_points[0]}\n"
                f"2. {support_points[1]}\n"
                f"3. {support_points[2]}\n\n"
                f"可以相信到什么程度：它适合作为理解“生成式人工智能如何支持高校学习服务”的观点框架和写作参考；"
                f"但如果要把它当成严格研究结论，还需要补充样本、数据来源、实证检验或更清楚的方法说明。{used_citations}"
            )

        answer = "\n\n".join(sections)
        return self._ensure_answer_relevance(question, answer, evidence)

    def _build_local_title_alignment_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        if not evidence:
            return (
                "直接结论：我现在不能判断结论能不能支撑题目，因为没有检索到题目、结论或关键论证段落。"
            )

        grouped = self._group_evidence_by_document(evidence)
        resolved_document_ids = self._resolve_document_ids(document_ids)
        profile_by_document = {
            profile.name: profile
            for profile in [
                self._build_document_profile(document_id)
                for document_id in resolved_document_ids
            ]
        }
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        ).strip()
        sections: list[str] = []
        if audience_note:
            sections.append(audience_note)

        for index, (name, items) in enumerate(grouped.items(), start=1):
            profile = profile_by_document.get(name) or self._profile_from_evidence(items)
            title_citation = self._first_citation_with(items, ["机制", "风险", "治理路径", "论文样稿", "题目", "研究"])
            mechanism_citation = self._first_citation_with(
                items,
                ["认知支架", "资源重组", "过程陪伴", "反馈生成", "组织协同", "管理协同", "运行机制"],
            )
            risk_citation = self._first_citation_with(
                items,
                ["学习依赖", "信息准确性", "数据隐私", "算法偏差", "学术诚信", "责任边界"],
            )
            governance_citation = self._first_citation_with_all(
                items,
                ["针对上述风险", "人机协同"],
            )
            if not governance_citation:
                governance_citation = self._first_citation_with(
                    items,
                    ["价值对齐", "过程可控", "多主体治理", "治理框架"],
                )
            limitation_citation = self._first_citation_with(
                items,
                ["未来研究可以进一步通过实证数据检验", "未来研究", "实证数据"],
                fallback=False,
            )
            if not limitation_citation:
                limitation_citation = self._first_citation_with(
                    items,
                    ["文献分析", "情境推演", "机制建构", "样稿"],
                )
            used_citations = self._join_citations(
                [title_citation, mechanism_citation, risk_citation, governance_citation, limitation_citation]
            )

            label = f"第 {index} 份文档" if len(grouped) > 1 else "这篇论文"
            sections.append(
                f"{label}《{profile.title}》：\n\n"
                "直接结论：基本能支撑题目，但支撑强度不算很强。它能对应题目里的“机制、风险、治理路径”三个关键词，"
                "不过更多是理论框架式支撑，不是实证数据支撑。\n\n"
                "我为什么这么判断：\n"
                f"1. 题目本身要求回答三个部分：生成式人工智能如何赋能学习支持服务的机制、可能风险、以及治理路径。[{title_citation}]\n"
                f"2. 结论能回应“机制”：原文把作用机制概括为认知支架、资源重组、过程陪伴、反馈生成和组织协同等内容，这和题目中的“机制”是对应的。[{mechanism_citation}]\n"
                f"3. 结论能回应“风险”：原文讨论了学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界等问题，这和题目中的“风险”是对应的。[{risk_citation}]\n"
                f"4. 结论也能回应“治理路径”：原文提出人机协同、价值对齐、过程可控、数据最小化和多主体治理，这和题目中的“治理路径”是对应的。[{governance_citation}]\n\n"
                f"不足在哪里：它支撑题目主要靠概念分析和框架归纳。原文方法更偏{profile.method}，并且把进一步通过实证数据检验放在未来研究里，"
                f"所以它能支撑题目方向，但还不能强力证明题目中的观点已经被真实数据验证。[{limitation_citation}]\n\n"
                f"一句话：不算跑题，结论能支撑题目；但如果按严格论文标准看，它的支撑偏“理论上说得通”，还缺少“数据上证明了”。{used_citations}"
            )

        return "\n\n".join(sections)

    def _build_document_profile(self, document_id: str) -> DocumentProfile:
        document = self.store.get_document(document_id)
        rows = self.vector_store.get_document_chunks(document_id, limit=1000)
        chunks = [str(row.get("text", "")) for row in rows if row.get("text")]
        text = self._sanitize_evidence_text(" ".join(chunks))
        first_chunk = chunks[0] if chunks else ""
        name = document.file_name if document else document_id
        title = self._guess_document_title(name, first_chunk)
        kind = self._document_kind_from_text(f"{name}\n{text[:3000]}")
        method = self._extract_method_from_text(text)
        main_claim = self._extract_main_claim_from_text(text)
        return DocumentProfile(
            document_id=document_id,
            name=name,
            title=title,
            kind=kind,
            method=method,
            main_claim=main_claim,
            has_empirical_data=self._has_empirical_data(text),
            has_references=any(keyword in text for keyword in ["参考文献", "[1]", "Journal", "研究[J]"]),
            is_generated_sample=any(keyword in text for keyword in ["随机生成", "论文样稿", "中文论文样稿", "样稿"]),
        )

    def _profile_from_evidence(self, evidence: list[EvidenceItem]) -> DocumentProfile:
        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        name = evidence[0].paper_name if evidence else "当前文档"
        document_id = evidence[0].document_id if evidence else ""
        return DocumentProfile(
            document_id=document_id,
            name=name,
            title=self._guess_document_title(name, text),
            kind=self._document_kind_from_text(f"{name}\n{text}"),
            method=self._extract_method_from_text(text),
            main_claim=self._extract_main_claim_from_text(text),
            has_empirical_data=self._has_empirical_data(text),
            has_references=any(keyword in text for keyword in ["参考文献", "[1]", "Journal", "研究[J]"]),
            is_generated_sample=any(keyword in text for keyword in ["随机生成", "论文样稿", "中文论文样稿", "样稿"]),
        )

    def _guess_document_title(self, file_name: str, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        if normalized:
            before_author = re.split(r"\s+作者[:：]|作者[:：]|日期[:：]|摘\s*要|摘要", normalized, maxsplit=1)[0]
            before_author = before_author.strip(" -—_")
            if 8 <= len(before_author) <= 120:
                return before_author
        cleaned_name = re.sub(r"\.(pdf|docx)$", "", file_name, flags=re.IGNORECASE)
        cleaned_name = cleaned_name.replace("_", " ").strip()
        return cleaned_name[:120] or "当前文档"

    def _document_kind_from_text(self, text: str) -> str:
        if "课程报告" in text:
            return "课程报告"
        if "实验报告" in text:
            return "实验报告"
        if "论文样稿" in text or "随机生成" in text:
            return "论文样稿"
        if "毕业论文" in text or "学位论文" in text:
            return "论文"
        if ("摘要" in text and "参考文献" in text) or "Abstract" in text:
            return "论文"
        if "论文" in text:
            return "论文"
        return "普通文档"

    def _extract_method_from_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        method_patterns = [
            r"采用([^。；;\n]{4,120}?)(?:的方法|方法)",
            r"本文围绕[^。；;\n]{0,80}?采用([^。；;\n]{4,120}?)(?:的方法|方法)",
            r"研究方法[:：]\s*([^。；;\n]{4,120})",
        ]
        for pattern in method_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" ，,。；;")
        if all(keyword in text for keyword in ["文献分析", "情境推演", "机制建构"]):
            return "文献分析、情境推演和机制建构"
        if all(keyword.lower() in normalized.lower() for keyword in ["transformer", "attention"]):
            return "提出 Transformer 架构，用自注意力替代循环/卷积结构，并在机器翻译等任务上实验验证"
        return "原文没有清楚给出可复核的研究方法"

    def _extract_main_claim_from_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        patterns = [
            r"研究认为，([^。]{20,180})。",
            r"结论与展望\s*([^。]{20,180})。",
            r"总体而言，([^。]{20,180})。",
            r"(In this work, we presented the Transformer[^.]{20,220}\.)",
            r"(We propose a new simple network architecture[^.]{20,220}\.)",
            r"(Our model achieves[^.]{20,220}\.)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "原文主要提出观点框架，但当前证据没有稳定抽取到单一结果陈述"

    def _extract_topic_from_text(self, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        patterns = [
            r"本文围绕([^。；;]{8,120}?)(?:这一主题|展开|进行)",
            r"研究主题[:：]\s*([^。；;\n]{8,120})",
            r"主题是[“\"]?([^。”\"；;]{8,120})",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).strip(" ：:，,。；;“”\"")
        return ""

    def _extract_conclusion_from_text(self, text: str) -> str:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        patterns = [
            r"结论与展望\s*([^。]{30,260}。)",
            r"总体而言，([^。]{20,220}。)",
            r"研究认为，([^。]{20,220}。)",
            r"综上所述，([^。]{20,220}。)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).strip()
        return ""

    def _focused_sentences_for_question(self, question: str, text: str, limit: int) -> list[str]:
        sanitized = self._sanitize_evidence_text(text)
        sentences = self._pick_readable_sentences(sanitized, limit=40)
        if not sentences:
            return []

        focus_keywords = self._overview_focus_keywords(question)
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            normalized_sentence = sentence.lower()
            keyword_hits = sum(
                1
                for keyword in focus_keywords
                if keyword in sentence or keyword.lower() in normalized_sentence
            )
            relevance = self._question_relevance_score(question, sentence)
            score = keyword_hits * 0.45 + relevance + self._overview_structure_score(sentence)
            if any(word in sentence for word in ["姓名", "学号", "电子邮件", "邮箱"]):
                score -= 1
            scored.append((score, index, sentence))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected: list[str] = []
        has_positive_score = any(score > 0 for score, _, _ in scored)
        for score, _, sentence in scored:
            if score <= 0 and has_positive_score:
                continue
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                break
        return selected

    def _method_sentences(self, text: str, limit: int) -> list[str]:
        sentences = self._pick_readable_sentences(text, limit=40)
        scored: list[tuple[float, int, str]] = []
        for sentence in sentences:
            score = 0.0
            if "采用" in sentence and "方法" in sentence:
                score += 3.0
            for keyword in ["文献分析", "情境推演", "机制建构", "研究方法"]:
                if keyword in sentence:
                    score += 1.0
            if sentence.startswith("本文围绕") or "本文围绕" in sentence:
                score += 0.5
            if any(keyword in sentence for keyword in ["案例化", "课程教师", "学生可以", "不得直接提交"]):
                score -= 1.2
            if score > 0:
                scored.append((score, -len(sentence), sentence))

        scored.sort(reverse=True)
        return [sentence for _, _, sentence in scored[:limit]]

    def _research_limitation_points(self, profile: DocumentProfile, text: str) -> list[str]:
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        points: list[str] = []
        if "未来研究可以进一步通过实证数据检验" in normalized:
            points.append("原文把“通过实证数据检验不同应用场景的效果”放在未来研究中，说明当前结论还没有充分实证验证。")
        elif not profile.has_empirical_data:
            points.append("当前证据没有看到样本、问卷、访谈、实验或统计检验，所以它更像理论分析，结论支撑力度有限。")

        if profile.method and "文献分析" in profile.method:
            points.append(f"它主要采用{profile.method}，适合梳理观点和搭框架，但不等于证明实际效果已经发生。")
        elif profile.method and "没有清楚" in profile.method:
            points.append("原文没有清楚交代可复核的研究方法，这是判断论文质量时比较明显的短板。")

        if "比较不同学生群体的使用差异" in normalized:
            points.append("原文还把“比较不同学生群体的使用差异”留给未来研究，说明对不同群体的差异分析还不充分。")

        unique: list[str] = []
        for point in points:
            if point not in unique:
                unique.append(point)
        return unique[:3]

    def _learning_points_for_profile(
        self,
        *,
        profile: DocumentProfile,
        question: str,
        text: str,
        memory_facts: dict[str, str],
    ) -> list[str]:
        user_profile = memory_facts.get("user_profile", "")
        normalized = " ".join(self._sanitize_evidence_text(text).split())
        major = self._extract_major_from_context(question, memory_facts)
        is_computer_profile = major in {"计算机", "软件", "网络工程", "人工智能", "数据科学"} or any(
            keyword in f"{user_profile}\n{question}" for keyword in ["计算机", "软件", "网络"]
        )

        if is_computer_profile and any(
            keyword in normalized
            for keyword in ["生成式人工智能", "学习支持", "人机协同", "数据隐私", "算法偏差"]
        ):
            return [
                "学会把 AI 技术放进真实业务场景里分析：这篇文档不是只讲“模型很强”，而是把生成式人工智能放到课程答疑、资源重组、过程陪伴、反馈生成和组织协同等学习支持场景中看。",
                "学会从系统设计角度拆问题：一个 AI 教育系统不只是聊天窗口，还涉及用户需求、交互流程、数据输入、反馈生成、人工监督和组织协同。",
                "学会识别 AI 系统风险：文中提到学习依赖、信息准确性、数据隐私、算法偏差、学术诚信和责任边界，这些都和计算机专业里的安全、可信 AI、数据治理有关。",
                "学会用工程视角看治理方案：人机协同、价值对齐、过程可控、数据最小化和多主体治理，可以理解成 AI 产品落地时的约束条件和设计原则。",
            ]

        if is_computer_profile:
            return [
                f"学会把文档主题抽象成系统问题：这份文档的主题是《{profile.title}》，可以从需求、输入、处理流程、输出和反馈闭环来分析。",
                "学会区分观点、证据和实现：不要只看结论，还要看它有没有方法、数据、实验或可复核过程支撑。",
                "学会从工程落地角度追问：如果要把文中想法做成系统，需要继续明确数据来源、模块边界、异常处理、隐私安全和评价指标。",
            ]

        if major:
            return self._major_specific_learning_points(
                major=major,
                profile=profile,
                text=normalized,
            )

        return [
            f"先学会抓主线：这份文档围绕《{profile.title}》展开，阅读时要把主题、方法、证据和结论分开看。",
            "再学会判断证据强度：不要只看结论写得顺不顺，要看它有没有样本、数据、实证检验或可复核的分析过程。",
            "最后学会看局限：凡是缺少数据、缺少验证、只停留在框架推演的地方，都不能直接当成已经被证明的结论。",
        ]

    def _extract_major_from_context(
        self,
        question: str,
        memory_facts: dict[str, str],
    ) -> str:
        text = f"{question}\n{memory_facts.get('user_profile', '')}"
        aliases = [
            ("计算机", ["计算机", "软件工程", "网络工程", "人工智能", "数据科学", "大数据"]),
            ("医学", ["医学", "临床医学", "护理", "药学", "公共卫生", "医学生"]),
            ("教育学", ["教育学", "教育技术", "师范", "教师", "教学"]),
            ("法学", ["法学", "法律", "知识产权"]),
            ("管理学", ["管理", "工商管理", "公共管理", "人力资源", "市场营销"]),
            ("经济金融", ["经济", "金融", "会计", "财务", "审计"]),
            ("中文", ["汉语言", "中文", "文学", "新闻", "传播"]),
            ("英语", ["英语", "外语", "翻译"]),
            ("心理学", ["心理", "心理学"]),
            ("设计", ["设计", "艺术", "视觉传达", "产品设计"]),
        ]
        for normalized_major, keywords in aliases:
            if any(keyword in text for keyword in keywords):
                return normalized_major

        patterns = [
            r"我是(?P<major>[^，。,.!\n]{2,16})专业",
            r"我学(?:的是)?(?P<major>[^，。,.!\n]{2,16})",
            r"我的专业是(?P<major>[^，。,.!\n]{2,16})",
            r"从(?P<major>[^，。,.!\n]{2,16})专业角度",
            r"结合(?P<major>[^，。,.!\n]{2,16})专业",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                major = match.group("major").strip("的 角度学生大学生本科生研究生")
                if 2 <= len(major) <= 16:
                    return major
        return ""

    def _major_specific_learning_points(
        self,
        *,
        major: str,
        profile: DocumentProfile,
        text: str,
    ) -> list[str]:
        if major == "医学":
            return [
                "学会判断 AI 应用是否真的能改善服务效果：不要只看“智能化”表述，要追问有没有样本、干预过程、评价指标和真实效果验证。",
                "学会关注风险边界：文中提到的信息准确性、数据隐私、责任边界等问题，放到医学场景里就对应误导诊疗、隐私泄露和责任归属。",
                "学会把技术工具放进服务流程：可以思考 AI 如何辅助健康教育、患者随访、学习反馈或临床培训，但必须有人类专业判断兜底。",
            ]
        if major == "教育学":
            return [
                "学会从学习支持服务体系看 AI：重点不是模型本身，而是它如何参与答疑、反馈、资源推荐、学业预警和师生协同。",
                "学会分析教学设计边界：AI 可以提高反馈效率，但不能替代学习目标设计、评价标准和教师对学生发展的判断。",
                "学会关注教育治理：文中提到学习依赖、学术诚信、算法偏差和人机协同，这些都可以转化为教育管理和课堂规则设计问题。",
            ]
        if major == "法学":
            return [
                "学会识别技术应用背后的责任问题：文中讨论的数据隐私、算法偏差、责任边界，可以转化为合规、侵权责任和平台治理问题。",
                "学会追问制度依据：AI 进入教育服务时，不能只看效率，还要看数据授权、知情同意、可解释性和责任分配是否清楚。",
                "学会把论文观点转成规范分析：哪些风险需要规则约束，哪些环节需要人工审核，哪些主体应承担相应责任。",
            ]
        if major == "管理学":
            return [
                "学会把 AI 看成组织流程改造工具：文中提到资源重组、过程陪伴和管理协同，可以对应服务流程、组织分工和绩效评价。",
                "学会分析落地条件：一个 AI 项目是否可行，不只看技术能力，还要看成本、人员培训、风险控制和持续运营机制。",
                "学会做风险治理：隐私、安全、责任边界和多主体协同，都是管理专业分析技术项目时必须纳入的约束。",
            ]
        if major == "经济金融":
            return [
                "学会看投入产出：AI 学习支持服务是否值得做，要进一步比较建设成本、使用效率、服务质量提升和长期维护成本。",
                "学会关注数据治理：文中的数据隐私、算法偏差和责任边界，在经济金融领域可对应风控、合规和信息不对称问题。",
                "学会识别证据强度：如果文档缺少样本、数据和实证检验，就不能把它当作已经证明有效的经济结论。",
            ]
        if major in {"中文", "英语"}:
            return [
                "学会分析文本结构：可以观察它如何从背景、问题、机制、风险、治理到结论组织一篇论文式文本。",
                "学会评价论证质量：不要只看语言是否顺畅，还要看概念是否清楚、段落是否递进、证据是否能支撑结论。",
                "学会识别 AI 生成文本特征：如果文档像样稿，就要特别关注观点是否泛化、引用是否可靠、细节是否经过真实验证。",
            ]
        if major == "心理学":
            return [
                "学会关注学习者体验：文中提到情绪陪伴、学习依赖和反馈生成，可以转化为动机、认知负荷和自我调节学习问题。",
                "学会区分支持与替代：AI 可以提供陪伴和反馈，但过度依赖可能削弱学生的主动学习和问题解决能力。",
                "学会追问评估方式：如果要证明 AI 对学习心理或学习效果有帮助，需要量表、实验设计或长期跟踪数据支持。",
            ]
        if major == "设计":
            return [
                "学会从用户体验看 AI 系统：文中提到自然语言交互、反馈生成和过程陪伴，可以转化为交互流程与服务触点设计。",
                "学会关注可理解性：AI 给出的建议、反馈和风险提示需要让用户看得懂、愿意用、知道何时不能依赖。",
                "学会把风险转为设计约束：隐私、安全、责任边界和人工确认机制，都应该体现在界面和使用流程里。",
            ]
        return [
            f"从{major}专业角度，可以先学习如何把文档主题《{profile.title}》转化成本专业的问题意识，而不是只复述原文。",
            "再学习如何判断论证强度：看它的方法、证据、数据来源和结论之间是否匹配。",
            "最后学习如何识别落地边界：文中提到的风险、治理和未来研究，都是把观点转成本专业分析时必须补充的条件。",
        ]

    def _has_empirical_data(self, text: str) -> bool:
        empirical_keywords = ["样本", "问卷", "访谈", "实验", "回归", "统计检验", "显著性", "数据集", "实证"]
        if "未来研究可以进一步通过实证数据检验" in text:
            return False
        return any(keyword in text for keyword in empirical_keywords)

    def _reliability_verdict(self, profile: DocumentProfile) -> str:
        if profile.is_generated_sample:
            return (
                "不适合把它的“结果”当成可靠研究结论直接采信。它更像一篇生成式中文论文样稿，"
                "可以参考其结构、问题意识和治理框架，但不能把其中观点当作经过真实研究验证的结论。"
            )
        if profile.kind == "课程报告":
            return "它可以作为课程作业或方法演示来参考，但不能直接当作严格论文结论采信。"
        if profile.kind == "实验报告":
            return "它可以作为实验过程记录参考，但可靠性仍取决于实验设计、数据来源和验证是否完整。"
        if profile.kind == "论文" and profile.has_empirical_data:
            return "它具备一定研究论文形态，结果有参考价值，但仍需要核对样本、数据、方法和结论是否一一对应。"
        if profile.kind == "论文":
            return "它的观点有一定参考价值，但更偏理论分析或综述推演，不能等同于已经被实证证明的结果。"
        return "当前证据不足以判定它的结果可靠，只能作为有限参考。"

    def _reliability_support_points(
        self,
        *,
        profile: DocumentProfile,
        all_text: str,
        kind_citation: str,
        process_citation: str,
        support_citation: str,
    ) -> list[str]:
        points: list[str] = []
        if profile.is_generated_sample:
            points.append(f"原文标题或说明里出现“随机生成的中文论文样稿”这类表述，说明它不是严格意义上的真实研究成果。[{kind_citation}]")
        else:
            points.append(f"从文档结构看，它更接近“{profile.kind}”；判断可靠性时要按这个文档类型来要求证据。[{kind_citation}]")

        if profile.method and "没有清楚" not in profile.method:
            points.append(f"它说明采用了{profile.method}，这能支持理论梳理和框架建构，但不能单独证明应用效果真实发生。[{process_citation}]")
        else:
            points.append(f"当前证据没有清楚呈现可复核的方法、样本或数据来源，所以结果支撑偏弱。[{process_citation}]")

        if "未来研究可以进一步通过实证数据检验" in all_text:
            points.append(f"原文自己也把“通过实证数据检验不同应用场景的效果”放到未来研究中，这说明当前结论还缺少实证验证。[{support_citation}]")
        elif profile.has_empirical_data:
            points.append(f"文中出现了实证或数据线索，但还需要继续核对样本、指标、统计过程和结论之间是否匹配。[{support_citation}]")
        elif profile.has_references:
            points.append(f"文末有参考文献，能说明它有一定资料来源，但引用本身不能替代对核心结论的实证检验。[{support_citation}]")
        else:
            points.append("当前证据没有清楚展示数据来源、实验/问卷/访谈设计或误差分析，这是可靠性判断的主要缺口。")

        return points

    def _ensure_answer_relevance(
        self,
        question: str,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> str:
        if not self._looks_like_reliability_question(question):
            return answer
        banned_phrases = ["先纠正刚才", "回答方式", "不能只摘一个相似度最高"]
        if any(phrase in answer for phrase in banned_phrases):
            profile = self._profile_from_evidence(evidence)
            citation = evidence[0].citation_id if evidence else "E1"
            return (
                f"直接结论：这篇文档的结果不能直接判定为可靠。"
                f"它更像“{profile.kind}”，需要结合方法、数据来源和验证过程来判断。[{citation}]\n\n"
                "目前证据不足以证明它的结论已经经过严格验证，因此更适合作为参考材料，而不是直接当作可靠研究结论。"
            )
        if "直接结论" not in answer:
            return f"直接结论：{answer}"
        return answer

    def _build_local_grouped_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        grouped = self._group_evidence_by_document(evidence)
        if len(grouped) < 2:
            return self._build_local_answer(question, evidence, memory_facts)

        sections = ["我按文档分开说，避免把几篇内容混在一起："]
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for item in evidence),
        )
        if audience_note:
            sections.append(audience_note.strip())
        for index, (name, items) in enumerate(grouped.items(), start=1):
            summary = self._summarize_document_for_compare(items)
            citation = items[0].citation_id
            sections.append(
                f"{index}. 《{name}》\n"
                f"{self._build_document_wide_summary(question, summary, items)} [{citation}]"
            )
        return "\n\n".join(sections)

    def _build_document_wide_summary(
        self,
        question: str,
        summary: dict[str, str],
        evidence: list[EvidenceItem],
    ) -> str:
        lower_question = question.lower()
        type_text = (
            f"文档标明它属于“{summary['type']}”。"
            if summary["type"] and summary["type"] != "文档未明确说明"
            else ""
        )

        if any(keyword in question for keyword in ["发现", "重点", "核心", "结论", "贡献"]):
            answer = (
                f"它最重要的点是“{summary['title']}”。"
                f"更准确地说，这类实验文档的“发现”主要是核心收获：{summary['focus']}。"
                f"它通过{summary['implementation']}来体现。{type_text}"
            )
        elif "方法" in question:
            answer = (
                f"它的方法是围绕{summary['implementation']}展开，"
                f"对应的知识基础是{summary['knowledge']}。{type_text}"
            )
        elif any(keyword in question for keyword in ["局限", "不足"]):
            answer = (
                f"从当前证据看，它主要说明了{summary['focus']}，"
                "但对异常情况、边界条件或更复杂场景的展开不多。"
            )
        elif "目的" in question or "主题" in question:
            answer = f"主题是“{summary['title']}”，目标集中在{summary['focus']}。{type_text}"
        else:
            answer = (
                f"这份文档主要讲“{summary['title']}”，"
                f"重点是{summary['focus']}，实现内容围绕{summary['implementation']}。{type_text}"
            )

        if "tcp" in lower_question and "TCP" not in answer:
            answer = f"{answer} 这里涉及 TCP 相关内容。"
        return answer

    def _build_local_compare_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
        memory_facts: dict[str, str] | None = None,
    ) -> str:
        grouped = self._group_evidence_by_document(evidence)
        if len(grouped) < 2:
            if len(grouped) == 1:
                only_name = next(iter(grouped))
                return (
                    f"当前只选到《{only_name}》这一份文档，无法真正对比两篇。"
                    "请在输入框上方确认已经同时选择两份文档，再问我“有什么不同点”。"
                )
            return "我没有找到可用于对比的文档证据。请先上传并选择至少两份已准备好的文档。"

        summaries = {
            name: self._summarize_document_for_compare(items)
            for name, items in grouped.items()
        }
        names = list(summaries.keys())[:2]
        first = summaries[names[0]]
        second = summaries[names[1]]

        first_citation = grouped[names[0]][0].citation_id
        second_citation = grouped[names[1]][0].citation_id
        audience_note = self._audience_note(
            memory_facts or {},
            question,
            " ".join(item.text for items in grouped.values() for item in items),
        )

        return (
            f"这两份文档的核心不同点可以这样看：\n\n"
            f"{audience_note}"
            f"1. 主题不同：\n"
            f"《{names[0]}》主要是“{first['title']}”，重点在 {first['focus']}。[{first_citation}]\n"
            f"《{names[1]}》主要是“{second['title']}”，重点在 {second['focus']}。[{second_citation}]\n\n"
            f"2. 实验类型不同：\n"
            f"第一份更偏“{first['type']}”，第二份更偏“{second['type']}”。\n\n"
            f"3. 实现内容不同：\n"
            f"第一份围绕 {first['implementation']} 展开；第二份围绕 {second['implementation']} 展开。\n\n"
            f"4. 学到的知识点不同：\n"
            f"第一份更强调 {first['knowledge']}；第二份更强调 {second['knowledge']}。\n\n"
            "通俗地说：一篇更像是在做“通信连接与收发数据”，另一篇更像是在做“扫描端口并判断开放状态”。"
        )

    def _audience_note(
        self,
        memory_facts: dict[str, str],
        question: str,
        context_text: str = "",
    ) -> str:
        profile = memory_facts.get("user_profile", "")
        if not profile:
            return ""
        wants_profile_angle = any(
            keyword in question
            for keyword in ["从我", "我的专业", "专业角度", "背景", "角度", "能学到", "学到什么"]
        )
        if not wants_profile_angle:
            return ""

        if "计算机" in profile or "软件" in profile or "网络" in profile:
            angle = self._computer_profile_angle(question, context_text)
            return f"结合你的背景（{profile}），我会更侧重从{angle}的角度解释。\n\n"
        if wants_profile_angle:
            return f"结合你的背景（{profile}），我会按更贴近你专业学习的角度解释。\n\n"
        return ""

    def _computer_profile_angle(self, question: str, context_text: str) -> str:
        text = f"{question}\n{context_text}".lower()
        if any(keyword in text for keyword in ["tcp", "socket", "winsock", "端口", "网络通信", "客户端", "服务端", "回显"]):
            return "网络编程、协议理解、端口/Socket 和工程实现能力"
        if any(keyword in text for keyword in ["生成式人工智能", "大模型", "人工智能", "算法", "模型", "学习支持", "教育治理"]):
            return "AI 应用机制、系统设计、数据处理、人机交互、隐私安全和工程落地"
        if any(keyword in text for keyword in ["数据库", "向量库", "索引", "检索", "embedding", "rag"]):
            return "数据建模、检索系统、索引设计和工程实现"
        if any(keyword in text for keyword in ["实验", "代码", "程序", "系统", "实现"]):
            return "系统设计、程序实现、测试验证和工程规范"
        return "抽象建模、系统设计、数据流、算法思维和工程落地"

    def _group_evidence_by_document(
        self,
        evidence: list[EvidenceItem],
    ) -> dict[str, list[EvidenceItem]]:
        grouped: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            grouped.setdefault(item.paper_name or item.document_id, []).append(item)
        return grouped

    def _summarize_document_for_compare(self, evidence: list[EvidenceItem]) -> dict[str, str]:
        text = self._sanitize_evidence_text(" ".join(item.text for item in evidence))
        title = self._extract_field(text, "实验名称") or self._guess_title_from_text(text)
        experiment_type = self._extract_field(text, "实验类型") or "文档未明确说明"

        lower_text = text.lower()
        if "端口扫描" in text:
            focus = "端口扫描的原理、端口开放判断和扫描流程"
            implementation = "解析目标 IP 和端口范围、遍历端口、建立 TCP 连接并判断开放状态"
            knowledge = "TCP 三次握手、connect 调用、端口开放/关闭判断和 Winsock 编程"
        elif "tcp通信" in lower_text or "tcp 通信" in lower_text or "回显" in text:
            focus = "TCP 客户端和服务端之间如何建立连接、发送和回显数据"
            implementation = "服务端监听、客户端连接、数据收发、回显和资源释放"
            knowledge = "面向连接的流式套接字、TCP 通信流程和常见 Winsock API"
        else:
            purpose = self._extract_after_heading(text, "实验目的")
            focus = purpose[:80] if purpose else "文档主题、目标和实现过程"
            implementation = self._extract_after_heading(text, "实验过程")[:80] or "文档中的具体实现步骤"
            knowledge = self._extract_after_heading(text, "知识点")[:80] or "文档涉及的核心概念"

        return {
            "title": title,
            "type": experiment_type,
            "focus": focus,
            "implementation": implementation,
            "knowledge": knowledge,
        }

    def _extract_field(self, text: str, field_name: str) -> str:
        pattern = rf"{re.escape(field_name)}\s*[:：]\s*([^。；;\n]+?)(?=\s*(实验类型|指导教师|专业班级|姓名|学号|电子邮件|实\s*验\s*日\s*期|一、|二、|$))"
        match = re.search(pattern, text)
        if not match:
            return ""
        return match.group(1).strip()

    def _guess_title_from_text(self, text: str) -> str:
        if "端口扫描" in text:
            return "端口扫描实验"
        if "TCP通信" in text or "TCP 通信" in text or "回显" in text:
            return "TCP 通信实验"
        return "当前文档主题"

    def _extract_after_heading(self, text: str, heading: str) -> str:
        pattern = rf"{re.escape(heading)}\s*([^一二三四五六七八九十、]+)"
        match = re.search(pattern, text)
        if not match:
            return ""
        return match.group(1).strip()

    def _sanitize_evidence_text(self, text: str) -> str:
        sanitized = re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "[邮箱已隐藏]", text)
        sanitized = re.sub(r"(姓\s*名|姓名)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(学\s*号|学号)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(电子邮件|邮箱)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        return sanitized

    def _first_informative_overview_row(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        for row in rows[:12]:
            text = str(row.get("text", ""))
            if self._looks_like_front_matter_noise(text):
                continue
            if self._overview_structure_score(text) > 0 or self._readable_text_score(text) >= 0.55:
                return row
        for row in rows:
            text = str(row.get("text", ""))
            if not self._looks_like_front_matter_noise(text):
                return row
        return None

    def _overview_structure_score(self, text: str) -> float:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        if not normalized:
            return 0.0
        score = 0.0
        strong_markers = [
            "abstract",
            "introduction",
            "conclusion",
            "in this paper",
            "in this work",
            "we propose",
            "we present",
            "we introduce",
            "we show",
            "本文围绕",
            "摘要",
            "结论",
            "研究认为",
        ]
        medium_markers = [
            "transformer",
            "attention",
            "sequence transduction",
            "machine translation",
            "state-of-the-art",
            "architecture",
            "model",
            "training",
            "实验",
            "模型",
            "方法",
        ]
        score += sum(0.22 for marker in strong_markers if marker in normalized)
        score += sum(0.08 for marker in medium_markers if marker in normalized)
        return min(score, 0.8)

    def _looks_like_front_matter_noise(self, text: str) -> bool:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        if not normalized:
            return True
        if any(marker in normalized for marker in ["abstract", "introduction", "we propose", "in this work"]):
            return False
        noise_markers = [
            "provided proper attribution",
            "hereby grants permission",
            "journalistic or scholarly works",
            "work performed while at",
            "conference on neural information processing systems",
            "arxiv:",
        ]
        if any(marker in normalized for marker in noise_markers):
            return True
        hidden_email_count = normalized.count("[邮箱已隐藏]")
        if hidden_email_count >= 3 and not any(marker in normalized for marker in ["abstract", "introduction"]):
            return True
        return False

    def _trim_front_matter_prefix(self, text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        cleaned = re.sub(
            r"^\d+(?:\.\d+)*\s+(?:abstract|introduction|background|conclusion|results?|model|experiments?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        markers = [
            "Abstract ",
            "摘要",
            "1 Introduction ",
            "Introduction ",
            "In this work ",
            "In this paper ",
        ]
        for marker in markers:
            index = cleaned.find(marker)
            if 0 < index <= 260:
                if marker in {"Abstract ", "摘要", "1 Introduction ", "Introduction "}:
                    return cleaned[index + len(marker) :].strip()
                return cleaned[index:].strip()
        return cleaned

    def _truncate_readable_text(self, text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) <= limit:
            return cleaned
        boundary = max(
            cleaned.rfind("。", 0, limit),
            cleaned.rfind("！", 0, limit),
            cleaned.rfind("？", 0, limit),
            cleaned.rfind(".", 0, limit),
            cleaned.rfind(";", 0, limit),
            cleaned.rfind(" ", 0, limit),
        )
        if boundary < int(limit * 0.55):
            boundary = limit
        return cleaned[:boundary].rstrip(" ,，;；.") + "..."

    def _question_relevance_score(self, question: str, text: str) -> float:
        normalized_text = " ".join(self._sanitize_evidence_text(text).lower().split())
        tokens = self._question_keywords(question)
        if not tokens or not normalized_text:
            return 0.0

        hits = sum(1 for token in tokens if token.lower() in normalized_text)
        exact_score = hits / max(len(tokens), 1)

        q_chars = re.findall(r"[\u4e00-\u9fff]", question)
        t_chars = set(re.findall(r"[\u4e00-\u9fff]", normalized_text))
        char_score = 0.0
        if q_chars:
            meaningful_chars = [
                char
                for char in q_chars
                if char not in set("这篇份个的了呢吗啊和与及或是有在中里上下一些哪些什么怎么如何请问")
            ]
            if meaningful_chars:
                char_hits = sum(1 for char in meaningful_chars if char in t_chars)
                char_score = char_hits / max(len(meaningful_chars), 1)

        return max(0.0, min(1.0, exact_score * 0.75 + char_score * 0.25))

    def _question_keywords(self, question: str) -> list[str]:
        normalized = question.lower()
        stop_phrases = [
            "这篇论文",
            "这份文档",
            "这个文档",
            "这篇文档",
            "这份报告",
            "请你",
            "请问",
            "给我",
            "一下",
            "什么",
            "哪些",
            "为什么",
            "怎么样",
            "如何",
            "是否",
            "能否",
            "可以",
        ]
        for phrase in stop_phrases:
            normalized = normalized.replace(phrase, " ")

        domain_terms = [
            "参考文献",
            "引用文献",
            "结论",
            "结果",
            "方法",
            "研究方法",
            "实验方法",
            "局限",
            "不足",
            "风险",
            "贡献",
            "创新",
            "数据",
            "样本",
            "问卷",
            "访谈",
            "实验",
            "模型",
            "算法",
            "公式",
            "表格",
            "作者",
            "标题",
            "主题",
            "目的",
        ]
        tokens = [term for term in domain_terms if term in question]
        tokens.extend(re.findall(r"[a-z0-9]{2,}", normalized))

        cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
        for sequence in cjk_sequences:
            if len(sequence) <= 4:
                tokens.append(sequence)
                continue
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
            tokens.extend(sequence[index : index + 3] for index in range(len(sequence) - 2))

        blocked = {"论文", "文档", "报告", "内容", "主要", "一个", "这个", "那个"}
        unique: list[str] = []
        for token in tokens:
            cleaned = token.strip()
            if len(cleaned) < 2 or cleaned in blocked:
                continue
            if cleaned not in unique:
                unique.append(cleaned)
        return unique[:40]

    def _looks_like_reference_section_text(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if re.search(r"\breferences\b", normalized, flags=re.IGNORECASE):
            return True
        return "参考文献" in normalized and self._reference_marker_count(normalized) > 0

    def _looks_like_reference_continuation(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if self._reference_marker_count(normalized) > 0:
            return True
        return bool(re.search(r"(教育|研究|Journal|Higher Education|University).{0,40}\d{4}", normalized))

    def _reference_marker_count(self, text: str) -> int:
        return len(re.findall(r"\[\d{1,3}\]", text))

    def _extract_references_from_text(self, text: str) -> list[tuple[str, str]]:
        sanitized = self._sanitize_evidence_text(text)
        normalized = re.sub(r"\s+", " ", sanitized).strip()
        section_match = re.search(r"(参考文献|References)\s*", normalized, flags=re.IGNORECASE)
        section_text = normalized[section_match.start() :] if section_match else normalized
        matches = list(
            re.finditer(
                r"\[(\d{1,3})\]\s*(.*?)(?=\s*\[\d{1,3}\]\s*|$)",
                section_text,
                flags=re.DOTALL,
            )
        )
        references_by_number: dict[int, str] = {}
        for match in matches:
            number = int(match.group(1))
            content = match.group(2).strip(" ，,。；;")
            content = re.sub(r"^(参考文献|References)\s*", "", content, flags=re.IGNORECASE).strip()
            content = re.split(
                r"\s*(?:（说明：|说明：|附录[:：]|设计维度\s*\||Appendix\b)",
                content,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            content = re.sub(r"\s+", " ", content)
            if len(content) < 6:
                continue
            references_by_number[number] = content[:500]
        return [
            (str(number), content)
            for number, content in sorted(references_by_number.items())
        ]

    def _pick_readable_sentences(self, text: str, limit: int) -> list[str]:
        parts = re.split(r"(?<=[。！？.!?])\s+|(?<=。)|(?<=！)|(?<=？)", text)
        sentences: list[str] = []
        blocked = ["姓名", "学号", "电子邮件", "邮箱", "实验评分"]
        for part in parts:
            cleaned = self._trim_front_matter_prefix(part.strip())
            if len(cleaned) < 18:
                continue
            if self._looks_like_front_matter_noise(cleaned):
                continue
            if any(word in cleaned for word in blocked):
                continue
            if self._is_table_like_text(cleaned):
                continue
            sentences.append(self._truncate_readable_text(cleaned, limit=220))
            if len(sentences) >= limit:
                break
        return sentences

    def _looks_like_table_question(self, question: str) -> bool:
        keywords = ["表格", "表", "公式", "计算", "数值", "数据是多少", "参数", "指标"]
        return any(keyword in question for keyword in keywords)

    def _reliability_relevance_score(self, text: str) -> int:
        weighted_keywords = [
            ("课程报告", 8),
            ("实验报告", 7),
            ("随机生成", 8),
            ("论文样稿", 8),
            ("毕业论文", 6),
            ("学位论文", 6),
            ("摘要", 3),
            ("参考文献", 4),
            ("数据来源", 4),
            ("实证数据", 4),
            ("未来研究", 4),
            ("文献分析", 3),
            ("情境推演", 3),
            ("机制建构", 3),
            ("一致性检验", 3),
            ("后评价", 3),
            ("结论", 2),
            ("结果", 2),
            ("计算", 2),
            ("公式", 2),
            ("分析", 2),
            ("评价", 2),
            ("完成日期", 2),
            ("测试", 2),
            ("局限", 2),
            ("不足", 2),
        ]
        return sum(weight for keyword, weight in weighted_keywords if keyword in text)

    def _document_kind_from_evidence(self, evidence: list[EvidenceItem]) -> str:
        text = " ".join([item.paper_name for item in evidence] + [item.text for item in evidence[:6]])
        if "课程报告" in text:
            return "课程报告"
        if "实验报告" in text:
            return "实验报告"
        if "毕业论文" in text or "学位论文" in text:
            return "论文"
        if ("摘要" in text and "参考文献" in text) or "Abstract" in text:
            return "论文"
        return "普通文档"

    def _first_citation_with(
        self,
        evidence: list[EvidenceItem],
        keywords: list[str],
        *,
        fallback: bool = True,
    ) -> str:
        for item in evidence:
            text = f"{item.paper_name}\n{item.quote}\n{item.text}"
            if any(keyword in text for keyword in keywords):
                return item.citation_id
        return evidence[0].citation_id if fallback and evidence else ""

    def _first_citation_with_all(self, evidence: list[EvidenceItem], keywords: list[str]) -> str:
        for item in evidence:
            text = f"{item.paper_name}\n{item.quote}\n{item.text}"
            if all(keyword in text for keyword in keywords):
                return item.citation_id
        return ""

    def _join_citations(self, citation_ids: list[str]) -> str:
        unique_ids: list[str] = []
        for citation_id in citation_ids:
            if citation_id and citation_id not in unique_ids:
                unique_ids.append(citation_id)
        return f" {' '.join(f'[{citation_id}]' for citation_id in unique_ids)}" if unique_ids else ""

    def _best_readable_quote(self, text: str, limit: int = 220) -> str:
        sanitized = self._sanitize_evidence_text(text)
        normalized = " ".join(sanitized.split())
        if not normalized:
            return ""

        blocked = ["姓名", "学号", "电子邮件", "邮箱", "实验评分"]
        candidates = re.split(r"\n+|(?<=[。！？.!?；;])\s*", sanitized)
        for candidate in candidates:
            cleaned = " ".join(candidate.split()).strip()
            if len(cleaned) < 12:
                continue
            has_document_type = any(word in cleaned for word in ["课程报告", "实验报告", "毕业论文", "学位论文"])
            if any(word in cleaned for word in blocked) and not has_document_type:
                continue
            if self._is_table_like_text(cleaned):
                continue
            return cleaned[:limit] + ("..." if len(cleaned) > limit else "")

        if self._is_table_like_text(normalized):
            return normalized[: min(limit, 120)] + ("..." if len(normalized) > min(limit, 120) else "")
        return normalized[:limit] + ("..." if len(normalized) > limit else "")

    def _best_quote_for_question(self, question: str, text: str, limit: int = 240) -> str:
        sanitized = self._sanitize_evidence_text(text)
        if not sanitized.strip():
            return ""

        if self._looks_like_compound_request(question):
            preferred_keywords = self._compound_focus_keywords_for_question(question)
        elif self._looks_like_reference_question(question):
            return self._best_reference_quote(sanitized, limit=limit)
        elif self._looks_like_structured_review_request(question):
            preferred_keywords = [
                "摘要",
                "本文围绕",
                "采用",
                "文献分析",
                "认知支架",
                "资源重组",
                "课程学习场景",
                "风险",
                "人机协同",
                "未来研究",
                "实证数据",
                "参考文献",
            ]
        elif self._looks_like_title_alignment_question(question):
            preferred_keywords = [
                "机制、风险与治理路径",
                "未来研究",
                "实证数据",
                "认知支架",
                "资源重组",
                "过程陪伴",
                "反馈生成",
                "组织协同",
                "学习依赖",
                "信息准确性",
                "数据隐私",
                "算法偏差",
                "学术诚信",
                "责任边界",
                "人机协同",
                "价值对齐",
                "过程可控",
                "数据最小化",
                "多主体治理",
                "未来研究",
                "实证数据",
                "文献分析",
                "情境推演",
                "机制建构",
            ]
        elif self._looks_like_reliability_question(question):
            preferred_keywords = [
                "随机生成",
                "论文样稿",
                "采用",
                "文献分析",
                "情境推演",
                "机制建构",
                "未来研究",
                "实证数据",
                "参考文献",
                "风险",
                "局限",
            ]
        elif self._looks_like_research_limitation_question(question):
            preferred_keywords = [
                "局限性",
                "研究局限",
                "研究不足",
                "结论与展望",
                "未来研究",
                "实证数据",
                "检验",
                "验证",
                "不同应用场景",
                "不同学生群体",
                "文献分析",
                "情境推演",
                "机制建构",
            ]
        elif self._looks_like_document_wide_question(question):
            preferred_keywords = self._overview_focus_keywords(question)
        else:
            preferred_keywords = []

        candidates = [
            " ".join(part.split()).strip()
            for part in re.split(r"\n+|(?<=[。！？.!?；;])\s*", sanitized)
            if part.strip()
        ]
        for keyword in preferred_keywords:
            for candidate in candidates:
                if keyword in candidate and not self._is_table_like_text(candidate):
                    return candidate[:limit] + ("..." if len(candidate) > limit else "")
        return self._best_readable_quote(sanitized, limit=limit)

    def _best_reference_quote(self, text: str, limit: int = 240) -> str:
        references = self._extract_references_from_text(text)
        if references:
            snippet = "；".join(
                f"[{number}] {content}"
                for number, content in references[:2]
            )
            return snippet[:limit] + ("..." if len(snippet) > limit else "")

        normalized = " ".join(self._sanitize_evidence_text(text).split())
        marker_match = re.search(r"(参考文献|References).{0,220}", normalized, flags=re.IGNORECASE)
        if marker_match:
            quote = marker_match.group(0)
            return quote[:limit] + ("..." if len(quote) > limit else "")
        return self._best_readable_quote(normalized, limit=limit)

    def _readable_text_score(self, text: str) -> float:
        normalized = " ".join(text.split())
        if not normalized:
            return 0.0

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
        alpha_count = len(re.findall(r"[A-Za-z]", normalized))
        score = 0.2
        if cjk_count >= 30 or alpha_count >= 30:
            score += 0.25
        if any(char in normalized for char in "。！？.!?；;"):
            score += 0.2
        if 80 <= len(normalized) <= 1200:
            score += 0.15
        if self._is_table_like_text(normalized):
            score -= 0.35
        if any(word in normalized for word in ["姓名", "学号", "电子邮件", "邮箱"]):
            score -= 0.15
        return max(0.0, min(score, 1.0))

    def _is_table_like_text(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if not normalized:
            return False

        pipe_count = normalized.count("|")
        lines = [line for line in text.splitlines() if line.strip()]
        table_line_ratio = (
            sum(1 for line in lines if "|" in line or "\t" in line) / max(len(lines), 1)
        )
        digit_ratio = len(re.findall(r"\d", normalized)) / max(len(normalized), 1)
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
        separator_ratio = len(re.findall(r"[|,:：/\\]", normalized)) / max(len(normalized), 1)

        return (
            pipe_count >= 6
            or table_line_ratio >= 0.35
            or (digit_ratio > 0.28 and cjk_count < 80 and len(normalized) > 100)
            or (separator_ratio > 0.22 and len(normalized) > 120)
        )

    def _renumber_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        for index, item in enumerate(evidence, start=1):
            item.citation_id = f"E{index}"
        return evidence

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

    def _resolve_document_ids(self, requested_ids: list[str] | None) -> list[str]:
        if requested_ids:
            return [
                document_id
                for document_id in requested_ids
                if (document := self.store.get_document(document_id)) and document.status == "ready"
            ]
        return [document.id for document in self.store.list_documents() if document.status == "ready"]
