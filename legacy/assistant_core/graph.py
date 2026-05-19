from __future__ import annotations

from typing import Any, Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import numpy as np

from assistant_core.types import AgentTurnResult, Chunk, NodeExecutionLog, PaperOverview


class PaperAgentGraphMixin:
    def ask_agent(
        self,
        question: str,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        top_k: int,
        paper_ids: set[str] | None,
        thread_id: str,
    ) -> AgentTurnResult:
        final_result: AgentTurnResult | None = None
        for event in self.stream_agent_turn(
            question=question,
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            top_k=top_k,
            paper_ids=paper_ids,
            thread_id=thread_id,
        ):
            if event["type"] == "final":
                final_result = event["result"]

        if final_result is None:
            raise RuntimeError("Agent turn did not produce a final result.")

        return final_result

    def stream_agent_turn(
        self,
        question: str,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        top_k: int,
        paper_ids: set[str] | None,
        thread_id: str,
    ) -> Iterator[dict[str, Any]]:
        agent = self._build_chat_agent(
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            default_top_k=top_k,
            default_paper_ids=paper_ids,
        )
        config = {"configurable": {"thread_id": thread_id}}
        previous_messages = self._get_thread_messages(agent, config)
        graph_mermaid = agent.get_graph().draw_mermaid()
        runtime_node_logs: list[NodeExecutionLog] = []

        yield {
            "type": "graph",
            "graph_mermaid": graph_mermaid,
        }

        for event in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="updates",
        ):
            new_logs = self._extract_runtime_logs_from_event(
                event=event,
                start_index=len(runtime_node_logs) + 1,
            )
            runtime_node_logs.extend(new_logs)
            traversed_nodes = [log.node_name for log in runtime_node_logs]
            for log in new_logs:
                yield {
                    "type": "node_log",
                    "log": log,
                    "traversed_nodes": traversed_nodes.copy(),
                    "graph_mermaid": graph_mermaid,
                }

        all_messages = self._get_thread_messages(agent, config)
        new_messages = all_messages[len(previous_messages) :]
        traversed_nodes = [log.node_name for log in runtime_node_logs]

        yield {
            "type": "final",
            "result": AgentTurnResult(
                answer=self._extract_final_answer(new_messages, all_messages),
                evidence=self._extract_latest_evidence(new_messages),
                tool_traces=self._extract_tool_traces(new_messages),
                messages=all_messages,
                runtime_node_logs=runtime_node_logs,
                traversed_nodes=traversed_nodes,
                graph_mermaid=graph_mermaid,
            ),
        }

    def _build_chat_agent(
        self,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        default_top_k: int,
        default_paper_ids: set[str] | None,
    ):
        paper_overviews = self.list_papers(chunks)
        paper_name_to_id = {
            overview.paper_name: overview.paper_id for overview in paper_overviews
        }
        available_names = list(paper_name_to_id.keys())
        default_scope_names = self._scope_names_from_ids(paper_name_to_id, default_paper_ids)
        tools = self._build_agent_tools(
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            default_top_k=default_top_k,
            default_paper_ids=default_paper_ids,
            paper_name_to_id=paper_name_to_id,
            paper_overviews=paper_overviews,
        )
        model_with_tools = self.llm.bind_tools(tools)
        system_prompt = self._build_agent_prompt(
            paper_overviews=paper_overviews,
            default_paper_ids=default_paper_ids,
        )

        def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            response = model_with_tools.invoke(
                [SystemMessage(content=system_prompt), *state["messages"]]
            )
            return {"messages": [response]}

        def plan_next_tool(state: MessagesState) -> dict[str, list[BaseMessage]]:
            tool_call = self._build_planner_tool_call(
                messages=state["messages"],
                available_names=available_names,
                default_scope_names=default_scope_names,
            )
            if tool_call is None:
                return {}
            return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        builder = StateGraph(MessagesState)
        builder.add_node("planner", plan_next_tool)
        builder.add_node("assistant", call_model)
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=self._handle_tool_error))
        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"tools": "tools", "assistant": "assistant"},
        )
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", "__end__": END},
        )
        builder.add_edge("tools", "assistant")
        return builder.compile(checkpointer=self.agent_memory)

    def _handle_tool_error(self, exc: Exception) -> str:
        return self._to_json(
            {
                "type": "tool_error",
                "error": str(exc),
                "guidance": "工具执行时出现了临时问题。请优先用中文向用户解释，并建议稍后重试。",
            }
        )

    def _build_agent_tools(
        self,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        default_top_k: int,
        default_paper_ids: set[str] | None,
        paper_name_to_id: dict[str, str],
        paper_overviews: Sequence[PaperOverview],
    ) -> list:
        available_names = list(paper_name_to_id.keys())

        @tool
        def list_indexed_papers() -> str:
            """列出当前索引中的论文名称、页数和 chunk 数。"""

            payload = {
                "type": "paper_index",
                "paper_count": len(paper_overviews),
                "papers": [
                    {
                        "paper_id": overview.paper_id,
                        "paper_name": overview.paper_name,
                        "page_count": overview.page_count,
                        "chunk_count": overview.chunk_count,
                    }
                    for overview in paper_overviews
                ],
            }
            return self._to_json(payload)

        @tool
        def retrieve_paper_evidence(
            query: str,
            paper_names: list[str] | None = None,
            top_k: int | None = None,
        ) -> str:
            """检索与问题相关的论文证据片段。回答论文事实问题前应先调用这个工具。"""

            resolved_paper_ids = default_paper_ids
            matched_names = self._scope_names_from_ids(paper_name_to_id, default_paper_ids)
            unmatched_names: list[str] = []

            if paper_names:
                matched_names, unmatched_names = self._resolve_paper_names(
                    requested_names=paper_names,
                    available_names=available_names,
                )
                resolved_paper_ids = {
                    paper_name_to_id[name] for name in matched_names if name in paper_name_to_id
                }

            effective_top_k = top_k or default_top_k
            evidence = self.search(
                query=query,
                chunks=chunks,
                chunk_vectors=chunk_vectors,
                top_k=effective_top_k,
                paper_ids=resolved_paper_ids,
            )

            payload = {
                "type": "retrieval_result",
                "query": query,
                "requested_paper_names": paper_names or [],
                "matched_paper_names": matched_names,
                "unmatched_paper_names": unmatched_names,
                "effective_top_k": effective_top_k,
                "result_count": len(evidence),
                "evidence": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "paper_id": item.chunk.paper_id,
                        "paper_name": item.chunk.paper_name,
                        "page": item.chunk.page,
                        "score": round(item.score, 4),
                        "text": item.chunk.text,
                    }
                    for item in evidence
                ],
            }
            if not evidence:
                payload["guidance"] = "当前范围没有检索到足够证据，请尝试更具体的问题或调整论文范围。"

            return self._to_json(payload)

        @tool
        def generate_paper_digest_tool(paper_name: str) -> str:
            """为单篇论文生成结构化摘要卡片。需要传入论文名称。"""

            matched_names, _ = self._resolve_paper_names(
                requested_names=[paper_name],
                available_names=available_names,
            )
            if not matched_names:
                return self._to_json(
                    {
                        "type": "paper_digest",
                        "paper_name": paper_name,
                        "error": "未找到对应论文，请先调用 list_indexed_papers 查看可用论文名。",
                    }
                )

            resolved_name = matched_names[0]
            resolved_paper_id = paper_name_to_id[resolved_name]
            paper_chunks = [chunk for chunk in chunks if chunk.paper_id == resolved_paper_id]
            digest = self.generate_paper_digest(resolved_name, paper_chunks)
            return self._to_json(
                {
                    "type": "paper_digest",
                    "paper_name": resolved_name,
                    "digest": digest,
                }
            )

        return [list_indexed_papers, retrieve_paper_evidence, generate_paper_digest_tool]

    def _build_agent_prompt(
        self,
        paper_overviews: Sequence[PaperOverview],
        default_paper_ids: set[str] | None,
    ) -> str:
        all_papers_text = "\n".join(
            f"- {overview.paper_name}（{overview.page_count} 页，{overview.chunk_count} 个片段）"
            for overview in paper_overviews
        )
        default_scope_names = [
            overview.paper_name
            for overview in paper_overviews
            if default_paper_ids is None or overview.paper_id in default_paper_ids
        ]
        default_scope_text = "、".join(default_scope_names) if default_scope_names else "当前未限制"

        return (
            "你是一个谨慎、可靠的中文论文阅读 agent。\n"
            "你运行在一个带工具节点和会话记忆的 LangGraph 工作流里，需要根据用户问题决定是否调用工具。\n"
            "规则如下：\n"
            "1. 默认把用户输入视为有效的论文任务，不要轻易回复“无法识别需求”。\n"
            "2. 只要问题涉及论文内容、结论、方法、实验、结果、对比，请先调用 retrieve_paper_evidence，再基于证据回答。\n"
            "3. 如果用户提到“这两篇论文”“第一篇”“前两篇”“那篇论文”“刚才更好的那篇”“继续说它”这类指代，"
            "请优先结合当前对话记忆和默认检索范围理解；如果仍不明确，再调用 list_indexed_papers。\n"
            "4. 如果用户要求摘要卡片、结构化总结或阅读卡片，请调用 generate_paper_digest_tool。\n"
            "5. 回答必须基于工具结果或当前对话历史，不要猜测。\n"
            "6. 如果工具返回 error 或 tool_error，要说明这是临时接口或网络问题，不要假装已经拿到证据。\n"
            "7. 如果证据不足，要明确说明缺了什么。\n"
            "8. 如果使用了检索到的证据，请在相关句子后面用 [paper_name p.page] 形式做行内引用。\n"
            "9. 默认使用中文回答，除非用户明确要求其他语言。\n\n"
            "下面这些都属于有效需求，你应该直接处理，而不是拒绝：\n"
            "- “这两篇论文里哪一篇结果更好？”\n"
            "- “继续说刚才那篇的方法重点是什么？”\n"
            "- “请对比前两篇论文的实验结论。”\n\n"
            f"当前默认检索范围：{default_scope_text}\n"
            "当前索引中的论文：\n"
            f"{all_papers_text}"
        )

    def _route_after_planner(self, state: MessagesState) -> str:
        latest_message = state["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.tool_calls:
            return "tools"
        return "assistant"
