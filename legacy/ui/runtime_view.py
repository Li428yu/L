from __future__ import annotations

from typing import Any

import streamlit as st

from ui.state import get_runtime_focus


def node_label(node_name: str) -> str:
    labels = {
        "planner": "planner",
        "tools": "tools",
        "assistant": "assistant",
        "memory": "memory",
    }
    return labels.get(node_name, node_name)


def format_runtime_path(nodes: list[str] | None) -> str:
    nodes = nodes or []
    if not nodes:
        return "`__start__` -> `planner` -> `...` -> `__end__`"
    return " -> ".join(["`__start__`", *[f"`{node}`" for node in nodes], "`__end__`"])


def build_runtime_graphviz_dot(
    traversed_nodes: list[str] | None = None,
    active_node: str | None = None,
) -> str:
    traversed_nodes = traversed_nodes or []

    def node_style(node_name: str) -> str:
        if node_name == active_node:
            return 'fillcolor="#fed7aa", color="#ea580c", penwidth="2.4"'
        if node_name in traversed_nodes:
            return 'fillcolor="#dcfce7", color="#16a34a", penwidth="2.2"'
        return 'fillcolor="#eef2ff", color="#94a3b8", penwidth="1.4"'

    return f"""
digraph LangGraphRuntime {{
    rankdir=LR;
    bgcolor="transparent";
    graph [pad="0.24", nodesep="0.55", ranksep="0.72"];
    node [shape=box, style="rounded,filled", fontname="Microsoft YaHei", fontsize=13, margin="0.18,0.12"];
    edge [fontname="Microsoft YaHei", fontsize=11, color="#64748b"];

    start [label="__start__", shape=oval, fillcolor="#f8fafc", color="#94a3b8"];
    planner [label="planner\\n判断这一轮先走哪条路", {node_style("planner")}];
    tools [label="tools\\n调用检索 / 列论文 / 生成摘要卡", {node_style("tools")}];
    assistant [label="assistant\\n整合工具结果并生成最终回答", {node_style("assistant")}];
    end [label="__end__", shape=oval, fillcolor="#f8fafc", color="#94a3b8"];
    memory [label="InMemorySaver\\n线程级多轮记忆", shape=note, fillcolor="#fff7ed", color="#f59e0b", style="rounded,filled,dashed"];

    start -> planner [label="收到用户问题"];
    planner -> tools [label="需要先调工具"];
    planner -> assistant [label="可直接回答", style=dashed];
    tools -> assistant [label="拿到工具结果"];
    assistant -> tools [label="需要继续调工具", style=dashed];
    assistant -> end [label="完成回答"];
    memory -> planner [label="恢复上文", style=dotted, color="#f59e0b"];
    assistant -> memory [label="写入本轮消息", style=dotted, color="#f59e0b"];
}}
""".strip()


def render_runtime_legend() -> None:
    st.caption("图例：绿色 = 本轮已经执行过，橙色 = 当前正在执行，灰蓝 = 这一轮还没走到，黄色便签 = 会话记忆组件。")


def render_runtime_learning_notes() -> None:
    note_cols = st.columns(2)
    note_cols[0].markdown(
        "**planner**\n\n先判断这一轮应该直接回答，还是先去检索论文证据、列出论文、生成摘要卡。"
    )
    note_cols[0].markdown(
        "**tools**\n\n真正执行工具调用。当前内置了检索证据、列出论文、生成摘要卡三类工具。"
    )
    note_cols[1].markdown(
        "**assistant**\n\n把工具结果和上文记忆拼起来，生成你最终看到的回答。"
    )
    note_cols[1].markdown(
        "**memory**\n\n不是单独执行的 node，而是在每一轮开始和结束时帮 agent 恢复、保存上下文。"
    )


def render_tool_traces(tool_traces: list[Any]) -> None:
    if not tool_traces:
        st.info("这一轮没有工具调用记录。")
        return

    for trace in tool_traces:
        with st.expander(trace.tool_name, expanded=False):
            if trace.tool_args:
                st.json(trace.tool_args)
            preview = trace.tool_output
            if len(preview) > 1200:
                preview = f"{preview[:1200]}..."
            st.code(preview, language="json")


def render_evidence(evidence: list[Any]) -> None:
    if not evidence:
        st.info("这一轮没有产生新的检索证据。")
        return

    for item in evidence:
        with st.expander(
            f"{item.chunk.paper_name} | Page {item.chunk.page} | Score {item.score:.3f}",
            expanded=False,
        ):
            st.write(item.chunk.text)


def render_runtime_logs(runtime_logs: list[Any]) -> None:
    if not runtime_logs:
        st.info("这一轮还没有节点日志。")
        return

    for log in runtime_logs:
        st.markdown(f"**{log.step_index}. `{node_label(log.node_name)}`**")
        st.write(log.summary)


def render_runtime_panel() -> None:
    focus_index, focus_turn = get_runtime_focus()

    st.subheader("Runtime 可视化")
    st.caption("右侧单独解释 LangGraph 在这一轮里怎么走，方便新手边用边理解底层结构。")

    if focus_turn is None:
        st.graphviz_chart(build_runtime_graphviz_dot(), width="stretch")
        render_runtime_legend()
        render_runtime_learning_notes()
        st.info("发送问题后，这里会显示流程图、节点日志、工具调用和证据。")
        return

    latest_index = len(st.session_state.turn_records) - 1
    turn_number = (focus_index or 0) + 1
    if focus_index == latest_index:
        st.caption(f"当前查看：第 {turn_number} 轮（最新一轮）")
    else:
        st.caption(f"当前查看：第 {turn_number} 轮（从左侧历史切换而来）")

    tab_graph, tab_logs, tab_tools, tab_evidence, tab_source = st.tabs(
        ["流程图", "节点日志", "工具调用", "检索证据", "Mermaid 源码"]
    )

    with tab_graph:
        st.graphviz_chart(
            build_runtime_graphviz_dot(traversed_nodes=focus_turn.get("traversed_nodes")),
            width="stretch",
        )
        render_runtime_legend()
        st.markdown(f"**执行路径**\n\n{format_runtime_path(focus_turn.get('traversed_nodes'))}")
        render_runtime_learning_notes()

    with tab_logs:
        render_runtime_logs(focus_turn.get("runtime_node_logs", []))

    with tab_tools:
        render_tool_traces(focus_turn.get("tool_traces", []))

    with tab_evidence:
        render_evidence(focus_turn.get("evidence", []))

    with tab_source:
        graph_mermaid = focus_turn.get("graph_mermaid") or st.session_state.agent_graph_mermaid
        if graph_mermaid:
            st.code(graph_mermaid, language="mermaid")
        else:
            st.info("这一轮还没有可展示的 Mermaid 源码。")
