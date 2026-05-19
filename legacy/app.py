from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
import streamlit as st

from paper_assistant import PaperAssistant
from ui.conversation_view import (
    render_current_turn_panel,
    render_current_turn_preview,
    render_history_panel,
)
from ui.runtime_view import (
    build_runtime_graphviz_dot,
    node_label,
    render_runtime_legend,
    render_runtime_panel,
)
from ui.setup_view import render_index_library, render_setup_panel
from ui.state import build_turn_record, init_session_state, sync_selected_paper_names


load_dotenv()

st.set_page_config(
    page_title="论文阅读助手",
    page_icon="📚",
    layout="wide",
)


def render_scope_selector() -> tuple[dict[str, str], list[str]]:
    if not st.session_state.paper_overviews:
        return {}, []

    sync_selected_paper_names()

    with st.expander("对话范围设置", expanded=False):
        paper_name_to_id = {
            overview.paper_name: overview.paper_id for overview in st.session_state.paper_overviews
        }
        default_scope = list(paper_name_to_id.keys())
        selected_paper_names = st.multiselect(
            "默认检索范围",
            options=default_scope,
            key="selected_paper_names",
            help="Agent 默认会优先在这些论文里检索；如果记忆不够清楚，它也可以先调用工具查看全部论文。",
        )
        st.caption("如果你想换话题，可以点中间区域的“新建对话”，这会清空记忆，但不会删除索引。")

    return paper_name_to_id, selected_paper_names


def render_three_column_workbench(
    selected_paper_names: list[str],
) -> tuple[Any, Any]:
    history_col, main_col, runtime_col = st.columns([0.95, 1.45, 1.25], gap="large")

    with history_col:
        history_mount = st.empty()
    with main_col:
        main_mount = st.empty()
    with runtime_col:
        runtime_mount = st.empty()

    with history_mount.container():
        render_history_panel()

    chat_disabled = not st.session_state.paper_overviews
    with main_mount.container():
        render_current_turn_panel(
            scope_names=selected_paper_names,
            chat_disabled=chat_disabled,
        )

    with runtime_mount.container():
        render_runtime_panel()

    return main_mount, runtime_mount


def handle_agent_turn(
    assistant: PaperAssistant,
    paper_name_to_id: dict[str, str],
    selected_paper_names: list[str],
    top_k: int,
    runtime_demo_delay: float,
    main_mount: Any,
    runtime_mount: Any,
) -> None:
    ask_clicked = bool(st.session_state.get("_ask_clicked"))
    if not ask_clicked:
        return

    cleaned_question = str(st.session_state.get("_latest_question_input_snapshot", "")).strip()
    selected_paper_ids = {
        paper_name_to_id[paper_name]
        for paper_name in selected_paper_names
        if paper_name in paper_name_to_id
    }

    if not cleaned_question:
        st.warning("先输入一个问题。")
        return
    if not selected_paper_ids:
        st.warning("至少选择一篇论文作为默认检索范围。")
        return

    try:
        live_logs: list[Any] = []
        turn_result = None

        with main_mount.container():
            render_current_turn_preview(cleaned_question)

        with runtime_mount.container():
            st.subheader("Runtime 可视化")
            st.caption("这一轮正在实时运行中。你可以直接看到 node 是怎样一步步经过的。")
            live_status_placeholder = st.empty()
            live_graph_placeholder = st.empty()
            live_logs_placeholder = st.empty()

            live_status_placeholder.info("Agent 已启动，准备进入 `planner`。")
            live_graph_placeholder.graphviz_chart(
                build_runtime_graphviz_dot(),
                width="stretch",
            )
            render_runtime_legend()

        for event in assistant.stream_agent_turn(
            question=cleaned_question,
            chunks=st.session_state.chunks,
            chunk_vectors=st.session_state.chunk_vectors,
            top_k=top_k,
            paper_ids=selected_paper_ids,
            thread_id=st.session_state.agent_thread_id,
        ):
            if event["type"] == "graph":
                live_graph_placeholder.graphviz_chart(
                    build_runtime_graphviz_dot(),
                    width="stretch",
                )
                continue

            if event["type"] == "node_log":
                log = event["log"]
                live_logs.append(log)
                traversed_nodes = event["traversed_nodes"]

                live_status_placeholder.info(f"当前执行：`{node_label(log.node_name)}`")
                live_graph_placeholder.graphviz_chart(
                    build_runtime_graphviz_dot(
                        traversed_nodes=traversed_nodes,
                        active_node=log.node_name,
                    ),
                    width="stretch",
                )

                with live_logs_placeholder.container():
                    st.markdown("#### 本轮实时节点日志")
                    for item in live_logs:
                        st.write(f"{item.step_index}. `{item.node_name}` - {item.summary}")

                if runtime_demo_delay > 0:
                    time.sleep(runtime_demo_delay)
                continue

            if event["type"] == "final":
                turn_result = event["result"]

        if turn_result is None:
            raise RuntimeError("Agent 没有返回最终结果。")

        live_status_placeholder.success("本轮 LangGraph 执行完成。")
        live_graph_placeholder.graphviz_chart(
            build_runtime_graphviz_dot(traversed_nodes=turn_result.traversed_nodes),
            width="stretch",
        )

        st.session_state.turn_records.append(
            build_turn_record(
                question=cleaned_question,
                answer=turn_result.answer,
                scope_names=selected_paper_names.copy(),
                turn_result=turn_result,
            )
        )
        st.session_state.agent_graph_mermaid = turn_result.graph_mermaid
        st.session_state.runtime_focus_turn_index = len(st.session_state.turn_records) - 1
        st.session_state.pending_clear_question = True
        st.rerun()
    except Exception as exc:
        st.error("提问失败，请检查对话模型配置、工具调用能力和接口权限。")
        st.exception(exc)


def main() -> None:
    init_session_state()

    st.title("论文阅读助手")
    st.caption("这个版本已经升级为带工具节点和多轮记忆的 LangGraph agent，并把 runtime 学习区单独拆到了右侧。")

    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL")

    if not api_key:
        st.error("没有检测到 API_KEY。请先在 `.env` 文件里填入模型平台密钥。")
        st.stop()

    assistant = PaperAssistant(api_key=api_key, base_url=base_url)

    _, _, _, top_k, runtime_demo_delay = render_setup_panel(
        assistant=assistant,
        base_url=base_url,
    )

    paper_name_to_id, selected_paper_names = render_scope_selector()
    render_index_library(assistant)

    if st.session_state.pending_clear_question:
        st.session_state.question_input = ""
        st.session_state.pending_clear_question = False

    main_mount, runtime_mount = render_three_column_workbench(selected_paper_names)

    handle_agent_turn(
        assistant=assistant,
        paper_name_to_id=paper_name_to_id,
        selected_paper_names=selected_paper_names,
        top_k=top_k,
        runtime_demo_delay=runtime_demo_delay,
        main_mount=main_mount,
        runtime_mount=runtime_mount,
    )

    st.divider()
    st.markdown(
        """
        当前版本已经支持：
        1. 多篇论文共同建索引，并按默认范围做多轮问答。
        2. 用显式 `ToolNode` 构建 LangGraph agent，而不是把检索写死在一个函数里。
        3. 用 `InMemorySaver` 保存线程级记忆，让“继续说刚才那篇”这类追问真正能接上文。
        4. 在网页右侧实时展示流程图、节点日志、工具调用和检索证据，方便新手理解 runtime。
        """
    )


if __name__ == "__main__":
    main()
