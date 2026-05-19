from __future__ import annotations

import streamlit as st

from ui.runtime_view import format_runtime_path
from ui.state import get_latest_turn_record, get_runtime_focus, reset_conversation_state


def truncate_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def render_prompt_shortcuts(preset_questions: list[tuple[str, str]], disabled: bool) -> None:
    st.caption("快捷提问")
    for start_index in range(0, len(preset_questions), 3):
        row_items = preset_questions[start_index : start_index + 3]
        row_cols = st.columns(len(row_items))
        for col, (label, prompt) in zip(row_cols, row_items):
            if col.button(
                label,
                key=f"preset-{label}",
                width="stretch",
                disabled=disabled,
            ):
                st.session_state.question_input = prompt
                st.rerun()


def render_history_panel() -> None:
    focus_index, _ = get_runtime_focus()
    turns = st.session_state.turn_records
    history_turns = turns[:-1] if len(turns) > 1 else []

    st.subheader("历史对话")
    st.caption("左侧只负责回看前面的轮次，避免当前阅读区被长对话挤满。")

    if turns:
        st.caption(f"累计 {len(turns)} 轮，右侧当前查看第 {(focus_index or 0) + 1} 轮流程。")
        if st.button(
            "右侧切回当前轮",
            width="stretch",
            disabled=(focus_index == len(turns) - 1),
        ):
            st.session_state.runtime_focus_turn_index = len(turns) - 1
            st.rerun()

    if not history_turns:
        st.info("当前还没有可浏览的历史轮次。发起第二轮之后，这里会自动沉淀成时间线。")
        return

    for index in range(len(history_turns) - 1, -1, -1):
        turn = history_turns[index]
        turn_number = index + 1
        scope_text = "、".join(turn.get("scope_names", [])) or "当前全部论文"
        with st.expander(
            f"第 {turn_number} 轮 · {truncate_text(turn['question'], 26)}",
            expanded=focus_index == index,
        ):
            st.markdown(f"**问题**\n\n{turn['question']}")
            st.markdown(f"**回答摘要**\n\n{truncate_text(turn['answer'], 180)}")
            st.caption(f"默认检索范围：{scope_text}")
            st.caption(f"经过节点：{format_runtime_path(turn.get('traversed_nodes'))}")
            if st.button("在右侧查看这一轮流程", key=f"focus-turn-{index}", width="stretch"):
                st.session_state.runtime_focus_turn_index = index
                st.rerun()


def render_current_turn_panel(scope_names: list[str], chat_disabled: bool) -> None:
    latest_turn = get_latest_turn_record()
    header_cols = st.columns([1.75, 0.75])
    header_cols[0].subheader("当前对话")
    header_cols[0].caption("中间只保留当前这一轮的问答，前面的轮次请到左侧浏览。")
    if header_cols[1].button(
        "新建对话",
        width="stretch",
        disabled=not st.session_state.turn_records,
    ):
        reset_conversation_state()
        st.rerun()

    scope_text = "、".join(scope_names) if scope_names else "未选择默认范围"
    st.caption(f"这一轮默认检索范围：{scope_text}")

    if latest_turn is None:
        st.info("这里会显示当前轮的提问和回答。你可以先在上方建立索引，然后从下面发起第一轮提问。")
    else:
        st.caption(f"当前显示：第 {len(st.session_state.turn_records)} 轮")
        with st.chat_message("user"):
            st.markdown(latest_turn["question"])
        with st.chat_message("assistant"):
            st.markdown(latest_turn["answer"])

    preset_questions = [
        ("一句话总结", "请用通俗的话概括这篇论文的核心思想。"),
        ("核心贡献", "这篇论文最重要的贡献是什么？"),
        ("方法细节", "这篇论文的方法设计和关键模块是什么？"),
        ("实验结果", "这篇论文的实验设置、主要结果和结论是什么？"),
        ("生成摘要卡", "请为当前最重要的一篇论文生成结构化阅读卡片。"),
    ]
    if len(scope_names) > 1:
        preset_questions.append(
            ("论文对比", "请对比这些论文的研究问题、方法设计和实验结论，有哪些关键差异？")
        )

    render_prompt_shortcuts(preset_questions, disabled=chat_disabled)

    question = st.text_area(
        "输入这一轮想问的问题",
        key="question_input",
        height=120,
        placeholder="例如：继续比较刚才那两篇论文的方法差异。",
        disabled=chat_disabled,
    )
    if chat_disabled:
        st.caption("先上传论文并建立索引，才能开始问答。")

    ask_clicked = st.button(
        "发送给 Agent",
        type="primary",
        width="stretch",
        disabled=chat_disabled,
    )
    st.session_state["_latest_question_input_snapshot"] = question
    st.session_state["_ask_clicked"] = ask_clicked


def render_current_turn_preview(question: str) -> None:
    st.subheader("当前对话")
    st.caption("这一轮正在进行中。中间只保留当前问答，方便你专注看最终回答。")
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown("正在阅读记忆、决定是否调用工具，并组织最终回答……")
