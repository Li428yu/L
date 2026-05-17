from __future__ import annotations

import uuid
from typing import Any

import streamlit as st


def new_thread_id() -> str:
    return uuid.uuid4().hex


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "chunks": [],
        "chunk_vectors": None,
        "paper_overviews": [],
        "paper_digests": {},
        "index_signature": None,
        "question_input": "",
        "selected_paper_names": None,
        "turn_records": [],
        "agent_graph_mermaid": "",
        "agent_thread_id": new_thread_id(),
        "runtime_focus_turn_index": None,
        "pending_clear_question": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation_state() -> None:
    st.session_state.question_input = ""
    st.session_state.turn_records = []
    st.session_state.agent_thread_id = new_thread_id()
    st.session_state.runtime_focus_turn_index = None
    st.session_state.pending_clear_question = False


def clear_index_state() -> None:
    st.session_state.chunks = []
    st.session_state.chunk_vectors = None
    st.session_state.paper_overviews = []
    st.session_state.paper_digests = {}
    st.session_state.index_signature = None
    st.session_state.selected_paper_names = None
    reset_conversation_state()


def sync_selected_paper_names() -> None:
    available_names = [overview.paper_name for overview in st.session_state.paper_overviews]
    current = st.session_state.selected_paper_names

    if not available_names:
        st.session_state.selected_paper_names = None
        return

    if current is None:
        st.session_state.selected_paper_names = available_names.copy()
        return

    valid_names = set(available_names)
    if any(name not in valid_names for name in current):
        filtered_names = [name for name in current if name in valid_names]
        st.session_state.selected_paper_names = filtered_names or available_names.copy()


def get_latest_turn_record() -> dict[str, Any] | None:
    if not st.session_state.turn_records:
        return None
    return st.session_state.turn_records[-1]


def get_runtime_focus() -> tuple[int | None, dict[str, Any] | None]:
    turns = st.session_state.turn_records
    if not turns:
        return None, None

    focus_index = st.session_state.runtime_focus_turn_index
    if focus_index is None or not 0 <= focus_index < len(turns):
        focus_index = len(turns) - 1
        st.session_state.runtime_focus_turn_index = focus_index

    return focus_index, turns[focus_index]


def build_turn_record(
    question: str,
    answer: str,
    scope_names: list[str],
    turn_result: Any,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "scope_names": scope_names,
        "tool_traces": turn_result.tool_traces,
        "evidence": turn_result.evidence,
        "runtime_node_logs": turn_result.runtime_node_logs,
        "traversed_nodes": turn_result.traversed_nodes,
        "graph_mermaid": turn_result.graph_mermaid,
    }
