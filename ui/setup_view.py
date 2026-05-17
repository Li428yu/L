from __future__ import annotations

import streamlit as st

from paper_assistant import PaperAssistant, estimate_cost_hint
from ui.indexing import build_index, build_index_signature, get_chunks_for_paper
from ui.state import clear_index_state, reset_conversation_state


def render_index_library(assistant: PaperAssistant) -> None:
    if not st.session_state.paper_overviews:
        return

    paper_count = len(st.session_state.paper_overviews)
    page_count = sum(item.page_count for item in st.session_state.paper_overviews)
    chunk_count = len(st.session_state.chunks)

    with st.expander("已建立的论文索引", expanded=False):
        metric_cols = st.columns(3)
        metric_cols[0].metric("论文数", paper_count)
        metric_cols[1].metric("总页数", page_count)
        metric_cols[2].metric("Chunk 数", chunk_count)
        st.caption(estimate_cost_hint(len(st.session_state.chunks)))

        for overview in st.session_state.paper_overviews:
            title = f"{overview.paper_name} · {overview.page_count} 页 · {overview.chunk_count} 个片段"
            with st.expander(title, expanded=paper_count == 1):
                if st.button("生成摘要卡", key=f"digest-{overview.paper_id}"):
                    try:
                        with st.spinner(f"正在生成《{overview.paper_name}》的结构化摘要卡..."):
                            digest = assistant.generate_paper_digest(
                                paper_name=overview.paper_name,
                                chunks=get_chunks_for_paper(overview.paper_id),
                            )
                        st.session_state.paper_digests[overview.paper_id] = digest
                    except Exception as exc:
                        st.error("摘要卡生成失败，请检查模型配置。")
                        st.exception(exc)

                digest = st.session_state.paper_digests.get(overview.paper_id)
                if digest:
                    st.markdown(digest)
                else:
                    st.caption("点击上方按钮后，这里会展示这篇论文的结构化阅读卡片。")


def render_setup_panel(
    assistant: PaperAssistant,
    base_url: str | None,
) -> tuple[list[dict[str, bytes | str]], int, int, int, float]:
    with st.expander("索引与运行设置", expanded=not st.session_state.paper_overviews):
        left_col, right_col = st.columns([1.25, 1.0], gap="large")

        with left_col:
            uploaded_files = st.file_uploader(
                "上传论文文件",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="支持一次上传一篇或多篇论文。建立索引后，Agent 才能检索和多轮记忆。",
            )

            file_payloads = [
                {
                    "name": uploaded_file.name,
                    "bytes": uploaded_file.getvalue(),
                }
                for uploaded_file in uploaded_files
            ]

            if file_payloads:
                st.write(f"已选择 {len(file_payloads)} 篇文件")
                for payload in file_payloads:
                    st.caption(f"- {payload['name']}")
            else:
                st.info("先上传一篇或多篇论文，然后建立索引。")

        with right_col:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=400,
                max_value=1400,
                value=900,
                step=100,
            )
            overlap = st.slider(
                "Overlap",
                min_value=50,
                max_value=400,
                value=180,
                step=10,
            )
            top_k = st.slider(
                "Top-K",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                help="每次检索默认取回多少条证据。",
            )
            runtime_demo_delay = st.select_slider(
                "节点演示速度",
                options=[0.0, 0.25, 0.6],
                value=0.25,
                format_func=lambda value: {
                    0.0: "即时",
                    0.25: "教学模式",
                    0.6: "慢速演示",
                }[value],
            )

            button_cols = st.columns(2)
            build_label = "建立索引" if not st.session_state.index_signature else "更新索引"
            build_clicked = button_cols[0].button(
                build_label,
                type="primary",
                width="stretch",
                disabled=not file_payloads,
            )
            clear_clicked = button_cols[1].button(
                "清空当前索引",
                width="stretch",
                disabled=not st.session_state.chunks,
            )

            st.markdown(
                """
                这一版页面把学习重点拆成三块：
                - 上方负责上传文件、建索引、设定检索范围
                - 中间只保留当前这一轮问答，降低阅读干扰
                - 右侧专门解释 LangGraph runtime 发生了什么
                """
            )
            if base_url:
                st.caption(f"当前接口地址：`{base_url}`")

        pending_signature = (
            build_index_signature(file_payloads, chunk_size=chunk_size, overlap=overlap)
            if file_payloads
            else None
        )
        index_outdated = bool(file_payloads) and pending_signature != st.session_state.index_signature

        if clear_clicked:
            clear_index_state()
            st.rerun()

        if build_clicked:
            try:
                with st.spinner("正在解析论文并建立索引..."):
                    chunks, chunk_vectors, paper_overviews, skipped_files = build_index(
                        assistant=assistant,
                        file_payloads=file_payloads,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )

                st.session_state.chunks = chunks
                st.session_state.chunk_vectors = chunk_vectors
                st.session_state.paper_overviews = paper_overviews
                st.session_state.paper_digests = {}
                st.session_state.index_signature = pending_signature
                st.session_state.selected_paper_names = [
                    overview.paper_name for overview in paper_overviews
                ]
                reset_conversation_state()

                st.success(f"已完成：为 {len(paper_overviews)} 篇论文建立索引。")
                if skipped_files:
                    st.warning(
                        f"这些文件没有解析出可用文本，已跳过：{', '.join(skipped_files)}"
                    )
            except Exception as exc:
                st.error("建立索引失败，请检查文件内容、模型配置和接口权限。")
                st.exception(exc)

        if file_payloads and index_outdated:
            st.warning("你已经更换了文件或调整了分块参数，当前问答仍然基于上一次建立的索引。")

    return file_payloads, chunk_size, overlap, top_k, runtime_demo_delay
