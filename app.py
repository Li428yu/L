from __future__ import annotations

import hashlib
import os

from dotenv import load_dotenv
import streamlit as st

from paper_assistant import PaperAssistant, estimate_cost_hint


load_dotenv()

st.set_page_config(
    page_title="论文阅读助手",
    page_icon="📚",
    layout="wide",
)


def init_session_state() -> None:
    defaults = {
        "chunks": [],
        "chunk_vectors": None,
        "paper_overviews": [],
        "paper_digests": {},
        "index_signature": None,
        "last_answer": None,
        "last_evidence": [],
        "last_question": "",
        "question_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_index_state() -> None:
    st.session_state.chunks = []
    st.session_state.chunk_vectors = None
    st.session_state.paper_overviews = []
    st.session_state.paper_digests = {}
    st.session_state.index_signature = None
    st.session_state.last_answer = None
    st.session_state.last_evidence = []
    st.session_state.last_question = ""


def build_index_signature(
    file_payloads: list[dict[str, bytes | str]],
    chunk_size: int,
    overlap: int,
) -> str:
    digest = hashlib.sha1()
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(str(overlap).encode("utf-8"))

    for payload in file_payloads:
        name = str(payload["name"])
        content = payload["bytes"]
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, byteorder="little", signed=False))
        digest.update(hashlib.sha1(content).digest())

    return digest.hexdigest()


def build_index(
    assistant: PaperAssistant,
    file_payloads: list[dict[str, bytes | str]],
    chunk_size: int,
    overlap: int,
):
    all_chunks = []
    next_chunk_id = 0
    skipped_files: list[str] = []

    for index, payload in enumerate(file_payloads, start=1):
        file_name = str(payload["name"])
        file_bytes = payload["bytes"]
        pages = assistant.extract_text_from_file(file_name, file_bytes)

        if not pages:
            skipped_files.append(file_name)
            continue

        paper_id = f"paper-{index}"
        paper_chunks = assistant.chunk_pages(
            pages=pages,
            chunk_size=chunk_size,
            overlap=overlap,
            paper_id=paper_id,
            paper_name=file_name,
            start_chunk_id=next_chunk_id,
        )
        if not paper_chunks:
            skipped_files.append(file_name)
            continue

        all_chunks.extend(paper_chunks)
        next_chunk_id += len(paper_chunks)

    if not all_chunks:
        raise ValueError("没有从已上传文件中解析出可用文本，请更换文件后再试。")

    chunk_vectors = assistant.embed_chunks(all_chunks)
    paper_overviews = assistant.list_papers(all_chunks)
    return all_chunks, chunk_vectors, paper_overviews, skipped_files


def get_chunks_for_paper(paper_id: str):
    return [chunk for chunk in st.session_state.chunks if chunk.paper_id == paper_id]


init_session_state()

st.title("论文阅读助手")
st.caption("支持多篇论文一起检索、筛选问答，并为单篇论文生成摘要卡片。")

api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("API_BASE_URL")

if not api_key:
    st.error("没有检测到 API_KEY。请先在 `.env` 文件里填入模型平台密钥。")
    st.stop()

assistant = PaperAssistant(api_key=api_key, base_url=base_url)

with st.sidebar:
    st.subheader("参数")
    chunk_size = st.slider("Chunk Size", min_value=400, max_value=1400, value=900, step=100)
    overlap = st.slider("Overlap", min_value=50, max_value=400, value=180, step=10)
    top_k = st.slider("Top-K", min_value=2, max_value=8, value=4, step=1)
    st.markdown(
        """
        当前流程会分成三步：
        - 先把论文切成适合检索的小段
        - 再把这些片段变成向量索引
        - 提问时只取最相关的证据交给模型回答
        """
    )
    if base_url:
        st.caption(f"当前接口地址：`{base_url}`")

uploaded_files = st.file_uploader(
    "上传论文文件",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

file_payloads = [
    {
        "name": uploaded_file.name,
        "bytes": uploaded_file.getvalue(),
    }
    for uploaded_file in uploaded_files
]

pending_signature = (
    build_index_signature(file_payloads, chunk_size=chunk_size, overlap=overlap)
    if file_payloads
    else None
)
index_outdated = bool(file_payloads) and pending_signature != st.session_state.index_signature

top_left, top_right = st.columns([1.2, 1.0])

with top_left:
    if file_payloads:
        st.write(f"已选择 {len(file_payloads)} 篇文件")
        for payload in file_payloads:
            st.caption(f"- {payload['name']}")
    else:
        st.info("先上传一篇或多篇论文，然后建立索引。")

with top_right:
    build_label = "建立索引" if not st.session_state.index_signature else "更新索引"
    build_clicked = st.button(build_label, type="primary", disabled=not file_payloads)
    clear_clicked = st.button("清空当前索引", disabled=not st.session_state.chunks)

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
        st.session_state.last_answer = None
        st.session_state.last_evidence = []
        st.session_state.last_question = ""

        st.success(f"已完成：为 {len(paper_overviews)} 篇论文建立索引。")
        st.info(estimate_cost_hint(len(chunks)))
        if skipped_files:
            st.warning(f"这些文件没有解析出可用文本，已跳过：{', '.join(skipped_files)}")
    except Exception as exc:
        st.error("建立索引失败，请检查文件内容、模型配置和接口权限。")
        st.exception(exc)

if file_payloads and index_outdated:
    st.warning("你已经换了文件或调整了分块参数，当前问答仍基于上一次建立的索引。")

if st.session_state.paper_overviews:
    st.divider()
    st.subheader("当前索引")

    paper_count = len(st.session_state.paper_overviews)
    page_count = sum(item.page_count for item in st.session_state.paper_overviews)
    chunk_count = len(st.session_state.chunks)

    metric_cols = st.columns(3)
    metric_cols[0].metric("论文数", paper_count)
    metric_cols[1].metric("总页数", page_count)
    metric_cols[2].metric("Chunk 数", chunk_count)

    for overview in st.session_state.paper_overviews:
        with st.expander(
            f"{overview.paper_name} · {overview.page_count} 页 · {overview.chunk_count} 个片段",
            expanded=paper_count == 1,
        ):
            if st.button("生成摘要卡片", key=f"digest-{overview.paper_id}"):
                try:
                    with st.spinner(f"正在生成《{overview.paper_name}》的摘要卡片..."):
                        paper_chunks = get_chunks_for_paper(overview.paper_id)
                        digest = assistant.generate_paper_digest(
                            paper_name=overview.paper_name,
                            chunks=paper_chunks,
                        )
                    st.session_state.paper_digests[overview.paper_id] = digest
                except Exception as exc:
                    st.error("摘要卡片生成失败，请检查对话模型配置。")
                    st.exception(exc)

            digest = st.session_state.paper_digests.get(overview.paper_id)
            if digest:
                st.markdown(digest)
            else:
                st.caption("点击上方按钮后，这里会显示论文摘要卡片。")

    st.divider()
    st.subheader("开始提问")

    paper_name_to_id = {
        overview.paper_name: overview.paper_id for overview in st.session_state.paper_overviews
    }
    default_scope = list(paper_name_to_id.keys())
    selected_paper_names = st.multiselect(
        "检索范围",
        options=default_scope,
        default=default_scope,
        help="可以只勾选某几篇论文，也可以全部一起问。",
    )

    preset_questions = [
        ("一句话总结", "请用通俗的话概括这篇论文的核心思想。"),
        ("核心贡献", "这篇论文的核心贡献是什么？"),
        ("方法细节", "这篇论文的方法设计和关键模块是什么？"),
        ("实验结果", "这篇论文的实验设置、主要结果和结论是什么？"),
    ]
    if len(selected_paper_names) > 1:
        preset_questions.append(
            ("论文对比", "请对比这些论文的研究问题、方法设计和实验结论，有哪些关键差异？")
        )

    preset_cols = st.columns(len(preset_questions))
    for column, (label, prompt) in zip(preset_cols, preset_questions):
        if column.button(label, use_container_width=True):
            st.session_state.question_input = prompt

    question = st.text_input(
        "输入你的问题",
        key="question_input",
        placeholder="例如：这几篇论文在方法设计上最大的差异是什么？",
    )
    ask_clicked = st.button("开始提问", type="primary")

    if ask_clicked:
        cleaned_question = question.strip()
        selected_paper_ids = {
            paper_name_to_id[paper_name]
            for paper_name in selected_paper_names
            if paper_name in paper_name_to_id
        }

        if not cleaned_question:
            st.warning("先输入一个问题。")
        elif not selected_paper_ids:
            st.warning("至少选择一篇论文作为检索范围。")
        else:
            try:
                with st.spinner("正在检索相关内容并生成回答..."):
                    evidence = assistant.search(
                        query=cleaned_question,
                        chunks=st.session_state.chunks,
                        chunk_vectors=st.session_state.chunk_vectors,
                        top_k=top_k,
                        paper_ids=selected_paper_ids,
                    )
                    answer = assistant.answer_question(cleaned_question, evidence)

                st.session_state.last_question = cleaned_question
                st.session_state.last_answer = answer
                st.session_state.last_evidence = evidence
            except Exception as exc:
                st.error("提问失败，请检查对话模型配置和接口权限。")
                st.exception(exc)

    if st.session_state.last_answer:
        left_col, right_col = st.columns([1.4, 1.0])

        with left_col:
            st.subheader("回答")
            st.caption(f"问题：{st.session_state.last_question}")
            st.write(st.session_state.last_answer)

        with right_col:
            st.subheader("检索到的证据")
            for item in st.session_state.last_evidence:
                with st.expander(
                    f"{item.chunk.paper_name} | Page {item.chunk.page} | Score {item.score:.3f}"
                ):
                    st.write(item.chunk.text)

st.divider()
st.markdown(
    """
    现在这版已经支持：
    1. 多篇论文共同建索引和筛选问答
    2. 只在你确认后才重建索引，避免重复消耗 embedding
    3. 为单篇论文生成结构化摘要卡片

    下一步还可以继续补：
    1. 对答案里的引用做高亮和跳转
    2. 自动抽取摘要、方法、实验等章节
    3. 生成可导出的论文阅读笔记
    """
)
