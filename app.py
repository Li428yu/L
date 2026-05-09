from __future__ import annotations

import os

from dotenv import load_dotenv
import streamlit as st

from paper_assistant import PaperAssistant, estimate_cost_hint


load_dotenv()

st.set_page_config(
    page_title="论文阅读助手",
    page_icon="📘",
    layout="wide",
)

st.title("论文阅读助手")
st.caption("上传一篇 PDF 或 Word 文档，系统会先建立检索，再基于原文回答问题。")

api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("API_BASE_URL")

if not api_key:
    st.error("未检测到 API_KEY。请先在 .env 文件里填写你的模型平台密钥。")
    st.stop()

assistant = PaperAssistant(api_key=api_key, base_url=base_url)

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chunk_vectors" not in st.session_state:
    st.session_state.chunk_vectors = None
if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

with st.sidebar:
    st.subheader("参数")
    chunk_size = st.slider("Chunk Size", min_value=400, max_value=1400, value=900, step=100)
    overlap = st.slider("Overlap", min_value=50, max_value=400, value=180, step=10)
    top_k = st.slider("Top-K", min_value=2, max_value=8, value=4, step=1)
    st.markdown(
        """
        你现在做的是最小版 RAG：
        - `切分`：把论文拆成小块
        - `向量化`：把文本变成可检索表示
        - `召回`：找最相关的几段
        - `生成`：让模型基于证据回答
        """
    )
    if base_url:
        st.caption(f"当前接口地址：`{base_url}`")

uploaded_file = st.file_uploader("上传论文文件", type=["pdf", "docx"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()

    try:
        with st.spinner("正在解析论文并建立检索..."):
            if file_name.endswith(".pdf"):
                pages = assistant.extract_text(file_bytes)
            else:
                pages = assistant.extract_text_from_docx(file_bytes)

            if not pages:
                st.error("没有从文件中解析出可用文本，请换一个文件试试。")
                st.stop()

            chunks = assistant.chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
            chunk_vectors = assistant.embed_chunks(chunks)

            st.session_state.chunks = chunks
            st.session_state.chunk_vectors = chunk_vectors
            st.session_state.paper_name = uploaded_file.name

        st.success(f"已完成：{uploaded_file.name}")
        st.info(estimate_cost_hint(len(chunks)))
    except Exception as exc:
        st.error("建立检索失败，请检查模型配置、接口地址和账户额度。")
        st.exception(exc)

if st.session_state.paper_name:
    st.write(f"当前论文：`{st.session_state.paper_name}`")

question = st.text_input(
    "输入你的问题",
    placeholder="例如：这篇论文的核心贡献是什么？实验部分是怎么设计的？",
)

if st.button("开始提问", type="primary", disabled=not st.session_state.chunks):
    if not question.strip():
        st.warning("先输入一个问题。")
    else:
        try:
            with st.spinner("正在检索相关内容并生成回答..."):
                evidence = assistant.search(
                    query=question,
                    chunks=st.session_state.chunks,
                    chunk_vectors=st.session_state.chunk_vectors,
                    top_k=top_k,
                )
                answer = assistant.answer_question(question, evidence)

            left_col, right_col = st.columns([1.4, 1.0])

            with left_col:
                st.subheader("回答")
                st.write(answer)

            with right_col:
                st.subheader("检索到的证据")
                for item in evidence:
                    with st.expander(f"Page {item.chunk.page} | Score {item.score:.3f}"):
                        st.write(item.chunk.text)
        except Exception as exc:
            st.error("提问失败，请检查对话模型配置和接口权限。")
            st.exception(exc)

st.divider()
st.markdown(
    """
    你下一步可以继续优化：
    1. 支持多篇论文同时检索
    2. 抽取摘要、方法、实验等章节
    3. 为回答增加引用高亮
    4. 做一个“论文总结卡片”
    """
)
