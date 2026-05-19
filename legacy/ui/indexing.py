from __future__ import annotations

import hashlib

import streamlit as st

from paper_assistant import PaperAssistant


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
        raise ValueError("没有从已上传文件里解析出可用文本，请更换文件后再试。")

    chunk_vectors = assistant.embed_chunks(all_chunks)
    paper_overviews = assistant.list_papers(all_chunks)
    return all_chunks, chunk_vectors, paper_overviews, skipped_files


def get_chunks_for_paper(paper_id: str):
    return [chunk for chunk in st.session_state.chunks if chunk.paper_id == paper_id]
