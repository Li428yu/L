from __future__ import annotations

from io import BytesIO
from typing import Iterable, Sequence

from docx import Document
import fitz
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from assistant_core.types import Chunk, PaperOverview


class PaperDocumentMixin:
    def extract_text_from_file(self, file_name: str, file_bytes: bytes) -> list[tuple[int, str]]:
        lowered_name = file_name.lower()
        if lowered_name.endswith(".pdf"):
            return self.extract_text(file_bytes)
        if lowered_name.endswith(".docx"):
            return self.extract_text_from_docx(file_bytes)
        raise ValueError(f"暂不支持的文件类型：{file_name}")

    def extract_text(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((index, text))
        return pages

    def extract_text_from_docx(self, docx_bytes: bytes) -> list[tuple[int, str]]:
        document = Document(BytesIO(docx_bytes))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
        full_text = "\n".join(text for text in paragraphs if text)
        if not full_text:
            return []
        return [(1, full_text)]

    def chunk_pages(
        self,
        pages: Iterable[tuple[int, str]],
        chunk_size: int = 900,
        overlap: int = 180,
        paper_id: str = "paper-1",
        paper_name: str = "未命名论文",
        start_chunk_id: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_id = start_chunk_id

        for page_number, text in pages:
            start = 0
            step = max(chunk_size - overlap, 1)
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            paper_id=paper_id,
                            paper_name=paper_name,
                            page=page_number,
                            text=chunk_text,
                        )
                    )
                    chunk_id += 1
                start += step

        return chunks

    def list_papers(self, chunks: Sequence[Chunk]) -> list[PaperOverview]:
        paper_stats: dict[str, PaperOverview] = {}

        for chunk in chunks:
            overview = paper_stats.get(chunk.paper_id)
            if overview is None:
                paper_stats[chunk.paper_id] = PaperOverview(
                    paper_id=chunk.paper_id,
                    paper_name=chunk.paper_name,
                    page_count=chunk.page,
                    chunk_count=1,
                )
                continue

            overview.page_count = max(overview.page_count, chunk.page)
            overview.chunk_count += 1

        return list(paper_stats.values())

    def generate_paper_digest(
        self,
        paper_name: str,
        chunks: Sequence[Chunk],
        max_chunks: int = 8,
    ) -> str:
        if not chunks:
            return "当前论文还没有可用内容，暂时无法生成摘要卡片。"

        representative_chunks = self._pick_representative_chunks(chunks, max_chunks=max_chunks)
        context = self._format_plain_chunks(representative_chunks)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful paper reading assistant. "
                        "Summarize only from the provided paper excerpts. "
                        "If some information is missing from the excerpts, say so clearly instead "
                        "of guessing. Respond in Chinese."
                    )
                ),
                HumanMessage(
                    content=(
                        f"请为论文《{paper_name}》生成一张结构化阅读卡片。\n"
                        "输出格式请严格包含以下小标题：\n"
                        "1. 一句话总结\n"
                        "2. 研究问题\n"
                        "3. 核心方法\n"
                        "4. 主要贡献\n"
                        "5. 实验与结果\n"
                        "6. 局限性\n"
                        "7. 推荐追问\n\n"
                        "如果原文片段没有覆盖某部分，请直接写“原文片段未覆盖”。\n\n"
                        f"论文片段：\n{context}"
                    )
                ),
            ]
        )
        return self._message_to_text(response)

    def _format_plain_chunks(self, chunks: Sequence[Chunk]) -> str:
        context_blocks = []
        for chunk in chunks:
            context_blocks.append(f"[{chunk.paper_name} | Page {chunk.page}]\n{chunk.text}")
        return "\n\n".join(context_blocks)

    def _pick_representative_chunks(
        self,
        chunks: Sequence[Chunk],
        max_chunks: int,
    ) -> list[Chunk]:
        if len(chunks) <= max_chunks:
            return list(chunks)
        if max_chunks <= 1:
            return [chunks[0]]

        positions = np.linspace(0, len(chunks) - 1, num=max_chunks, dtype=int)
        selected_chunks: list[Chunk] = []
        seen_positions: set[int] = set()
        for position in positions:
            resolved = int(position)
            if resolved in seen_positions:
                continue
            selected_chunks.append(chunks[resolved])
            seen_positions.add(resolved)
        return selected_chunks
