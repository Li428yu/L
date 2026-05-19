from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
import fitz

from backend.app.models import ChunkStrategy


SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Abstract", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:abstract|摘要)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Introduction", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:introduction|引言|绪论)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Methods", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:methods?|methodology|方法|实验方法|研究方法)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Results", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:results?|结果|实验结果|研究结果)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Discussion", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:discussion|讨论|分析与讨论)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Limitations", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:limitations?|研究局限|局限性|局限与不足|研究不足|不足与展望|不足)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("FutureWork", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:future\s+work|future\s+research|未来研究|后续研究|研究展望|展望)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("Conclusion", re.compile(r"^\s*(?:(?:第?[一二三四五六七八九十]+[章节、.．]|\d+(?:\.\d+)*[.、．]?)\s*)?(?:conclusions?|结论与展望|结论|总结)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
    ("References", re.compile(r"^\s*(?:references|参考文献)\s*(?:[:：]|\s|$)", re.IGNORECASE)),
]


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    paper_name: str
    page: int
    section: str
    source: str
    file_hash: str
    text: str
    char_start: int
    char_end: int
    quote: str


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def safe_upload_name(file_hash: str, file_name: str) -> str:
    suffix = Path(file_name).suffix.lower() or ".pdf"
    return f"{file_hash[:16]}{suffix}"


def extract_pdf_text(pdf_path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(PageText(page=index, text=text))
    return pages


def extract_docx_text(docx_path: Path) -> list[PageText]:
    document = Document(docx_path)
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    pages: list[PageText] = []
    if not blocks:
        return pages

    buffer: list[str] = []
    page_number = 1
    char_count = 0
    for block in blocks:
        if buffer and char_count + len(block) > 2200:
            pages.append(PageText(page=page_number, text="\n\n".join(buffer)))
            page_number += 1
            buffer = []
            char_count = 0
        buffer.append(block)
        char_count += len(block)

    if buffer:
        pages.append(PageText(page=page_number, text="\n\n".join(buffer)))

    return pages


def extract_document_text(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".docx":
        return extract_docx_text(path)
    raise ValueError("暂不支持这个文件类型。请上传 PDF 或 DOCX。")


def analyze_pages(pages: list[PageText]) -> dict[str, int | str | float]:
    text = "\n".join(page.text for page in pages if page.text)
    char_count = len(text)
    paragraph_count = len([p for p in re.split(r"\n\s*\n|(?<=。)\s+|(?<=\.)\s+", text) if p.strip()])
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    formula_hits = len(re.findall(r"[\u03b1-\u03c9\u0391-\u03a9=+\-*/<>≤≥∑∫√]|p\s*[<=>]", text))
    table_hits = len(re.findall(r"\b(table|表)\s*\d+|\|", text, flags=re.IGNORECASE))
    page_count = len(pages)

    if cjk_chars > latin_chars * 1.5:
        language = "zh"
    elif latin_chars > cjk_chars * 1.5:
        language = "en"
    else:
        language = "mixed"

    return {
        "page_count": page_count,
        "char_count": char_count,
        "paragraph_count": paragraph_count,
        "language": language,
        "formula_table_ratio": (formula_hits + table_hits * 2) / max(char_count, 1),
        "avg_chars_per_page": char_count / max(page_count, 1),
    }


def choose_chunk_strategy(pages: list[PageText]) -> ChunkStrategy:
    stats = analyze_pages(pages)
    page_count = int(stats["page_count"])
    char_count = int(stats["char_count"])
    paragraph_count = int(stats["paragraph_count"])
    language = str(stats["language"])
    formula_table_ratio = float(stats["formula_table_ratio"])
    avg_chars_per_page = float(stats["avg_chars_per_page"])

    chunk_size = 900
    overlap = 160
    splitter = "paragraph-page-hybrid"
    reasons: list[str] = []

    if page_count >= 35 or char_count >= 90_000:
        chunk_size = 1200
        overlap = 220
        reasons.append("文档较长，使用更大的 chunk 减少跨页主题被切散。")
    elif page_count <= 8 and char_count <= 25_000:
        chunk_size = 700
        overlap = 120
        reasons.append("文档较短，使用较小 chunk 提高定位精度。")

    density = paragraph_count / max(page_count, 1)
    if density >= 18:
        chunk_size = max(650, chunk_size - 150)
        overlap = max(100, overlap - 30)
        reasons.append("段落密度高，降低 chunk 大小以避免一个片段混入过多论点。")
    elif density <= 5 and avg_chars_per_page > 1500:
        chunk_size += 150
        overlap += 40
        reasons.append("段落偏长，增加 overlap 保留上下文连续性。")

    if language == "zh":
        chunk_size = int(chunk_size * 0.9)
        overlap = int(overlap * 0.9)
        reasons.append("中文字符信息密度较高，适当缩小 chunk。")
    elif language == "mixed":
        reasons.append("检测到中英文混合，采用页面和段落混合切分。")

    if formula_table_ratio > 0.012:
        splitter = "layout-aware-page-hybrid"
        overlap = max(overlap, 220)
        reasons.append("公式或表格比例较高，优先沿页面/段落边界切分并增加 overlap。")

    if not reasons:
        reasons.append("文档长度和段落密度适中，使用默认论文阅读切分策略。")

    return ChunkStrategy(
        chunk_size=chunk_size,
        overlap=overlap,
        splitter=splitter,
        language=language,
        page_count=page_count,
        paragraph_count=paragraph_count,
        char_count=char_count,
        reasons=reasons,
    )


def split_pages_into_chunks(
    *,
    pages: list[PageText],
    document_id: str,
    paper_name: str,
    file_hash: str,
    source: str,
    strategy: ChunkStrategy,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    index = 0
    current_section = "Unknown"

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        current_section = detect_section(page_text, current_section)
        paragraphs = split_paragraphs(page_text)
        buffer = ""
        buffer_start = 0
        cursor = 0

        for paragraph in paragraphs:
            paragraph_start = page_text.find(paragraph, cursor)
            if paragraph_start < 0:
                paragraph_start = cursor
            paragraph_end = paragraph_start + len(paragraph)
            cursor = paragraph_end
            paragraph_section = detect_section(paragraph, current_section)
            if paragraph_section != current_section and buffer:
                index = _append_chunk(
                    chunks=chunks,
                    index=index,
                    document_id=document_id,
                    paper_name=paper_name,
                    page=page.page,
                    section=current_section,
                    source=source,
                    file_hash=file_hash,
                    text=buffer,
                    char_start=buffer_start,
                    char_end=buffer_start + len(buffer),
                )
                buffer = ""
                buffer_start = paragraph_start
            current_section = paragraph_section

            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= strategy.chunk_size:
                if not buffer:
                    buffer_start = paragraph_start
                buffer = candidate
                continue

            if buffer:
                index = _append_chunk(
                    chunks=chunks,
                    index=index,
                    document_id=document_id,
                    paper_name=paper_name,
                    page=page.page,
                    section=current_section,
                    source=source,
                    file_hash=file_hash,
                    text=buffer,
                    char_start=buffer_start,
                    char_end=buffer_start + len(buffer),
                )
                buffer = ""
                buffer_start = paragraph_start

            if len(paragraph) > strategy.chunk_size:
                for piece, start_offset in split_long_text(paragraph, strategy.chunk_size, strategy.overlap):
                    index = _append_chunk(
                        chunks=chunks,
                        index=index,
                        document_id=document_id,
                        paper_name=paper_name,
                        page=page.page,
                        section=current_section,
                        source=source,
                        file_hash=file_hash,
                        text=piece,
                        char_start=paragraph_start + start_offset,
                        char_end=paragraph_start + start_offset + len(piece),
                    )
                buffer = ""
                buffer_start = paragraph_end
            else:
                buffer = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph

        if buffer:
            index = _append_chunk(
                chunks=chunks,
                index=index,
                document_id=document_id,
                paper_name=paper_name,
                page=page.page,
                section=current_section,
                source=source,
                file_hash=file_hash,
                text=buffer,
                char_start=buffer_start,
                char_end=buffer_start + len(buffer),
            )

    return chunks


def split_paragraphs(text: str) -> list[str]:
    rough = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for item in rough:
        cleaned = " ".join(line.strip() for line in item.splitlines() if line.strip())
        if cleaned:
            paragraphs.append(cleaned)
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[。.!?])\s+", text) if part.strip()]
    return paragraphs or [text]


def detect_section(text: str, fallback: str) -> str:
    first_lines = [line.strip() for line in text.splitlines()[:10] if line.strip()]
    for line in first_lines:
        for name, pattern in SECTION_PATTERNS:
            if pattern.search(line):
                return name
    return fallback


def tail_overlap(text: str, overlap: int) -> str:
    normalized = text.strip()
    if len(normalized) <= overlap:
        return normalized
    return normalized[-overlap:].strip()


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    sentences = split_sentences(text)
    if len(sentences) > 1:
        pieces: list[tuple[str, int]] = []
        buffer = ""
        buffer_start = 0
        cursor = 0
        for sentence in sentences:
            sentence_start = text.find(sentence, cursor)
            if sentence_start < 0:
                sentence_start = cursor
            cursor = sentence_start + len(sentence)
            candidate = f"{buffer}{sentence}" if buffer else sentence
            if len(candidate) <= chunk_size:
                if not buffer:
                    buffer_start = sentence_start
                buffer = candidate
                continue
            if buffer:
                pieces.append((buffer.strip(), buffer_start))
            buffer = sentence
            buffer_start = sentence_start
        if buffer:
            pieces.append((buffer.strip(), buffer_start))
        if pieces:
            return pieces

    pieces: list[tuple[str, int]] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            pieces.append((piece, start))
        start += step
    return pieces


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?；;])", text)
    return [part.strip() for part in parts if part.strip()]


def _append_chunk(
    *,
    chunks: list[ChunkRecord],
    index: int,
    document_id: str,
    paper_name: str,
    page: int,
    section: str,
    source: str,
    file_hash: str,
    text: str,
    char_start: int,
    char_end: int,
) -> int:
    normalized = " ".join(text.split())
    if len(normalized) < 40:
        return index
    chunk_id = f"{document_id}_chunk_{index:05d}"
    chunks.append(
        ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            paper_name=paper_name,
            page=page,
            section=section,
            source=source,
            file_hash=file_hash,
            text=normalized,
            char_start=char_start,
            char_end=char_end,
            quote=normalized[:240],
        )
    )
    return index + 1
