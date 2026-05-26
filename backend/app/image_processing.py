from __future__ import annotations

import hashlib
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections.abc import Callable
from typing import Any
import xml.etree.ElementTree as ET

import fitz

from backend.app.document_processing import ChunkRecord, count_tokens


CAPTION_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?\.?|table)\s*\d*|[图表]\s*[\d１-９一二三四五六七八九十]+",
    re.IGNORECASE,
)

DOCX_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
RELATIONSHIP_NAMESPACE = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass
class ExtractedImage:
    id: str
    document_id: str
    image_hash: str
    page_start: int
    page_end: int
    bbox: tuple[float, float, float, float]
    image_path: str
    thumbnail_path: str
    width: int
    height: int
    kind: str
    ocr_text: str
    vision_summary: str
    caption_text: str
    status: str
    ocr_status: str = ""
    ocr_error: str = ""
    vision_error: str = ""
    bbox_json_override: str = ""

    @property
    def bbox_json(self) -> str:
        if self.bbox_json_override:
            return self.bbox_json_override
        return json.dumps(
            {
                "x0": self.bbox[0],
                "y0": self.bbox[1],
                "x1": self.bbox[2],
                "y1": self.bbox[3],
            },
            ensure_ascii=False,
        )

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "image_hash": self.image_hash,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bbox_json": self.bbox_json,
            "image_path": self.image_path,
            "thumbnail_path": self.thumbnail_path,
            "width": self.width,
            "height": self.height,
            "kind": self.kind,
            "ocr_text": self.ocr_text,
            "ocr_status": self.ocr_status,
            "ocr_error": self.ocr_error,
            "vision_summary": self.vision_summary,
            "caption_text": self.caption_text,
            "status": self.status,
            "vision_error": self.vision_error,
        }


@dataclass(frozen=True)
class OCRResult:
    text: str
    status: str
    error: str = ""


def extract_pdf_images(
    *,
    pdf_path: Path,
    output_dir: Path,
    document_id: str,
    max_images: int = 300,
    max_ocr_images: int = 40,
) -> list[ExtractedImage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    ocr_cache: dict[str, OCRResult] = {}
    page_heights: dict[int, float] = {}
    image_counter = 0

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_heights[page_index] = float(page.rect.height)
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
            for block in blocks:
                if block.get("type") != 1 or image_counter >= max_images:
                    continue
                bbox = _safe_bbox(block.get("bbox"))
                if bbox is None or _should_skip_image_bbox(bbox, page.rect):
                    continue

                rendered = _render_page_clip(page, bbox)
                if not rendered:
                    continue
                image_hash = hashlib.sha256(rendered).hexdigest()
                ext = "png"
                image_path = output_dir / f"p{page_index:04d}_{image_counter:04d}_{image_hash[:12]}.{ext}"
                if not image_path.exists():
                    image_path.write_bytes(rendered)

                caption_text = _caption_near_bbox(blocks, bbox)
                ocr_result = _ocr_pdf_image_with_status(
                    image_hash=image_hash,
                    image_path=image_path,
                    bbox=bbox,
                    page_rect=page.rect,
                    caption_text=caption_text,
                    max_ocr_images=max_ocr_images,
                    ocr_cache=ocr_cache,
                )
                ocr_text = ocr_result.text

                width = int(abs(bbox[2] - bbox[0]))
                height = int(abs(bbox[3] - bbox[1]))
                kind = _classify_image(caption_text=caption_text, ocr_text=ocr_text)
                vision_summary = _fallback_vision_summary(
                    page=page_index,
                    caption_text=caption_text,
                    ocr_text=ocr_text,
                    kind=kind,
                )
                status = "ready" if caption_text or ocr_text else "stored_needs_ocr"
                image = ExtractedImage(
                    id=f"{document_id}_image_{image_counter:05d}",
                    document_id=document_id,
                    image_hash=image_hash,
                    page_start=page_index,
                    page_end=page_index,
                    bbox=bbox,
                    image_path=str(image_path),
                    thumbnail_path=_make_thumbnail(image_path),
                    width=width,
                    height=height,
                    kind=kind,
                    ocr_text=ocr_text,
                    ocr_status=ocr_result.status,
                    ocr_error=ocr_result.error,
                    vision_summary=vision_summary,
                    caption_text=caption_text,
                    status=status,
                )
                images.append(image)
                image_counter += 1
            if image_counter >= max_images:
                break

    images.extend(_cross_page_image_records(images, page_heights, output_dir, document_id))
    return images


def extract_docx_images(
    *,
    docx_path: Path,
    output_dir: Path,
    document_id: str,
    max_images: int = 300,
    max_ocr_images: int = 40,
) -> list[ExtractedImage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    ocr_cache: dict[str, OCRResult] = {}
    image_counter = 0

    with zipfile.ZipFile(docx_path) as package:
        for part_name in _docx_content_parts(package):
            if image_counter >= max_images:
                break
            relationships = _docx_relationships(package, part_name)
            if not relationships:
                continue
            records = _docx_paragraph_records(package, part_name, relationships)
            for record_index, record in enumerate(records):
                if image_counter >= max_images:
                    break
                for image_ref in record["image_refs"]:
                    if image_counter >= max_images:
                        break
                    image_bytes = _docx_image_bytes(package, image_ref["path"])
                    if not image_bytes:
                        continue
                    saved = _save_docx_image_as_png(
                        image_bytes=image_bytes,
                        output_dir=output_dir,
                        page=int(record["page"]),
                        image_index=image_counter,
                    )
                    if saved is None:
                        continue
                    image_path, width, height, rendered = saved
                    image_hash = hashlib.sha256(rendered).hexdigest()
                    final_path = output_dir / f"docx_p{int(record['page']):04d}_{image_counter:04d}_{image_hash[:12]}.png"
                    if image_path != final_path:
                        if final_path.exists():
                            image_path.unlink(missing_ok=True)
                        else:
                            image_path.replace(final_path)
                        image_path = final_path

                    caption_text = _caption_near_docx_record(records, record_index)
                    ocr_result = _ocr_docx_image_with_status(
                        image_hash=image_hash,
                        image_path=image_path,
                        width=width,
                        height=height,
                        caption_text=caption_text,
                        max_ocr_images=max_ocr_images,
                        ocr_cache=ocr_cache,
                    )
                    ocr_text = ocr_result.text

                    kind = _classify_image(caption_text=caption_text, ocr_text=ocr_text)
                    vision_summary = _fallback_vision_summary(
                        page=int(record["page"]),
                        caption_text=caption_text,
                        ocr_text=ocr_text,
                        kind=kind,
                    )
                    status = "ready" if caption_text or ocr_text else "stored_needs_ocr"
                    bbox_json = json.dumps(
                        {
                            "docx_part": part_name,
                            "paragraph_index": record_index,
                            "relationship_id": image_ref["relationship_id"],
                            "target": image_ref["path"],
                            "logical_page": int(record["page"]),
                        },
                        ensure_ascii=False,
                    )
                    image = ExtractedImage(
                        id=f"{document_id}_image_{image_counter:05d}",
                        document_id=document_id,
                        image_hash=image_hash,
                        page_start=int(record["page"]),
                        page_end=int(record["page"]),
                        bbox=(0.0, 0.0, float(width), float(height)),
                        image_path=str(image_path),
                        thumbnail_path=_make_thumbnail(image_path),
                        width=width,
                        height=height,
                        kind=kind,
                        ocr_text=ocr_text,
                        ocr_status=ocr_result.status,
                        ocr_error=ocr_result.error,
                        vision_summary=vision_summary,
                        caption_text=caption_text,
                        status=status,
                        bbox_json_override=bbox_json,
                    )
                    images.append(image)
                    image_counter += 1
    return images


def image_records_to_chunks(
    *,
    images: list[ExtractedImage],
    document_id: str,
    paper_name: str,
    source: str,
    file_hash: str,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for index, image in enumerate(images):
        text = _image_text_for_retrieval(image)
        normalized = " ".join(text.split())
        if len(normalized) < 24:
            continue
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document_id}_image_chunk_{index:05d}",
                document_id=document_id,
                paper_name=paper_name,
                page=image.page_start,
                section="Image",
                source=source,
                file_hash=file_hash,
                text=normalized,
                char_start=0,
                char_end=len(normalized),
                quote=normalized[:280],
                page_start=image.page_start,
                page_end=image.page_end,
                token_count=count_tokens(normalized),
                chunk_type=image.kind,
                image_id=image.id,
                image_path=image.image_path,
                bbox_json=image.bbox_json,
            )
        )
    return chunks


def _docx_content_parts(package: zipfile.ZipFile) -> list[str]:
    names = set(package.namelist())
    parts = ["word/document.xml"]
    parts.extend(sorted(name for name in names if re.fullmatch(r"word/header\d+\.xml", name)))
    parts.extend(sorted(name for name in names if re.fullmatch(r"word/footer\d+\.xml", name)))
    return [part for part in parts if part in names]


def _docx_relationships(package: zipfile.ZipFile, part_name: str) -> dict[str, str]:
    rels_name = _docx_rels_name(part_name)
    if rels_name not in package.namelist():
        return {}
    try:
        root = ET.fromstring(package.read(rels_name))
    except ET.ParseError:
        return {}
    relationships: dict[str, str] = {}
    part_dir = posixpath.dirname(part_name)
    for relationship in root.findall(f"{RELATIONSHIP_NAMESPACE}Relationship"):
        relationship_id = str(relationship.attrib.get("Id", ""))
        target = str(relationship.attrib.get("Target", ""))
        rel_type = str(relationship.attrib.get("Type", ""))
        if not relationship_id or not target or "image" not in rel_type.lower():
            continue
        relationships[relationship_id] = _resolve_docx_target(part_dir, target)
    return relationships


def _docx_rels_name(part_name: str) -> str:
    part_dir = posixpath.dirname(part_name)
    part_base = posixpath.basename(part_name)
    return posixpath.join(part_dir, "_rels", f"{part_base}.rels")


def _resolve_docx_target(part_dir: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(part_dir, target))


def _docx_paragraph_records(
    package: zipfile.ZipFile,
    part_name: str,
    relationships: dict[str, str],
) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(package.read(part_name))
    except ET.ParseError:
        return []

    records: list[dict[str, Any]] = []
    current_page = 1
    page_tokens = 0
    for paragraph in root.findall(".//w:p", DOCX_NAMESPACES):
        text = _docx_paragraph_text(paragraph)
        image_refs = _docx_paragraph_image_refs(paragraph, relationships)
        token_count = count_tokens(text) if text else 0
        if text and page_tokens and page_tokens + token_count > 1200:
            current_page += 1
            page_tokens = 0
        records.append(
            {
                "text": text,
                "image_refs": image_refs,
                "page": current_page,
            }
        )
        page_tokens += token_count
    return records


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    parts = [
        item.text or ""
        for item in paragraph.findall(".//w:t", DOCX_NAMESPACES)
        if item.text
    ]
    return " ".join("".join(parts).split())[:1200]


def _docx_paragraph_image_refs(
    paragraph: ET.Element,
    relationships: dict[str, str],
) -> list[dict[str, str]]:
    relationship_ids: list[str] = []
    for blip in paragraph.findall(".//a:blip", DOCX_NAMESPACES):
        relationship_id = blip.attrib.get(f"{{{DOCX_NAMESPACES['r']}}}embed") or blip.attrib.get(
            f"{{{DOCX_NAMESPACES['r']}}}link"
        )
        if relationship_id and relationship_id not in relationship_ids:
            relationship_ids.append(relationship_id)
    for image_data in paragraph.findall(".//v:imagedata", DOCX_NAMESPACES):
        relationship_id = image_data.attrib.get(f"{{{DOCX_NAMESPACES['r']}}}id")
        if relationship_id and relationship_id not in relationship_ids:
            relationship_ids.append(relationship_id)

    refs: list[dict[str, str]] = []
    for relationship_id in relationship_ids:
        target = relationships.get(relationship_id)
        if not target:
            continue
        refs.append({"relationship_id": relationship_id, "path": target})
    return refs


def _docx_image_bytes(package: zipfile.ZipFile, image_path: str) -> bytes:
    if image_path not in package.namelist():
        return b""
    try:
        return package.read(image_path)
    except Exception:
        return b""


def _save_docx_image_as_png(
    *,
    image_bytes: bytes,
    output_dir: Path,
    page: int,
    image_index: int,
) -> tuple[Path, int, int, bytes] | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            normalized = image.convert("RGBA") if image.mode in {"P", "LA", "RGBA"} else image.convert("RGB")
            output_path = output_dir / f"docx_p{page:04d}_{image_index:04d}_pending.png"
            normalized.save(output_path, format="PNG")
            rendered = output_path.read_bytes()
            return output_path, int(normalized.width), int(normalized.height), rendered
    except Exception:
        return None


def _caption_near_docx_record(records: list[dict[str, Any]], record_index: int) -> str:
    candidates: list[str] = []
    search_indexes = [record_index, record_index + 1, record_index - 1, record_index + 2, record_index - 2]
    for index in search_indexes:
        if index < 0 or index >= len(records):
            continue
        text = str(records[index].get("text", "")).strip()
        if text and CAPTION_PATTERN.search(text):
            candidates.append(text)
    if candidates:
        return " ".join(dict.fromkeys(candidates))[:900]

    nearby: list[str] = []
    for index in [record_index - 1, record_index, record_index + 1]:
        if index < 0 or index >= len(records):
            continue
        text = str(records[index].get("text", "")).strip()
        if text:
            nearby.append(text)
    if nearby:
        return f"Nearby text: {' '.join(nearby)[:760]}"
    return ""


def _should_ocr_docx_image(*, width: int, height: int, caption_text: str) -> bool:
    if caption_text:
        return True
    return width >= 360 and height >= 180


def enrich_images_with_vision(
    *,
    images: list[ExtractedImage],
    analyze_image: Callable[[Path, str], str],
    max_images: int = 40,
) -> list[ExtractedImage]:
    analyzed = 0
    for image in images:
        if analyzed >= max_images:
            if _should_analyze_image_with_vision(image):
                image.status = "vision_skipped"
                image.vision_error = "max_images_limit"
            continue
        if not _should_analyze_image_with_vision(image):
            image.status = "vision_skipped"
            image.vision_error = "not_selected_by_policy"
            continue
        path = Path(image.image_path)
        if not path.exists():
            image.status = "vision_failed"
            image.vision_error = "image_file_missing"
            continue
        prompt = _vision_prompt_for_image(image)
        try:
            summary = analyze_image(path, prompt)
        except Exception as exc:
            image.status = "vision_failed"
            image.vision_error = _vision_error_summary(exc)
            continue
        summary = _clean_vision_summary(summary)
        if not summary:
            image.status = "vision_failed"
            image.vision_error = "empty_vision_summary"
            continue
        image.vision_summary = summary
        image.status = "vision_ready"
        image.vision_error = ""
        image.kind = _classify_image(caption_text=image.caption_text, ocr_text=f"{image.ocr_text} {summary}")
        analyzed += 1
    return images


def vision_status_counts(images: list[ExtractedImage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image in images:
        status = image.status or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def ocr_status_counts(images: list[ExtractedImage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image in images:
        status = image.ocr_status or ("ocr_ready" if image.ocr_text.strip() else "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _vision_error_summary(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = re.sub(r"\s+", " ", message)
    return f"{exc.__class__.__name__}: {message}"[:300]


def _safe_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _should_analyze_image_with_vision(image: ExtractedImage) -> bool:
    if not image.image_path:
        return False
    if image.kind in {"figure_image", "chart_image", "table_image", "cross_page_image"}:
        return True
    if image.status == "stored_needs_ocr":
        return True
    if image.caption_text and not image.ocr_text:
        return True
    return False


def _vision_prompt_for_image(image: ExtractedImage) -> str:
    return f"""
请基于图片、题注和 OCR 输出简洁中文说明，不猜测图外信息。

页码：{image.page_start}-{image.page_end}
类型：{image.kind}
题注：{image.caption_text or "无"}
OCR：{image.ocr_text or "无"}

输出三点：实际内容；图表的轴/表头/变量和主要趋势；可支持的论文事实。不清楚就写不确定。
""".strip()


def _clean_vision_summary(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned[:2200]


def _should_skip_image_bbox(bbox: tuple[float, float, float, float], page_rect: fitz.Rect) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    page_area = float(page_rect.width * page_rect.height) or 1.0
    if width < 48 or height < 48:
        return True
    if area / page_area < 0.006:
        return True
    return False


def _render_page_clip(page: fitz.Page, bbox: tuple[float, float, float, float]) -> bytes:
    rect = fitz.Rect(*bbox) & page.rect
    if rect.is_empty:
        return b""
    max_side = max(rect.width, rect.height)
    scale = 2.0 if max_side < 1200 else 1.2
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    return pixmap.tobytes("png")


def _caption_near_bbox(blocks: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> str:
    candidates: list[tuple[float, str]] = []
    x0, y0, x1, y1 = bbox
    for block in blocks:
        if block.get("type") != 0:
            continue
        text = _text_from_block(block)
        if not text or not CAPTION_PATTERN.search(text):
            continue
        block_bbox = _safe_bbox(block.get("bbox"))
        if block_bbox is None:
            continue
        bx0, by0, bx1, by1 = block_bbox
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0)) / max(min(x1 - x0, bx1 - bx0), 1.0)
        vertical_gap = min(abs(by0 - y1), abs(y0 - by1))
        if horizontal_overlap >= 0.25 and vertical_gap <= 120:
            candidates.append((vertical_gap, text))
    candidates.sort(key=lambda item: item[0])
    return " ".join(text for _, text in candidates[:2])[:600]


def _text_from_block(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        line_text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
        if line_text.strip():
            parts.append(line_text.strip())
    return " ".join(parts)


def _should_ocr_image(
    bbox: tuple[float, float, float, float],
    page_rect: fitz.Rect,
    caption_text: str,
) -> bool:
    area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(float(page_rect.width * page_rect.height), 1.0)
    return bool(caption_text) or area_ratio >= 0.08


def _ocr_pdf_image_with_status(
    *,
    image_hash: str,
    image_path: Path,
    bbox: tuple[float, float, float, float],
    page_rect: fitz.Rect,
    caption_text: str,
    max_ocr_images: int,
    ocr_cache: dict[str, OCRResult],
) -> OCRResult:
    if image_hash in ocr_cache:
        return ocr_cache[image_hash]
    if not _should_ocr_image(bbox, page_rect, caption_text):
        return OCRResult(text="", status="ocr_skipped", error="not_selected_by_policy")
    if len(ocr_cache) >= max_ocr_images:
        return OCRResult(text="", status="ocr_skipped", error="max_images_limit")
    result = _ocr_image_with_status(image_path)
    ocr_cache[image_hash] = result
    return result


def _ocr_docx_image_with_status(
    *,
    image_hash: str,
    image_path: Path,
    width: int,
    height: int,
    caption_text: str,
    max_ocr_images: int,
    ocr_cache: dict[str, OCRResult],
) -> OCRResult:
    if image_hash in ocr_cache:
        return ocr_cache[image_hash]
    if not _should_ocr_docx_image(width=width, height=height, caption_text=caption_text):
        return OCRResult(text="", status="ocr_skipped", error="not_selected_by_policy")
    if len(ocr_cache) >= max_ocr_images:
        return OCRResult(text="", status="ocr_skipped", error="max_images_limit")
    result = _ocr_image_with_status(image_path)
    ocr_cache[image_hash] = result
    return result


def _ocr_image_with_status(image_path: Path) -> OCRResult:
    try:
        from PIL import Image
        import pytesseract
    except Exception as exc:
        return OCRResult(text="", status="ocr_failed", error=_ocr_error_summary(exc))
    first_error: Exception | None = None
    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    except Exception as exc:
        first_error = exc
        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image, lang="eng")
        except Exception as fallback_exc:
            return OCRResult(
                text="",
                status="ocr_failed",
                error=_ocr_error_summary(fallback_exc, first_error=first_error),
            )
    text = " ".join(str(text).split())[:3000]
    if not text:
        return OCRResult(text="", status="ocr_empty", error="empty_ocr_text")
    return OCRResult(text=text, status="ocr_ready", error="")


def _ocr_image(image_path: Path) -> str:
    return _ocr_image_with_status(image_path).text


def _ocr_error_summary(exc: Exception, *, first_error: Exception | None = None) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = re.sub(r"\s+", " ", message)
    if first_error is not None:
        first_message = str(first_error).strip() or first_error.__class__.__name__
        first_message = re.sub(r"\s+", " ", first_message)
        message = f"primary={first_error.__class__.__name__}: {first_message}; fallback={exc.__class__.__name__}: {message}"
    else:
        message = f"{exc.__class__.__name__}: {message}"
    return message[:300]


def _combine_ocr_status(images: list[ExtractedImage]) -> str:
    if any(image.ocr_text.strip() for image in images):
        return "ocr_ready"
    statuses = [image.ocr_status for image in images if image.ocr_status]
    if not statuses:
        return ""
    for status in ["ocr_failed", "ocr_empty", "ocr_skipped", "ocr_ready"]:
        if status in statuses:
            return status
    return statuses[0]


def _combine_ocr_error(images: list[ExtractedImage]) -> str:
    errors = [image.ocr_error for image in images if image.ocr_error]
    return "; ".join(dict.fromkeys(errors))[:300]


def _make_thumbnail(image_path: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    try:
        target = image_path.with_name(f"{image_path.stem}_thumb.png")
        with Image.open(image_path) as image:
            image.thumbnail((360, 360))
            image.save(target, format="PNG")
        return str(target)
    except Exception:
        return ""


def _classify_image(*, caption_text: str, ocr_text: str) -> str:
    text = f"{caption_text} {ocr_text}".lower()
    if re.search(r"\btable\b|表\s*[\d１-９一二三四五六七八九十]+", text):
        return "table_image"
    if re.search(r"\bfig(?:ure)?\b|图\s*[\d１-９一二三四五六七八九十]+", text):
        return "figure_image"
    if len(re.findall(r"\d", text)) >= 12:
        return "chart_image"
    return "image"


def _fallback_vision_summary(
    *,
    page: int,
    caption_text: str,
    ocr_text: str,
    kind: str,
) -> str:
    if caption_text and ocr_text:
        return f"{kind} on page {page}; caption and OCR text were extracted."
    if caption_text:
        return f"{kind} on page {page}; caption text was extracted."
    if ocr_text:
        return f"{kind} on page {page}; OCR text was extracted."
    return f"{kind} on page {page}; stored for later OCR or vision analysis."


def _cross_page_image_records(
    images: list[ExtractedImage],
    page_heights: dict[int, float],
    output_dir: Path,
    document_id: str,
) -> list[ExtractedImage]:
    extras: list[ExtractedImage] = []
    by_page: dict[int, list[ExtractedImage]] = {}
    for image in images:
        by_page.setdefault(image.page_start, []).append(image)

    for page, current_images in sorted(by_page.items()):
        next_images = by_page.get(page + 1, [])
        if not next_images:
            continue
        page_height = page_heights.get(page, 0.0)
        next_page_height = page_heights.get(page + 1, 0.0)
        for current in current_images:
            if page_height <= 0 or current.bbox[3] < page_height * 0.86:
                continue
            for following in next_images:
                if next_page_height <= 0 or following.bbox[1] > next_page_height * 0.14:
                    continue
                if _horizontal_overlap(current.bbox, following.bbox) < 0.55:
                    continue
                composite_path = _make_composite_image(current, following, output_dir)
                composite_hash = hashlib.sha256(
                    f"{current.image_hash}:{following.image_hash}".encode("utf-8")
                ).hexdigest()
                bbox_json = json.dumps(
                    {
                        "segments": [
                            json.loads(current.bbox_json),
                            json.loads(following.bbox_json),
                        ]
                    },
                    ensure_ascii=False,
                )
                caption = " ".join(
                    item
                    for item in [current.caption_text, following.caption_text]
                    if item
                )[:900]
                ocr = " ".join(item for item in [current.ocr_text, following.ocr_text] if item)[:3000]
                status = "ready" if caption or ocr else "stored_needs_ocr"
                extra = ExtractedImage(
                    id=f"{document_id}_image_cross_{len(extras):05d}",
                    document_id=document_id,
                    image_hash=composite_hash,
                    page_start=current.page_start,
                    page_end=following.page_end,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    image_path=composite_path,
                    thumbnail_path=_make_thumbnail(Path(composite_path)) if composite_path else "",
                    width=max(current.width, following.width),
                    height=current.height + following.height,
                    kind="cross_page_image",
                    ocr_text=ocr,
                    ocr_status=_combine_ocr_status([current, following]),
                    ocr_error=_combine_ocr_error([current, following]),
                    vision_summary=f"Possible cross-page image spanning pages {current.page_start}-{following.page_end}.",
                    caption_text=caption,
                    status=status,
                    bbox_json_override=bbox_json,
                )
                extras.append(extra)
                break
    return extras


def _horizontal_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return overlap / max(min(left[2] - left[0], right[2] - right[0]), 1.0)


def _make_composite_image(left: ExtractedImage, right: ExtractedImage, output_dir: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    try:
        with Image.open(left.image_path) as top, Image.open(right.image_path) as bottom:
            width = max(top.width, bottom.width)
            height = top.height + bottom.height
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(top.convert("RGB"), ((width - top.width) // 2, 0))
            canvas.paste(bottom.convert("RGB"), ((width - bottom.width) // 2, top.height))
            path = output_dir / f"cross_{left.page_start:04d}_{right.page_start:04d}_{left.image_hash[:8]}_{right.image_hash[:8]}.png"
            canvas.save(path, format="PNG")
            return str(path)
    except Exception:
        return ""


def _image_text_for_retrieval(image: ExtractedImage) -> str:
    parts = [
        f"Image evidence. Type: {image.kind}. Pages: {image.page_start}-{image.page_end}.",
        f"Caption: {image.caption_text}" if image.caption_text else "",
        f"OCR: {image.ocr_text}" if image.ocr_text else "",
        f"Summary: {image.vision_summary}" if image.vision_summary else "",
    ]
    return "\n".join(part for part in parts if part)
