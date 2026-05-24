from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from backend.app.document_processing import ChunkRecord, count_tokens


CAPTION_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?\.?|table)\s*\d*|[\u56fe\u8868]\s*[\d\uff11-\uff19一二三四五六七八九十]+",
    re.IGNORECASE,
)


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
            "vision_summary": self.vision_summary,
            "caption_text": self.caption_text,
            "status": self.status,
        }


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
    ocr_cache: dict[str, str] = {}
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
                should_ocr = len(ocr_cache) < max_ocr_images and _should_ocr_image(bbox, page.rect, caption_text)
                if image_hash in ocr_cache:
                    ocr_text = ocr_cache[image_hash]
                elif should_ocr:
                    ocr_text = _ocr_image(image_path)
                    ocr_cache[image_hash] = ocr_text
                else:
                    ocr_text = ""

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


def _ocr_image(image_path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng")
    except Exception:
        try:
            text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
        except Exception:
            return ""
    return " ".join(text.split())[:3000]


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
    if re.search(r"\btable\b|[\u8868]\s*[\d\uff11-\uff19一二三四五六七八九十]+", text):
        return "table_image"
    if re.search(r"\bfig(?:ure)?\b|[\u56fe]\s*[\d\uff11-\uff19一二三四五六七八九十]+", text):
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
