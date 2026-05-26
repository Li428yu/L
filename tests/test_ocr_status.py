from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.image_processing import (
    OCRResult,
    ExtractedImage,
    _ocr_pdf_image_with_status,
    ocr_status_counts,
)
from backend.app.storage import MetadataStore


def make_image(
    *,
    ocr_status: str = "",
    ocr_error: str = "",
    ocr_text: str = "",
) -> ExtractedImage:
    return ExtractedImage(
        id="img-1",
        document_id="doc-1",
        image_hash="hash",
        page_start=1,
        page_end=1,
        bbox=(0.0, 0.0, 100.0, 100.0),
        image_path="image.png",
        thumbnail_path="thumb.png",
        width=100,
        height=100,
        kind="image",
        ocr_text=ocr_text,
        ocr_status=ocr_status,
        ocr_error=ocr_error,
        vision_summary="",
        caption_text="",
        status="stored_needs_ocr",
    )


class OCRStatusTests(unittest.TestCase):
    def test_policy_skip_records_reason_without_attempting_ocr(self) -> None:
        result = _ocr_pdf_image_with_status(
            image_hash="hash-1",
            image_path=Path("missing.png"),
            bbox=(0.0, 0.0, 10.0, 10.0),
            page_rect=SimpleNamespace(width=1000, height=1000),
            caption_text="",
            max_ocr_images=40,
            ocr_cache={},
        )

        self.assertEqual(result.status, "ocr_skipped")
        self.assertEqual(result.error, "not_selected_by_policy")
        self.assertEqual(result.text, "")

    def test_max_limit_skip_records_reason_without_attempting_ocr(self) -> None:
        result = _ocr_pdf_image_with_status(
            image_hash="hash-1",
            image_path=Path("missing.png"),
            bbox=(0.0, 0.0, 500.0, 500.0),
            page_rect=SimpleNamespace(width=1000, height=1000),
            caption_text="",
            max_ocr_images=0,
            ocr_cache={},
        )

        self.assertEqual(result.status, "ocr_skipped")
        self.assertEqual(result.error, "max_images_limit")

    def test_ocr_result_cache_reuses_status_and_text(self) -> None:
        cache: dict[str, OCRResult] = {}
        with patch(
            "backend.app.image_processing._ocr_image_with_status",
            return_value=OCRResult(text="recognized text", status="ocr_ready"),
        ) as ocr:
            first = _ocr_pdf_image_with_status(
                image_hash="same-hash",
                image_path=Path("image.png"),
                bbox=(0.0, 0.0, 500.0, 500.0),
                page_rect=SimpleNamespace(width=1000, height=1000),
                caption_text="",
                max_ocr_images=40,
                ocr_cache=cache,
            )
            second = _ocr_pdf_image_with_status(
                image_hash="same-hash",
                image_path=Path("image.png"),
                bbox=(0.0, 0.0, 500.0, 500.0),
                page_rect=SimpleNamespace(width=1000, height=1000),
                caption_text="",
                max_ocr_images=40,
                ocr_cache=cache,
            )

        self.assertEqual(first.text, "recognized text")
        self.assertEqual(second.status, "ocr_ready")
        self.assertEqual(ocr.call_count, 1)

    def test_ocr_status_counts_are_explicit(self) -> None:
        images = [
            make_image(ocr_status="ocr_ready", ocr_text="text"),
            make_image(ocr_status="ocr_failed", ocr_error="RuntimeError"),
            make_image(ocr_status="ocr_skipped", ocr_error="max_images_limit"),
        ]

        self.assertEqual(
            ocr_status_counts(images),
            {"ocr_ready": 1, "ocr_failed": 1, "ocr_skipped": 1},
        )

    def test_storage_persists_ocr_status_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MetadataStore(Path(temp_dir) / "metadata.sqlite3")
            image = make_image(
                ocr_status="ocr_failed",
                ocr_error="RuntimeError: tesseract unavailable",
            )

            store.replace_document_images(
                document_id="doc-1",
                images=[image.to_storage_dict()],
            )
            rows = store.list_document_images("doc-1")

        self.assertEqual(rows[0]["ocr_status"], "ocr_failed")
        self.assertEqual(rows[0]["ocr_error"], "RuntimeError: tesseract unavailable")


if __name__ == "__main__":
    unittest.main()
