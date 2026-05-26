from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.image_processing import ExtractedImage, enrich_images_with_vision, vision_status_counts
from backend.app.storage import MetadataStore


def make_image(
    *,
    image_path: str,
    kind: str = "figure_image",
    status: str = "stored_needs_ocr",
) -> ExtractedImage:
    return ExtractedImage(
        id="img-1",
        document_id="doc-1",
        image_hash="hash",
        page_start=1,
        page_end=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
        image_path=image_path,
        thumbnail_path=image_path,
        width=10,
        height=10,
        kind=kind,
        ocr_text="",
        vision_summary="",
        caption_text="",
        status=status,
    )


class VisionStatusTests(unittest.TestCase):
    def test_vision_success_marks_ready_and_clears_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_file = Path(temp_dir) / "image.png"
            image_file.write_bytes(b"fake")
            images = [make_image(image_path=str(image_file))]

            result = enrich_images_with_vision(
                images=images,
                analyze_image=lambda path, prompt: "A chart showing model accuracy.",
            )

        self.assertEqual(result[0].status, "vision_ready")
        self.assertEqual(result[0].vision_error, "")
        self.assertIn("model accuracy", result[0].vision_summary)

    def test_vision_failure_records_error_without_swallowing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_file = Path(temp_dir) / "image.png"
            image_file.write_bytes(b"fake")
            images = [make_image(image_path=str(image_file))]

            def fail(path, prompt):
                raise RuntimeError("provider unavailable")

            result = enrich_images_with_vision(images=images, analyze_image=fail)

        self.assertEqual(result[0].status, "vision_failed")
        self.assertIn("RuntimeError", result[0].vision_error)
        self.assertIn("provider unavailable", result[0].vision_error)

    def test_vision_skip_records_policy_reason(self) -> None:
        images = [
            make_image(
                image_path="missing-but-not-used.png",
                kind="image",
                status="ready",
            )
        ]

        result = enrich_images_with_vision(
            images=images,
            analyze_image=lambda path, prompt: "should not run",
        )

        self.assertEqual(result[0].status, "vision_skipped")
        self.assertEqual(result[0].vision_error, "not_selected_by_policy")

    def test_vision_status_counts_are_explicit(self) -> None:
        images = [
            make_image(image_path="a.png", status="vision_ready"),
            make_image(image_path="b.png", status="vision_failed"),
            make_image(image_path="c.png", status="vision_failed"),
        ]

        self.assertEqual(
            vision_status_counts(images),
            {"vision_ready": 1, "vision_failed": 2},
        )

    def test_storage_persists_vision_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MetadataStore(Path(temp_dir) / "metadata.sqlite3")
            image = make_image(image_path="failed.png", status="vision_failed")
            image.vision_error = "RuntimeError: provider unavailable"

            store.replace_document_images(
                document_id="doc-1",
                images=[image.to_storage_dict()],
            )
            rows = store.list_document_images("doc-1")

        self.assertEqual(rows[0]["status"], "vision_failed")
        self.assertEqual(rows[0]["vision_error"], "RuntimeError: provider unavailable")


if __name__ == "__main__":
    unittest.main()
