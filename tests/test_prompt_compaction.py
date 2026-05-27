from __future__ import annotations

import unittest

from backend.app.agent_parts.answering import ANSWER_SYSTEM_PROMPT
from backend.app.image_processing import ExtractedImage, _vision_prompt_for_image


class PromptCompactionTests(unittest.TestCase):
    def test_core_prompts_stay_compact(self) -> None:
        image = ExtractedImage(
            id="img-1",
            document_id="doc-1",
            image_hash="hash",
            page_start=1,
            page_end=1,
            bbox=(0.0, 0.0, 10.0, 10.0),
            image_path="",
            thumbnail_path="",
            width=10,
            height=10,
            kind="chart_image",
            ocr_text="accuracy 92%",
            vision_summary="",
            caption_text="Figure 1. Accuracy.",
            status="ready",
        )

        self.assertLessEqual(len(ANSWER_SYSTEM_PROMPT), 260)
        self.assertLessEqual(len(_vision_prompt_for_image(image)), 180)


if __name__ == "__main__":
    unittest.main()
