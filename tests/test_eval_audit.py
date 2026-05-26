from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.eval_audit import (
    EvalAuditContext,
    KnownChunk,
    KnownDocument,
    audit_eval_paths,
)


def write_suite(directory: Path, cases: list[dict]) -> Path:
    path = directory / "suite.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


def finding_codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


class EvalAuditTests(unittest.TestCase):
    def test_document_reference_can_match_local_document_id_or_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_suite(
                Path(tmp),
                [
                    {
                        "id": "by-id",
                        "question": "What does the paper say?",
                        "expected_document": "doc-1",
                        "expected_keywords": ["attention", "sequence"],
                        "expected_evidence_keywords": ["attention"],
                    },
                    {
                        "id": "by-name",
                        "question": "What else does the paper say?",
                        "expected_document": "paper.pdf",
                        "expected_keywords": ["encoder", "decoder"],
                        "expected_evidence_keywords": ["encoder"],
                    },
                ],
            )
            context = EvalAuditContext(
                documents=[KnownDocument(document_id="doc-1", file_name="paper.pdf", page_count=5, status="ready")]
            )

            report = audit_eval_paths([path], context=context)

            self.assertNotIn("local_missing_document", finding_codes(report))

    def test_gold_chunk_missing_and_document_mismatch_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_suite(
                Path(tmp),
                [
                    {
                        "id": "chunk-check",
                        "question": "What does the paper say?",
                        "expected_document": "paper.pdf",
                        "expected_keywords": ["attention", "sequence"],
                        "expected_chunk_ids": ["other-doc-chunk", "missing-chunk"],
                    }
                ],
            )
            context = EvalAuditContext(
                documents=[KnownDocument(document_id="doc-1", file_name="paper.pdf", page_count=5, status="ready")],
                chunks=[KnownChunk(chunk_id="other-doc-chunk", document_id="doc-2")],
            )

            report = audit_eval_paths([path], context=context)
            codes = finding_codes(report)

            self.assertIn("gold_chunk_document_mismatch", codes)
            self.assertIn("local_missing_gold_chunk", codes)

    def test_expected_pages_must_match_gold_chunk_page_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_suite(
                Path(tmp),
                [
                    {
                        "id": "aligned-pages",
                        "question": "What does the paper say?",
                        "expected_keywords": ["attention", "sequence"],
                        "expected_pages": [2, 3],
                        "expected_chunk_ids": ["gold-1"],
                    },
                    {
                        "id": "misaligned-pages",
                        "question": "What else does the paper say?",
                        "expected_keywords": ["encoder", "decoder"],
                        "expected_pages": [2],
                        "expected_chunk_ids": ["gold-1"],
                    },
                ],
            )
            context = EvalAuditContext(
                chunks=[KnownChunk(chunk_id="gold-1", document_id="doc-1", page_start=2, page_end=3)]
            )

            report = audit_eval_paths([path], context=context)
            page_findings = [
                finding
                for finding in report.findings
                if finding.code == "expected_pages_gold_chunk_mismatch"
            ]

            self.assertEqual([finding.case_id for finding in page_findings], ["misaligned-pages"])
            self.assertEqual(page_findings[0].value["gold_chunk_pages"], [2, 3])

    def test_modality_detection_avoids_port_scan_and_promptable_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_suite(
                Path(tmp),
                [
                    {
                        "id": "port-scan",
                        "question": "端口扫描如何建立 TCP 连接？",
                        "expected_keywords": ["TCP", "connect"],
                        "expected_evidence_keywords": ["connect"],
                    },
                    {
                        "id": "promptable",
                        "question": "What is the promptable segmentation task?",
                        "expected_keywords": ["prompt", "mask"],
                        "expected_evidence_keywords": ["prompt"],
                    },
                ],
            )

            report = audit_eval_paths([path])

            self.assertNotIn("question_modality_missing", finding_codes(report))

    def test_non_template_placeholder_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_suite(
                Path(tmp),
                [
                    {
                        "id": "placeholder",
                        "question": "What is <topic>?",
                        "expected_keywords": ["<keyword>"],
                    }
                ],
            )

            report = audit_eval_paths([path])

            self.assertIn("placeholder_value", finding_codes(report))


if __name__ == "__main__":
    unittest.main()
