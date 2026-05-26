from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app.eval_baselines import (
    load_eval_baselines,
    resolve_eval_document_ids,
    resolve_eval_suite_path,
)


def write_manifest(eval_dir: Path) -> None:
    (eval_dir / "baselines.json").write_text(
        json.dumps(
            {
                "default_baseline": "gold",
                "baselines": [
                    {
                        "id": "gold",
                        "label": "Gold",
                        "tier": "gold",
                        "status": "active",
                        "suite_path": "gold.json",
                        "suite_name": "gold_suite",
                        "document_policy": "expected_ready",
                    },
                    {
                        "id": "smoke",
                        "label": "Smoke",
                        "tier": "smoke",
                        "status": "supporting",
                        "suite_path": "smoke.json",
                        "suite_name": "smoke_suite",
                        "document_policy": "all_ready",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "gold.json").write_text('{"cases": []}', encoding="utf-8")
    (eval_dir / "smoke.json").write_text('{"cases": []}', encoding="utf-8")


class EvalBaselineTests(unittest.TestCase):
    def test_default_baseline_resolves_to_gold_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            write_manifest(eval_dir)

            suite_path, baseline = resolve_eval_suite_path(eval_dir=eval_dir)

            self.assertEqual(suite_path.name, "gold.json")
            self.assertIsNotNone(baseline)
            self.assertEqual(baseline.id, "gold")

    def test_suite_argument_can_select_baseline_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            write_manifest(eval_dir)

            suite_path, baseline = resolve_eval_suite_path(eval_dir=eval_dir, suite_name="smoke")

            self.assertEqual(suite_path.name, "smoke.json")
            self.assertIsNotNone(baseline)
            self.assertEqual(baseline.document_policy, "all_ready")

    def test_expected_ready_document_policy_filters_to_expected_documents(self) -> None:
        documents = [
            SimpleNamespace(id="doc-1", file_name="paper-a.pdf", status="ready"),
            SimpleNamespace(id="doc-2", file_name="paper-b.pdf", status="ready"),
            SimpleNamespace(id="doc-3", file_name="paper-c.pdf", status="failed"),
        ]
        cases = [
            SimpleNamespace(expected_document="paper-a.pdf", expected_documents=[]),
            SimpleNamespace(expected_document="", expected_documents=["doc-2"]),
        ]

        document_ids = resolve_eval_document_ids(
            documents=documents,
            cases=cases,
            document_policy="expected_ready",
        )

        self.assertEqual(document_ids, ["doc-1", "doc-2"])

    def test_manifest_loader_exposes_default_and_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            write_manifest(eval_dir)

            manifest = load_eval_baselines(eval_dir)

            self.assertEqual(manifest.default_baseline, "gold")
            self.assertEqual([baseline.status for baseline in manifest.baselines], ["active", "supporting"])


if __name__ == "__main__":
    unittest.main()
