from __future__ import annotations

import re
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "backend" / "app"

OVERFIT_PATTERNS = [
    r"attention-is-all-you-need",
    r"attention\s+is\s+all\s+you\s+need",
    r"\btransformer(s)?\b",
    r"\battention\b",
    r"sequence\s+transduction",
    r"machine\s+translation",
    r"\bbert\b",
    r"\bu-net\b",
    r"\bunet\b",
    r"\bgpt-2\b",
    r"\bgpt2\b",
    r"segment\s+anything",
    r"\bsa-1b\b",
    r"ocrmypdf_cardinal",
    r"linnsequencer",
    r"\bwmt\s+2014\b",
    r"english-to-german",
    r"english-to-french",
    r"\b28\.4\b",
    r"\b41\.8\b",
    r"\b400\s+million\b",
    r"\bbookscorpus\b",
    r"\bsquad\b",
    r"\bmultinli\b",
    r"\bisbi\b",
    r"\blambada\b",
    r"\bnist\b",
    r"\bbleu\b",
    r"\bglue\b",
    r"identify,\s*protect,\s*detect",
    r"map,\s*measure,\s*and\s*manage",
]


def test_runtime_code_does_not_embed_document_specific_eval_rules() -> None:
    offenders: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for pattern in OVERFIT_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(RUNTIME_ROOT.parents[1])}: {pattern}")

    assert not offenders, "Runtime code contains document-specific overfit markers:\n" + "\n".join(offenders)
