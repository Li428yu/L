from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_NAME = "baselines.json"
DEFAULT_BASELINE_ID = "sample_smoke"


@dataclass(frozen=True)
class EvalBaseline:
    id: str
    label: str
    tier: str
    status: str
    suite_path: str
    suite_name: str
    document_policy: str = "expected_ready"
    description: str = ""
    pause_reason: str = ""

    @property
    def runnable_by_default(self) -> bool:
        return self.status in {"active", "supporting"}


@dataclass(frozen=True)
class EvalBaselineManifest:
    default_baseline: str
    baselines: list[EvalBaseline]

    def by_id(self) -> dict[str, EvalBaseline]:
        return {baseline.id: baseline for baseline in self.baselines}


def load_eval_baselines(eval_dir: Path) -> EvalBaselineManifest:
    path = eval_dir / DEFAULT_MANIFEST_NAME
    if not path.exists():
        return EvalBaselineManifest(default_baseline=DEFAULT_BASELINE_ID, baselines=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    baselines = [
        EvalBaseline(
            id=str(item.get("id") or "").strip(),
            label=str(item.get("label") or "").strip(),
            tier=str(item.get("tier") or "").strip(),
            status=str(item.get("status") or "").strip(),
            suite_path=str(item.get("suite_path") or "").strip(),
            suite_name=str(item.get("suite_name") or "").strip(),
            document_policy=str(item.get("document_policy") or "expected_ready").strip(),
            description=str(item.get("description") or "").strip(),
            pause_reason=str(item.get("pause_reason") or "").strip(),
        )
        for item in payload.get("baselines", [])
        if isinstance(item, dict)
    ]
    return EvalBaselineManifest(
        default_baseline=str(payload.get("default_baseline") or DEFAULT_BASELINE_ID).strip(),
        baselines=baselines,
    )


def resolve_eval_baseline(eval_dir: Path, value: str | None = None) -> EvalBaseline | None:
    manifest = load_eval_baselines(eval_dir)
    baseline_id = (value or manifest.default_baseline or DEFAULT_BASELINE_ID).strip()
    return manifest.by_id().get(baseline_id)


def resolve_eval_suite_path(
    *,
    eval_dir: Path,
    suite_name: str | None = None,
    suite_path: str | None = None,
    baseline_id: str | None = None,
) -> tuple[Path, EvalBaseline | None]:
    baseline: EvalBaseline | None = None
    if suite_path:
        candidate = Path(suite_path)
        if not candidate.is_absolute():
            candidate = eval_dir.parent / candidate
    else:
        baseline = resolve_eval_baseline(eval_dir, baseline_id or suite_name)
        if baseline is not None:
            candidate = eval_dir / baseline.suite_path
        else:
            raw_name = (suite_name or "").strip()
            if not raw_name:
                default_baseline = resolve_eval_baseline(eval_dir)
                if default_baseline is not None:
                    baseline = default_baseline
                    candidate = eval_dir / default_baseline.suite_path
                else:
                    candidate = eval_dir / "sample_eval_set.json"
            else:
                if not raw_name.endswith(".json"):
                    raw_name = f"{raw_name}.json"
                candidate = eval_dir / raw_name

    resolved = candidate.resolve()
    eval_root = eval_dir.resolve()
    try:
        allowed = resolved.is_relative_to(eval_root)
    except AttributeError:
        allowed = str(resolved).startswith(str(eval_root))
    if not allowed:
        raise ValueError("评测集必须放在 evals 目录下。")
    return resolved, baseline


def expected_document_references(cases: list[Any]) -> list[str]:
    references: list[str] = []
    for case in cases:
        expected_documents = _get_value(case, "expected_documents") or []
        if isinstance(expected_documents, list):
            references.extend(str(item).strip() for item in expected_documents if str(item).strip())
        expected_document = str(_get_value(case, "expected_document") or "").strip()
        if expected_document:
            references.append(expected_document)
    return _dedupe(references)


def resolve_eval_document_ids(
    *,
    documents: list[Any],
    cases: list[Any],
    requested_document_ids: list[str] | None = None,
    document_policy: str = "expected_ready",
) -> list[str]:
    explicit_ids = [item.strip() for item in (requested_document_ids or []) if item.strip()]
    if explicit_ids:
        return explicit_ids

    ready_documents = [document for document in documents if _get_value(document, "status") == "ready"]
    if document_policy == "all_ready":
        return [str(_get_value(document, "id")) for document in ready_documents if _get_value(document, "id")]

    expected_refs = expected_document_references(cases)
    if not expected_refs:
        return [str(_get_value(document, "id")) for document in ready_documents if _get_value(document, "id")]

    by_id = {str(_get_value(document, "id")): document for document in ready_documents}
    by_name = {Path(str(_get_value(document, "file_name") or "")).name.lower(): document for document in ready_documents}
    matched_ids: list[str] = []
    for reference in expected_refs:
        document = by_id.get(reference) or by_name.get(Path(reference).name.lower())
        document_id = str(_get_value(document, "id") or "").strip() if document is not None else ""
        if document_id and document_id not in matched_ids:
            matched_ids.append(document_id)
    return matched_ids


def baseline_metadata(baseline: EvalBaseline | None) -> dict[str, Any]:
    if baseline is None:
        return {}
    return {
        "baseline_id": baseline.id,
        "baseline_label": baseline.label,
        "baseline_tier": baseline.tier,
        "baseline_status": baseline.status,
        "baseline_document_policy": baseline.document_policy,
        "baseline_pause_reason": baseline.pause_reason,
    }


def _get_value(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
