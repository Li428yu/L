from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ALLOWED_CASE_FIELDS = {
    "id",
    "question",
    "expected_keywords",
    "expected_answer",
    "expected_document",
    "expected_page",
    "expected_documents",
    "expected_pages",
    "expected_evidence_keywords",
    "expected_modalities",
    "expected_chunk_ids",
    "relation_keywords",
    "required_document_count",
    "expected_claims",
    "forbidden_claims",
    "expected_relation",
    "expected_refusal",
    "judge_rubric",
}

ALLOWED_MODALITIES = {"image", "figure", "chart", "vision", "table", "ocr"}
SEVERITIES = {"error": 3, "warning": 2, "info": 1}

GENERIC_TERMS = {
    "pdf",
    "docx",
    "paper",
    "document",
    "documents",
    "evidence",
    "method",
    "methods",
    "result",
    "results",
    "model",
    "models",
    "system",
    "task",
    "figure",
    "image",
    "table",
    "use",
    "uses",
    "data",
    "claim",
    "claims",
    "证据",
    "依据",
    "支持",
    "不同",
    "相同",
    "差异",
    "方法",
    "结果",
    "内容",
    "图片",
    "截图",
    "图",
    "图表",
    "表格",
    "文档",
    "论文",
    "实验",
    "模型",
    "系统",
    "核心",
    "贡献",
    "设计",
    "模块",
    "局限",
    "不足",
    "说明",
    "包括",
    "不能",
    "无法",
    "无法判断",
    "第",
    "页",
}

REFUSAL_MARKERS = {
    "证据不足",
    "无法证明",
    "不能证明",
    "没有证据",
    "无证据",
    "无法回答",
    "不能回答",
    "未说明",
    "未明确",
    "未索引",
    "not enough evidence",
    "insufficient evidence",
    "cannot answer",
    "not supported",
}


@dataclass(frozen=True)
class KnownDocument:
    document_id: str
    file_name: str
    page_count: int
    status: str


@dataclass(frozen=True)
class KnownChunk:
    chunk_id: str
    document_id: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str = ""


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    suite: str
    case_id: str
    code: str
    message: str
    field: str = ""
    value: Any = None


@dataclass(frozen=True)
class AuditReport:
    suites: list[dict[str, Any]]
    findings: list[AuditFinding]

    def to_dict(self) -> dict[str, Any]:
        counts = {"error": 0, "warning": 0, "info": 0}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return {
            "summary": {
                "suite_count": len(self.suites),
                "case_count": sum(int(item.get("case_count", 0)) for item in self.suites),
                "finding_count": len(self.findings),
                **counts,
            },
            "suites": self.suites,
            "findings": [asdict(item) for item in self.findings],
        }


class EvalAuditContext:
    def __init__(
        self,
        *,
        documents: list[KnownDocument] | None = None,
        chunks: list[KnownChunk] | None = None,
    ) -> None:
        self.documents = documents or []
        self.chunks = chunks or []
        self.documents_by_name = {
            normalize_name(document.file_name): document
            for document in self.documents
            if document.file_name
        }
        self.documents_by_id = {
            document.document_id: document
            for document in self.documents
            if document.document_id
        }
        self.chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in self.chunks
            if chunk.chunk_id
        }


def audit_eval_paths(paths: list[Path], *, context: EvalAuditContext | None = None) -> AuditReport:
    findings: list[AuditFinding] = []
    suites: list[dict[str, Any]] = []
    all_case_ids: dict[str, str] = {}

    for path in paths:
        suite_summary, suite_findings, case_ids = audit_eval_suite(path, context=context)
        suites.append(suite_summary)
        findings.extend(suite_findings)
        for case_id in case_ids:
            if case_id in all_case_ids:
                findings.append(
                    AuditFinding(
                        severity="info",
                        suite=path.name,
                        case_id=case_id,
                        code="duplicate_case_id_across_suites",
                        message=f"Case id also appears in {all_case_ids[case_id]}.",
                        field="id",
                    )
                )
            else:
                all_case_ids[case_id] = path.name

    findings.sort(key=lambda item: (-SEVERITIES.get(item.severity, 0), item.suite, item.case_id, item.code))
    return AuditReport(suites=suites, findings=findings)


def audit_eval_suite(path: Path, *, context: EvalAuditContext | None = None) -> tuple[dict[str, Any], list[AuditFinding], list[str]]:
    context = context or EvalAuditContext()
    findings: list[AuditFinding] = []
    case_ids: list[str] = []
    suite_name = path.name
    summary = {
        "path": str(path),
        "suite": suite_name,
        "case_count": 0,
        "refusal_case_count": 0,
        "gold_chunk_case_count": 0,
        "vision_case_count": 0,
        "table_case_count": 0,
        "ocr_case_count": 0,
    }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        findings.append(
            AuditFinding(
                severity="error",
                suite=suite_name,
                case_id="",
                code="invalid_json",
                message=f"Cannot parse eval suite JSON: {exc}",
            )
        )
        return summary, findings, case_ids

    cases = payload.get("cases")
    if not isinstance(cases, list):
        findings.append(
            AuditFinding(
                severity="error",
                suite=suite_name,
                case_id="",
                code="missing_cases",
                message="Eval suite must contain a cases list.",
                field="cases",
            )
        )
        return summary, findings, case_ids

    is_template = "templates" in path.parts or "template" in path.name
    summary["case_count"] = len(cases)
    if not cases:
        findings.append(
            AuditFinding(
                severity="error",
                suite=suite_name,
                case_id="",
                code="empty_suite",
                message="Eval suite has no cases.",
                field="cases",
            )
        )

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            findings.append(
                AuditFinding(
                    severity="error",
                    suite=suite_name,
                    case_id=f"#{index + 1}",
                    code="invalid_case",
                    message="Case must be a JSON object.",
                )
            )
            continue

        case_id = str(raw_case.get("id") or "").strip()
        case_ids.append(case_id)
        if bool(raw_case.get("expected_refusal")):
            summary["refusal_case_count"] += 1
        if _string_list(raw_case.get("expected_chunk_ids")):
            summary["gold_chunk_case_count"] += 1
        modalities = {item.lower() for item in _string_list(raw_case.get("expected_modalities"))}
        if modalities & {"image", "figure", "chart", "vision"}:
            summary["vision_case_count"] += 1
        if "table" in modalities:
            summary["table_case_count"] += 1
        if "ocr" in modalities:
            summary["ocr_case_count"] += 1

        findings.extend(
            _audit_case(
                suite=suite_name,
                case=raw_case,
                index=index,
                seen_ids=seen_ids,
                seen_questions=seen_questions,
                context=context,
                is_template=is_template,
            )
        )

    if suite_name.startswith("current_pdf") and summary["gold_chunk_case_count"] != summary["case_count"]:
        findings.append(
            AuditFinding(
                severity="warning",
                suite=suite_name,
                case_id="",
                code="partial_gold_chunk_coverage",
                message="PDF gold suite should give expected_chunk_ids for every case.",
                field="expected_chunk_ids",
                value={
                    "case_count": summary["case_count"],
                    "gold_chunk_case_count": summary["gold_chunk_case_count"],
                },
            )
        )

    return summary, findings, case_ids


def _audit_case(
    *,
    suite: str,
    case: dict[str, Any],
    index: int,
    seen_ids: set[str],
    seen_questions: dict[str, str],
    context: EvalAuditContext,
    is_template: bool,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    case_id = str(case.get("id") or "").strip()
    display_id = case_id or f"#{index + 1}"
    question = str(case.get("question") or "").strip()
    expected_refusal = bool(case.get("expected_refusal"))
    expected_keywords = _string_list(case.get("expected_keywords"))
    expected_claims = _string_list(case.get("expected_claims"))
    evidence_keywords = _string_list(case.get("expected_evidence_keywords"))
    expected_chunk_ids = _string_list(case.get("expected_chunk_ids"))
    modalities = _string_list(case.get("expected_modalities"))
    expected_documents = _expected_documents(case)
    expected_pages = _expected_pages(case)

    if not case_id:
        findings.append(_finding("error", suite, display_id, "missing_case_id", "Case id is required.", "id"))
    elif case_id in seen_ids:
        findings.append(_finding("error", suite, display_id, "duplicate_case_id", "Case id is duplicated.", "id"))
    seen_ids.add(case_id)

    if not question:
        findings.append(_finding("error", suite, display_id, "missing_question", "Question is required.", "question"))
    elif question in seen_questions:
        findings.append(
            _finding(
                "warning",
                suite,
                display_id,
                "duplicate_question",
                f"Question duplicates case {seen_questions[question]}.",
                "question",
            )
        )
    else:
        seen_questions[question] = display_id

    unknown_fields = sorted(set(case) - ALLOWED_CASE_FIELDS)
    if unknown_fields:
        findings.append(
            _finding(
                "warning",
                suite,
                display_id,
                "unknown_case_fields",
                "Case has fields ignored by EvaluationCase.",
                value=unknown_fields,
            )
        )

    if not is_template and _contains_placeholder(case):
        findings.append(
            _finding(
                "error",
                suite,
                display_id,
                "placeholder_value",
                "Non-template suite still contains placeholder values.",
            )
        )

    findings.extend(_audit_list_field(suite, display_id, case, "expected_keywords"))
    findings.extend(_audit_list_field(suite, display_id, case, "expected_claims"))
    findings.extend(_audit_list_field(suite, display_id, case, "expected_evidence_keywords"))
    findings.extend(_audit_list_field(suite, display_id, case, "expected_chunk_ids"))
    findings.extend(_audit_list_field(suite, display_id, case, "expected_modalities"))

    unknown_modalities = sorted({item.lower() for item in modalities} - ALLOWED_MODALITIES)
    if unknown_modalities:
        findings.append(
            _finding(
                "error",
                suite,
                display_id,
                "unknown_modality",
                "expected_modalities contains unsupported values.",
                "expected_modalities",
                unknown_modalities,
            )
        )

    if not expected_refusal:
        answer_anchors = _specific_terms([*expected_claims, *expected_keywords])
        if not expected_keywords and not expected_claims and not str(case.get("expected_answer") or "").strip():
            findings.append(
                _finding(
                    "error",
                    suite,
                    display_id,
                    "missing_answer_target",
                    "Positive case has no expected keywords, claims, or answer.",
                )
            )
        elif len(answer_anchors) < 2:
            findings.append(
                _finding(
                    "warning",
                    suite,
                    display_id,
                    "weak_answer_anchors",
                    "Positive case has fewer than two specific answer anchors after generic terms are ignored.",
                    value={"specific_terms": answer_anchors},
                )
            )
        if not evidence_keywords and not expected_chunk_ids:
            findings.append(
                _finding(
                    "warning",
                    suite,
                    display_id,
                    "weak_evidence_anchor",
                    "Case has no expected_evidence_keywords or expected_chunk_ids, so evidence quality is harder to judge.",
                )
            )
    else:
        refusal_terms = _all_terms(case)
        if not any(_contains_refusal_marker(term) for term in refusal_terms):
            findings.append(
                _finding(
                    "warning",
                    suite,
                    display_id,
                    "weak_refusal_target",
                    "Refusal case should include refusal markers such as insufficient evidence or cannot answer.",
                )
            )

    generic_terms = [
        term for term in expected_keywords
        if not _is_specific_term(term) and not _contains_refusal_marker(term)
    ]
    if not expected_refusal and expected_keywords and len(generic_terms) / max(len(expected_keywords), 1) >= 0.5:
        findings.append(
            _finding(
                "warning",
                suite,
                display_id,
                "generic_expected_keywords",
                "At least half of expected_keywords are generic and may inflate scores.",
                "expected_keywords",
                generic_terms,
            )
        )

    findings.extend(_audit_question_modality(suite, display_id, question, modalities))
    findings.extend(_audit_pages(suite, display_id, expected_pages, expected_documents, context))
    findings.extend(_audit_documents(suite, display_id, expected_documents, context))
    findings.extend(_audit_gold_chunks(suite, display_id, expected_chunk_ids, expected_documents, context))
    findings.extend(_audit_expected_pages_against_gold_chunks(suite, display_id, expected_pages, expected_chunk_ids, context))
    return findings


def _audit_list_field(suite: str, case_id: str, case: dict[str, Any], field: str) -> list[AuditFinding]:
    if field not in case:
        return []
    value = case.get(field)
    if not isinstance(value, list):
        return [_finding("error", suite, case_id, "invalid_list_field", f"{field} must be a list.", field, value)]
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    duplicates = sorted({item for item in cleaned if cleaned.count(item) > 1})
    if duplicates:
        return [
            _finding(
                "warning",
                suite,
                case_id,
                "duplicate_list_values",
                f"{field} contains duplicated values.",
                field,
                duplicates,
            )
        ]
    return []


def _audit_question_modality(suite: str, case_id: str, question: str, modalities: list[str]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    lowered = question.lower()
    modal_set = {item.lower() for item in modalities}
    visual_markers = ["图片", "截图", "图像", "figure", "image", "chart", "diagram"]
    if _contains_any_marker(lowered, visual_markers) and not (modal_set & {"image", "figure", "chart", "vision"}):
        findings.append(
            _finding(
                "warning",
                suite,
                case_id,
                "question_modality_missing",
                "Question appears visual but expected_modalities does not require visual evidence.",
                "expected_modalities",
            )
        )
    if ("表格" in lowered or re.search(r"\btable\b", lowered)) and "table" not in modal_set:
        findings.append(
            _finding(
                "warning",
                suite,
                case_id,
                "question_modality_missing",
                "Question appears table-related but expected_modalities does not require table evidence.",
                "expected_modalities",
            )
        )
    if (
        re.search(r"\bocr\b|\bscanned\b", lowered)
        or any(marker in lowered for marker in ["扫描型", "扫描件", "扫描页", "扫描文档"])
    ) and "ocr" not in modal_set:
        findings.append(
            _finding(
                "warning",
                suite,
                case_id,
                "question_modality_missing",
                "Question appears OCR-related but expected_modalities does not require OCR evidence.",
                "expected_modalities",
            )
        )
    return findings


def _audit_pages(
    suite: str,
    case_id: str,
    expected_pages: list[int],
    expected_documents: list[str],
    context: EvalAuditContext,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for page in expected_pages:
        if page <= 0:
            findings.append(_finding("error", suite, case_id, "invalid_expected_page", "Page numbers must be positive.", "expected_pages", page))
    if not context.documents_by_name or not expected_pages or not expected_documents:
        return findings
    for document_name in expected_documents:
        document = _find_document(context, document_name)
        if not document or document.page_count <= 0:
            continue
        too_large = [page for page in expected_pages if page > document.page_count]
        if too_large:
            findings.append(
                _finding(
                    "error",
                    suite,
                    case_id,
                    "expected_page_out_of_range",
                    "Expected page exceeds the local document page count.",
                    "expected_pages",
                    {"document": document.file_name, "page_count": document.page_count, "pages": too_large},
                )
            )
    return findings


def _audit_documents(
    suite: str,
    case_id: str,
    expected_documents: list[str],
    context: EvalAuditContext,
) -> list[AuditFinding]:
    if not context.documents_by_name:
        return []
    findings: list[AuditFinding] = []
    for document_name in expected_documents:
        if _find_document(context, document_name) is None:
            findings.append(
                _finding(
                    "error",
                    suite,
                    case_id,
                    "local_missing_document",
                    "Expected document is not present in the local ready document store.",
                    "expected_document",
                    document_name,
                )
            )
    return findings


def _audit_gold_chunks(
    suite: str,
    case_id: str,
    expected_chunk_ids: list[str],
    expected_documents: list[str],
    context: EvalAuditContext,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if not expected_chunk_ids:
        return findings

    duplicates = sorted({chunk_id for chunk_id in expected_chunk_ids if expected_chunk_ids.count(chunk_id) > 1})
    if duplicates:
        findings.append(
            _finding(
                "error",
                suite,
                case_id,
                "duplicate_expected_chunk_ids",
                "expected_chunk_ids must not contain duplicates.",
                "expected_chunk_ids",
                duplicates,
            )
        )

    for chunk_id in expected_chunk_ids:
        chunk = context.chunks_by_id.get(chunk_id)
        if context.chunks_by_id and not chunk:
            findings.append(
                _finding(
                    "error",
                    suite,
                    case_id,
                    "local_missing_gold_chunk",
                    "expected_chunk_id is not present in the local Chroma collection.",
                    "expected_chunk_ids",
                    chunk_id,
                )
            )
            continue
        if chunk and expected_documents:
            expected_doc_ids = {
                document.document_id
                for name in expected_documents
                for document in [_find_document(context, name)]
                if document
            }
            if expected_doc_ids and chunk.document_id not in expected_doc_ids:
                findings.append(
                    _finding(
                        "error",
                        suite,
                        case_id,
                        "gold_chunk_document_mismatch",
                        "expected_chunk_id belongs to a different local document than expected_document.",
                        "expected_chunk_ids",
                        {"chunk_id": chunk_id, "chunk_document_id": chunk.document_id, "expected_document_ids": sorted(expected_doc_ids)},
                    )
                )
    return findings


def _audit_expected_pages_against_gold_chunks(
    suite: str,
    case_id: str,
    expected_pages: list[int],
    expected_chunk_ids: list[str],
    context: EvalAuditContext,
) -> list[AuditFinding]:
    if not expected_pages or not expected_chunk_ids or not context.chunks_by_id:
        return []

    gold_pages = _gold_chunk_page_numbers(expected_chunk_ids, context)
    if not gold_pages:
        return []

    configured_pages = sorted(set(expected_pages))
    if configured_pages == gold_pages:
        return []

    return [
        _finding(
            "error",
            suite,
            case_id,
            "expected_pages_gold_chunk_mismatch",
            "expected_pages must match the indexed page_start/page_end ranges of expected_chunk_ids.",
            "expected_pages",
            {"expected_pages": configured_pages, "gold_chunk_pages": gold_pages},
        )
    ]


def _gold_chunk_page_numbers(expected_chunk_ids: list[str], context: EvalAuditContext) -> list[int] | None:
    pages: set[int] = set()
    for chunk_id in expected_chunk_ids:
        chunk = context.chunks_by_id.get(chunk_id)
        if chunk is None:
            return None
        if chunk.page_start is None and chunk.page_end is None:
            return None
        start = chunk.page_start if chunk.page_start is not None else chunk.page_end
        end = chunk.page_end if chunk.page_end is not None else start
        if start is None or end is None:
            return None
        if start <= 0 or end <= 0:
            return None
        if end < start:
            end = start
        pages.update(range(start, end + 1))
    return sorted(pages)


def load_local_audit_context(
    *,
    sqlite_path: Path,
    chroma_dir: Path | None = None,
    expected_chunk_ids: list[str] | None = None,
) -> EvalAuditContext:
    documents = load_known_documents(sqlite_path)
    chunks: list[KnownChunk] = []
    if chroma_dir is not None and expected_chunk_ids:
        chunks = load_known_chunks(chroma_dir=chroma_dir, chunk_ids=expected_chunk_ids)
    return EvalAuditContext(documents=documents, chunks=chunks)


def load_known_documents(sqlite_path: Path) -> list[KnownDocument]:
    if not sqlite_path.exists():
        return []
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = conn.execute(
            "select id, file_name, page_count, status from documents where status = 'ready'"
        ).fetchall()
    finally:
        conn.close()
    return [
        KnownDocument(
            document_id=str(row[0]),
            file_name=str(row[1]),
            page_count=int(row[2] or 0),
            status=str(row[3] or ""),
        )
        for row in rows
    ]


def load_known_chunks(*, chroma_dir: Path, chunk_ids: list[str]) -> list[KnownChunk]:
    unique_ids = sorted({chunk_id for chunk_id in chunk_ids if chunk_id})
    if not unique_ids or not chroma_dir.exists():
        return []
    try:
        import chromadb
    except ImportError:
        return []

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name="paper_chunks")
    result = collection.get(ids=unique_ids, include=["metadatas"])
    rows: list[KnownChunk] = []
    for chunk_id, metadata in zip(result.get("ids", []), result.get("metadatas", [])):
        payload = metadata or {}
        rows.append(
            KnownChunk(
                chunk_id=str(payload.get("chunk_id") or chunk_id),
                document_id=str(payload.get("document_id") or ""),
                page_start=_optional_int(payload.get("page_start") or payload.get("page")),
                page_end=_optional_int(payload.get("page_end") or payload.get("page")),
                chunk_type=str(payload.get("chunk_type") or ""),
            )
        )
    return rows


def collect_expected_chunk_ids(paths: list[Path]) -> list[str]:
    chunk_ids: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for case in payload.get("cases", []):
            if isinstance(case, dict):
                chunk_ids.extend(_string_list(case.get("expected_chunk_ids")))
    return sorted(set(chunk_ids))


def default_eval_paths(eval_dir: Path, *, include_templates: bool = False) -> list[Path]:
    paths = [path for path in sorted(eval_dir.glob("*.json")) if path.name != "baselines.json"]
    if include_templates:
        paths.extend(sorted((eval_dir / "templates").glob("*.json")))
    return paths


def normalize_name(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def _find_document(context: EvalAuditContext, reference: str) -> KnownDocument | None:
    value = str(reference or "").strip()
    if not value:
        return None
    return context.documents_by_id.get(value) or context.documents_by_name.get(normalize_name(value))


def _expected_documents(case: dict[str, Any]) -> list[str]:
    values = _string_list(case.get("expected_documents"))
    single = str(case.get("expected_document") or "").strip()
    if single:
        values.append(single)
    return _dedupe(values)


def _expected_pages(case: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    raw_pages = case.get("expected_pages")
    if isinstance(raw_pages, list):
        for value in raw_pages:
            parsed = _optional_int(value)
            if parsed is not None:
                pages.append(parsed)
    elif raw_pages is not None:
        parsed = _optional_int(raw_pages)
        if parsed is not None:
            pages.append(parsed)
    raw_page = case.get("expected_page")
    parsed_page = _optional_int(raw_page)
    if parsed_page is not None:
        pages.append(parsed_page)
    return sorted(set(pages))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return _dedupe([str(item).strip() for item in value if str(item).strip()])


def _all_terms(case: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for field in ["expected_keywords", "expected_claims", "expected_evidence_keywords"]:
        terms.extend(_string_list(case.get(field)))
    for field in ["expected_answer", "judge_rubric", "expected_relation"]:
        value = str(case.get(field) or "").strip()
        if value:
            terms.append(value)
    return terms


def _specific_terms(terms: list[str]) -> list[str]:
    return [term for term in _dedupe(terms) if _is_specific_term(term)]


def _is_specific_term(term: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(term).strip().lower())
    if not normalized or normalized in GENERIC_TERMS:
        return False
    if normalized in REFUSAL_MARKERS:
        return False
    if re.fullmatch(r"[\W_]+", normalized):
        return False
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", normalized))
    if has_cjk:
        return len(normalized) >= 2
    return len(re.sub(r"[^a-z0-9]", "", normalized)) >= 3


def _contains_refusal_marker(text: str) -> bool:
    normalized = str(text).strip().lower()
    return any(marker.lower() in normalized for marker in REFUSAL_MARKERS)


def _contains_any_marker(text: str, markers: list[str]) -> bool:
    for marker in markers:
        if re.fullmatch(r"[a-z]+", marker):
            if re.search(rf"\b{re.escape(marker)}\b", text):
                return True
        elif marker in text:
            return True
    return False


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    if isinstance(value, str):
        return bool(re.search(r"<[^>]+>", value))
    return False


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finding(
    severity: str,
    suite: str,
    case_id: str,
    code: str,
    message: str,
    field: str = "",
    value: Any = None,
) -> AuditFinding:
    return AuditFinding(
        severity=severity,
        suite=suite,
        case_id=case_id,
        code=code,
        message=message,
        field=field,
        value=value,
    )


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.paths] if args.paths else default_eval_paths(Path(args.eval_dir), include_templates=args.include_templates)
    context = EvalAuditContext()
    if args.local:
        expected_chunk_ids = collect_expected_chunk_ids(paths)
        context = load_local_audit_context(
            sqlite_path=Path(args.sqlite_path),
            chroma_dir=Path(args.chroma_dir),
            expected_chunk_ids=expected_chunk_ids,
        )
    report = audit_eval_paths(paths, context=context)
    payload = report.to_dict()
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_text_report(payload, max_findings=args.max_findings)
    if args.fail_on:
        threshold = SEVERITIES[args.fail_on]
        if any(SEVERITIES.get(item.severity, 0) >= threshold for item in report.findings):
            raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit eval suite quality.")
    parser.add_argument("paths", nargs="*", help="Eval JSON files to audit. Defaults to evals/*.json.")
    parser.add_argument("--eval-dir", default="evals", help="Directory containing eval JSON suites.")
    parser.add_argument("--include-templates", action="store_true", help="Also audit eval templates.")
    parser.add_argument("--local", action="store_true", help="Validate expected documents and gold chunks against local data.")
    parser.add_argument("--sqlite-path", default="data/assistant.sqlite3", help="Local metadata SQLite path.")
    parser.add_argument("--chroma-dir", default="data/chroma", help="Local Chroma directory.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--max-findings", type=int, default=80)
    parser.add_argument("--fail-on", choices=["error", "warning", "info"], default="")
    return parser.parse_args()


def print_text_report(payload: dict[str, Any], *, max_findings: int) -> None:
    summary = payload["summary"]
    print(
        "Eval audit: "
        f"{summary['suite_count']} suites, {summary['case_count']} cases, "
        f"{summary['error']} errors, {summary['warning']} warnings, {summary['info']} info."
    )
    print("\nSuites:")
    for suite in payload["suites"]:
        print(
            f"- {suite['suite']}: {suite['case_count']} cases, "
            f"{suite['refusal_case_count']} refusal, "
            f"{suite['gold_chunk_case_count']} gold-chunk, "
            f"{suite['vision_case_count']} visual, "
            f"{suite['table_case_count']} table, "
            f"{suite['ocr_case_count']} ocr"
        )
    print("\nFindings:")
    for finding in payload["findings"][:max_findings]:
        print(
            f"- [{finding['severity']}] {finding['suite']}::{finding['case_id']} "
            f"{finding['code']}: {finding['message']}"
        )
        if finding.get("value") is not None:
            print(f"  value: {json.dumps(finding['value'], ensure_ascii=False)}")
    remaining = len(payload["findings"]) - max_findings
    if remaining > 0:
        print(f"... {remaining} more findings hidden by --max-findings.")


if __name__ == "__main__":
    main()
