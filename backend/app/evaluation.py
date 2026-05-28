from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from backend.app.models import AskRequest, EvaluationCase, EvaluationResult, EvaluationRun
from backend.app.observability import ObservabilityClient, citation_ids_from_text, new_run_id, utc_now

if TYPE_CHECKING:
    from backend.app.agent import PaperAgentService


SMOKE_CHECK_NAMES = [
    "answer_completed",
    "evidence_present",
    "citation_linked",
    "document_match",
    "evidence_keywords",
    "answer_keywords",
    "refusal_when_expected",
    "embedding_not_fallback",
]

GOLD_CHECK_NAMES = [
    "answer_completed",
    "evidence_present",
    "citation_linked",
    "document_match",
    "retrieval_recall_at_k",
    "citation_support_rate",
    "answer_point_coverage",
    "refusal_accuracy",
    "embedding_not_fallback",
]


def load_eval_suite(path: Path) -> list[EvaluationCase]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        return []
    return [EvaluationCase(**item) for item in raw_cases if isinstance(item, dict)]


def run_eval_suite(
    *,
    suite_name: str,
    cases: list[EvaluationCase],
    agent: "PaperAgentService",
    document_ids: list[str],
    observer: ObservabilityClient | None = None,
    model_preset: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    top_k: int | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    audit_gold_evidence: bool = False,
) -> EvaluationRun:
    results: list[EvaluationResult] = []
    for case in cases:
        started = time.perf_counter()
        case_document_ids = document_ids_for_case(case=case, document_ids=document_ids, agent=agent)
        try:
            response = agent.ask(
                AskRequest(
                    question=case.question,
                    document_ids=case_document_ids,
                    model_preset=model_preset,
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                    top_k=top_k,
                )
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append(failed_eval_result(case=case, error=exc, latency_ms=elapsed_ms))
            continue

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        metrics = score_case(case=case, response=response)
        trace_summary = evaluation_trace_summary(response)
        trace_summary["case_metadata"] = eval_case_metadata(case)
        trace_summary["case_document_ids"] = case_document_ids
        trace_summary["checks"] = metrics["checks"]
        if audit_gold_evidence:
            trace_summary["gold_evidence_audit"] = audit_gold_evidence_stages(
                case=case,
                response=response,
                agent=agent,
            )
        results.append(
            EvaluationResult(
                case_id=case.id,
                question=case.question,
                answer=response.answer,
                evidence=evaluation_evidence_summary(response),
                trace_summary=trace_summary,
                evidence_count=int(metrics["evidence_count"]),
                citation_count=int(metrics["citation_count"]),
                valid_citation_count=int(metrics["valid_citation_count"]),
                evidence_keyword_hit_rate=float(metrics["evidence_keyword_hit_rate"]),
                evidence_document_hit=bool(metrics["document_hit"]),
                retrieval_hit=bool(metrics["has_evidence"]),
                citation_hit=bool(metrics["citation_hit"]),
                keyword_hit_rate=float(metrics["answer_keyword_hit_rate"]),
                context_precision=float(metrics.get("context_precision", metrics["evidence_keyword_hit_rate"])),
                context_recall=float(metrics.get("context_recall", metrics["evidence_keyword_hit_rate"])),
                document_coverage=float(metrics["document_coverage"]),
                citation_accuracy=float(metrics["citation_accuracy"]),
                embedding_used_fallback=bool(metrics["embedding_used_fallback"]),
                score=float(metrics["score"]),
                score_breakdown=dict(metrics["score_breakdown"]),
                result_status=str(metrics["status"]),
                failure_categories=list(metrics["failure_categories"]),
                grading_reasons=list(metrics["reasons"]),
                grading_report={
                    "status": metrics["status"],
                    "checks": metrics["checks"],
                    "reasons": metrics["reasons"],
                },
                latency_ms=elapsed_ms,
            )
        )

    total = max(len(results), 1)
    status_counts = Counter(result.result_status or "ungraded" for result in results)
    category_counts: Counter[str] = Counter()
    for result in results:
        category_counts.update(result.failure_categories)

    pass_count = status_counts.get("pass", 0)
    fail_count = len(results) - pass_count
    embedding_fallback_count = sum(1 for item in results if item.embedding_used_fallback)
    grading_summary = build_smoke_grading_summary(
        results=results,
        status_counts=status_counts,
        category_counts=category_counts,
    )

    run = EvaluationRun(
        run_id=new_run_id("eval"),
        suite_name=suite_name,
        created_at=utc_now(),
        document_ids=document_ids,
        case_count=len(results),
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_count / total,
        score_version=score_version_for_cases(cases),
        results=results,
        retrieval_hit_rate=sum(item.retrieval_hit for item in results) / total,
        citation_hit_rate=sum(item.citation_hit for item in results) / total,
        avg_keyword_hit_rate=sum(item.keyword_hit_rate for item in results) / total,
        avg_context_precision=sum(item.context_precision for item in results) / total,
        avg_context_recall=sum(item.context_recall for item in results) / total,
        avg_document_coverage=sum(item.document_coverage for item in results) / total,
        avg_citation_accuracy=sum(item.citation_accuracy for item in results) / total,
        embedding_fallback_count=embedding_fallback_count,
        embedding_fallback_rate=embedding_fallback_count / total,
        result_status_counts=dict(status_counts),
        failure_category_counts=dict(category_counts),
        grading_summary=grading_summary,
        evaluation_trustworthy=embedding_fallback_count == 0,
        experiment_metadata={
            "mode": eval_mode_for_cases(cases),
            "suite_name": suite_name,
            "case_count": len(cases),
            "checks": check_names_for_cases(cases),
            **(experiment_metadata or {}),
        },
        avg_score=sum(item.score for item in results) / total,
        avg_latency_ms=int(sum(item.latency_ms for item in results) / total),
    )
    if observer is not None:
        observer.record_eval_run(run)
    return run


def document_ids_for_case(
    *,
    case: EvaluationCase,
    document_ids: list[str],
    agent: "PaperAgentService",
) -> list[str]:
    expected_documents = expected_document_names(case)
    if not expected_documents:
        return document_ids
    matched_ids: list[str] = []
    store = getattr(agent, "store", None)
    for document_id in document_ids:
        document = store.get_document(document_id) if store is not None else None
        if document is None:
            continue
        if any(_document_info_matches(document, expected_document) for expected_document in expected_documents):
            matched_ids.append(document_id)
    return matched_ids or document_ids


def score_case(*, case: EvaluationCase, response: Any) -> dict[str, Any]:
    if is_gold_case(case):
        return score_gold_case(case=case, response=response)
    return score_smoke_case(case=case, response=response)


def is_gold_case(case: EvaluationCase) -> bool:
    return bool(case.gold_evidence or case.expected_answer_points or case.case_type or case.difficulty)


def score_smoke_case(*, case: EvaluationCase, response: Any) -> dict[str, Any]:
    answer = str(getattr(response, "answer", "") or "").strip()
    evidence = list(getattr(response, "evidence", []) or [])
    cited_ids = citation_ids_from_text(answer)
    evidence_by_citation = {item.citation_id: item for item in evidence if getattr(item, "citation_id", "")}
    valid_cited_ids = [citation_id for citation_id in cited_ids if citation_id in evidence_by_citation]
    cited_evidence = [evidence_by_citation[citation_id] for citation_id in valid_cited_ids]
    checked_evidence = cited_evidence or evidence

    expected_documents = expected_document_names(case)
    document_coverage = expected_document_coverage(
        expected_documents=expected_documents,
        required_document_count=case.required_document_count,
        evidence=evidence,
    )
    document_hit = document_coverage >= 1.0 if expected_documents or case.required_document_count else True

    answer_terms = clean_terms([*case.expected_keywords, *case.relation_keywords])
    evidence_terms = clean_terms(case.expected_evidence_keywords)
    answer_keyword_hit_rate = term_hit_rate(answer_terms, normalize_eval_text(answer)) if answer_terms else 1.0
    evidence_keyword_hit_rate = evidence_keyword_coverage(evidence_terms, checked_evidence)
    evidence_keyword_hit = evidence_keyword_hit_rate > 0.0 if evidence_terms else True

    expected_refusal = expected_refusal_case(case)
    refusal_hit = answer_has_refusal_signal(answer) if expected_refusal else True
    trace = getattr(response, "rag_trace", None)
    embedding_used_fallback = bool(getattr(trace, "embedding_used_fallback", False))
    has_answer = bool(answer)
    has_evidence = bool(evidence)
    citation_hit = bool(valid_cited_ids)
    citation_accuracy = len(valid_cited_ids) / max(len(cited_ids), 1) if cited_ids else 0.0

    checks = {
        "answer_completed": has_answer,
        "evidence_present": has_evidence,
        "citation_linked": citation_hit,
        "document_match": document_hit,
        "evidence_keywords": evidence_keyword_hit,
        "answer_keywords": answer_keyword_hit_rate > 0.0 if answer_terms else True,
        "refusal_when_expected": refusal_hit,
        "embedding_not_fallback": not embedding_used_fallback,
    }
    failures: list[tuple[str, str]] = []
    if not has_answer:
        failures.append(("answer_generation_failure", "没有生成可检查的回答。"))
    if not has_evidence:
        failures.append(("no_evidence", "回答没有返回证据。"))
    if not citation_hit:
        failures.append(("citation_missing", "回答没有引用可点击的证据编号。"))
    if not document_hit:
        failures.append(("wrong_document", "证据没有覆盖评测用例要求的文档。"))
    if not evidence_keyword_hit:
        failures.append(("evidence_keyword_missing", "被引用证据没有命中人工设定的关键依据词。"))
    if answer_terms and answer_keyword_hit_rate <= 0.0:
        failures.append(("answer_keyword_missing", "回答没有覆盖评测用例的基础关键词。"))
    if not refusal_hit:
        failures.append(("refusal_missing", "证据不足场景没有明确拒答或说明证据不足。"))
    if embedding_used_fallback:
        failures.append(("embedding_fallback", "本轮使用了本地备用检索，不能作为可信证据准确性结果。"))

    passed_checks = sum(1 for value in checks.values() if value)
    return {
        "status": "pass" if not failures else "fail",
        "evidence_count": len(evidence),
        "citation_count": len(cited_ids),
        "valid_citation_count": len(valid_cited_ids),
        "has_evidence": has_evidence,
        "citation_hit": citation_hit,
        "citation_accuracy": citation_accuracy,
        "document_hit": document_hit,
        "document_coverage": document_coverage,
        "answer_keyword_hit_rate": answer_keyword_hit_rate,
        "evidence_keyword_hit_rate": evidence_keyword_hit_rate,
        "embedding_used_fallback": embedding_used_fallback,
        "score": passed_checks / max(len(checks), 1),
        "checks": checks,
        "failure_categories": [category for category, _ in failures],
        "reasons": [reason for _, reason in failures],
        "score_breakdown": {key: 1.0 if value else 0.0 for key, value in checks.items()},
    }


def score_gold_case(*, case: EvaluationCase, response: Any) -> dict[str, Any]:
    answer = str(getattr(response, "answer", "") or "").strip()
    evidence = list(getattr(response, "evidence", []) or [])
    cited_ids = citation_ids_from_text(answer)
    evidence_by_citation = {item.citation_id: item for item in evidence if getattr(item, "citation_id", "")}
    valid_cited_ids = [citation_id for citation_id in cited_ids if citation_id in evidence_by_citation]
    cited_evidence = [evidence_by_citation[citation_id] for citation_id in valid_cited_ids]

    expected_documents = expected_document_names(case)
    document_coverage = expected_document_coverage(
        expected_documents=expected_documents,
        required_document_count=case.required_document_count,
        evidence=evidence,
    )
    document_hit = document_coverage >= 1.0 if expected_documents or case.required_document_count else True
    expected_refusal = expected_refusal_case(case)
    refusal_hit = answer_has_refusal_signal(answer) if expected_refusal else True
    trace = getattr(response, "rag_trace", None)
    embedding_used_fallback = bool(getattr(trace, "embedding_used_fallback", False))

    retrieval_recall = gold_evidence_recall(case.gold_evidence, evidence)
    citation_support = gold_citation_support_rate(case.gold_evidence, cited_evidence)
    answer_coverage = answer_point_coverage(case.expected_answer_points, answer)
    has_answer = bool(answer)
    has_evidence = bool(evidence)
    citation_hit = bool(valid_cited_ids)
    citation_accuracy = len(valid_cited_ids) / max(len(cited_ids), 1) if cited_ids else 0.0

    has_gold_evidence = bool(case.gold_evidence)
    has_answer_points = bool(case.expected_answer_points)
    checks = {
        "answer_completed": has_answer,
        "evidence_present": has_evidence,
        "citation_linked": citation_hit,
        "document_match": document_hit,
        "retrieval_recall_at_k": retrieval_recall >= 0.5 if has_gold_evidence else True,
        "citation_support_rate": citation_support >= 0.5 if has_gold_evidence else True,
        "answer_point_coverage": answer_coverage >= 0.67 if has_answer_points else True,
        "refusal_accuracy": refusal_hit,
        "embedding_not_fallback": not embedding_used_fallback,
    }
    failures: list[tuple[str, str]] = []
    if not has_answer:
        failures.append(("answer_generation_failure", "没有生成可检查的回答。"))
    if not has_evidence:
        failures.append(("no_evidence", "回答没有返回证据。"))
    if not citation_hit:
        failures.append(("citation_missing", "回答没有引用可点击的证据编号。"))
    if not document_hit:
        failures.append(("wrong_document", "证据没有覆盖评测用例要求的文档。"))
    if has_gold_evidence and retrieval_recall < 0.5:
        failures.append(("gold_evidence_missed", "检索结果没有覆盖足够的 gold evidence。"))
    if has_gold_evidence and citation_support < 0.5:
        failures.append(("citation_unsupported", "回答引用的证据与 gold evidence 支撑关系不足。"))
    if has_answer_points and answer_coverage < 0.67:
        failures.append(("answer_point_missing", "回答没有覆盖足够的 expected answer points。"))
    if not refusal_hit:
        failures.append(("refusal_missing", "证据不足场景没有明确拒答或说明证据不足。"))
    if embedding_used_fallback:
        failures.append(("embedding_fallback", "本轮使用了本地备用检索，不能作为可信证据准确性结果。"))

    passed_checks = sum(1 for value in checks.values() if value)
    return {
        "status": "pass" if not failures else "fail",
        "evidence_count": len(evidence),
        "citation_count": len(cited_ids),
        "valid_citation_count": len(valid_cited_ids),
        "has_evidence": has_evidence,
        "citation_hit": citation_hit,
        "citation_accuracy": citation_accuracy,
        "document_hit": document_hit,
        "document_coverage": document_coverage,
        "answer_keyword_hit_rate": answer_coverage,
        "evidence_keyword_hit_rate": retrieval_recall,
        "context_precision": citation_support,
        "context_recall": retrieval_recall,
        "embedding_used_fallback": embedding_used_fallback,
        "score": passed_checks / max(len(checks), 1),
        "checks": checks,
        "failure_categories": [category for category, _ in failures],
        "reasons": [reason for _, reason in failures],
        "score_breakdown": {
            **{key: 1.0 if value else 0.0 for key, value in checks.items()},
            "retrieval_recall_at_k": retrieval_recall,
            "citation_support_rate": citation_support,
            "answer_point_coverage": answer_coverage,
            "refusal_accuracy": 1.0 if refusal_hit else 0.0,
        },
    }


def gold_evidence_recall(gold_evidence: list[dict[str, Any]], evidence: Any) -> float:
    if not gold_evidence:
        return 1.0
    items = list(evidence or [])
    if not items:
        return 0.0
    hits = sum(1 for gold in gold_evidence if any(evidence_matches_gold(item, gold) for item in items))
    return hits / max(len(gold_evidence), 1)


def gold_citation_support_rate(gold_evidence: list[dict[str, Any]], cited_evidence: Any) -> float:
    if not gold_evidence:
        return 1.0
    items = list(cited_evidence or [])
    if not items:
        return 0.0
    supported = sum(1 for item in items if any(evidence_matches_gold(item, gold) for gold in gold_evidence))
    return supported / max(len(items), 1)


def audit_gold_evidence_stages(
    *,
    case: EvaluationCase,
    response: Any,
    agent: Any,
) -> dict[str, Any]:
    gold_items = list(case.gold_evidence or [])
    if not gold_items:
        return {}

    trace = getattr(response, "rag_trace", None)
    trace_rows = _quality_trace_rows(getattr(trace, "evidence_quality_trace", []) if trace is not None else [])
    candidate_items = _audit_items_from_trace_rows(rows=trace_rows, agent=agent)
    selected_items = [
        item
        for item in candidate_items
        if getattr(item, "selected_rank", None) is not None
        or str(getattr(item, "selection_status", "")) in {"selected_by_retrieval_filter", "selected_for_answer"}
    ]
    prompt_citation_ids = _prompt_citation_ids(getattr(trace, "final_prompt_evidence", []) if trace is not None else [])
    prompt_items = [
        item
        for item in selected_items
        if str(getattr(item, "citation_id", "") or "") in prompt_citation_ids
    ]
    visible_items = list(getattr(response, "evidence", []) or [])
    cited_ids = set(citation_ids_from_text(str(getattr(response, "answer", "") or "")))
    cited_items = [
        item
        for item in visible_items
        if str(getattr(item, "citation_id", "") or "") in cited_ids
    ]
    stages = {
        "candidate": candidate_items,
        "retrieval_selected": selected_items,
        "prompt": prompt_items,
        "visible": visible_items,
        "cited": cited_items,
    }

    gold_results: list[dict[str, Any]] = []
    for index, gold in enumerate(gold_items, start=1):
        stage_hits = {stage: any(evidence_matches_gold(item, gold) for item in items) for stage, items in stages.items()}
        first_stage = next((stage for stage in stages if stage_hits[stage]), "")
        last_stage = next((stage for stage in reversed(list(stages)) if stage_hits[stage]), "")
        best_item = _first_matching_audit_item(candidate_items, gold) or _first_matching_audit_item(visible_items, gold)
        gold_results.append(
            {
                "gold_index": index,
                "document": str(gold.get("document") or gold.get("document_name") or ""),
                "phrases": list(gold.get("text_contains") or gold.get("phrases") or []),
                "stage_hits": stage_hits,
                "first_hit_stage": first_stage,
                "last_hit_stage": last_stage,
                "matched_candidate": _audit_item_summary(best_item) if best_item is not None else {},
            }
        )

    summary = {
        stage: sum(1 for result in gold_results if result["stage_hits"].get(stage))
        for stage in stages
    }
    summary["gold_count"] = len(gold_results)
    return {
        "summary": summary,
        "gold": gold_results,
    }


def answer_point_coverage(points: list[str], answer: str) -> float:
    cleaned = clean_terms(points)
    if not cleaned:
        return 1.0
    normalized = normalize_eval_text(answer)
    hits = sum(1 for point in cleaned if alternative_term_hit(point, normalized))
    return hits / max(len(cleaned), 1)


def _quality_trace_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            normalized.append(dict(row))
        elif hasattr(row, "model_dump"):
            normalized.append(dict(row.model_dump()))
        else:
            normalized.append(dict(getattr(row, "__dict__", {}) or {}))
    return normalized


def _audit_items_from_trace_rows(*, rows: list[dict[str, Any]], agent: Any) -> list[Any]:
    return [_audit_item_from_trace_row(row=row, agent=agent) for row in rows]


def _audit_item_from_trace_row(*, row: dict[str, Any], agent: Any) -> Any:
    document_id = str(row.get("document_id") or "")
    chunk_id = str(row.get("chunk_id") or "")
    chunk = _lookup_chunk(agent=agent, document_id=document_id, chunk_id=chunk_id)
    metadata = dict(chunk.get("metadata") or {}) if chunk else {}
    text = str(chunk.get("text") or "") if chunk else str(row.get("quote") or "")
    return SimpleNamespace(
        citation_id=str(row.get("citation_id") or ""),
        chunk_id=chunk_id,
        document_id=document_id,
        paper_name=str(row.get("paper_name") or metadata.get("paper_name") or ""),
        page=int(row.get("page") or metadata.get("page") or 0),
        page_start=row.get("page_start") if row.get("page_start") is not None else metadata.get("page_start"),
        page_end=row.get("page_end") if row.get("page_end") is not None else metadata.get("page_end"),
        section=str(row.get("section") or metadata.get("section") or ""),
        quote=str(row.get("quote") or metadata.get("quote") or ""),
        text=text,
        candidate_rank=row.get("candidate_rank"),
        selected_rank=row.get("selected_rank"),
        selection_status=str(row.get("selection_status") or ""),
        score=float(row.get("score") or 0.0),
        score_source=str(row.get("score_source") or ""),
    )


def _lookup_chunk(*, agent: Any, document_id: str, chunk_id: str) -> dict[str, Any] | None:
    vector_store = getattr(agent, "vector_store", None)
    if vector_store is None or not document_id or not chunk_id:
        return None
    try:
        rows = vector_store.get_document_chunks(document_id, limit=1000)
    except Exception:
        return None
    for row in rows:
        metadata = row.get("metadata") or {}
        if str(row.get("id") or "") == chunk_id or str(metadata.get("chunk_id") or "") == chunk_id:
            return row
    return None


def _prompt_citation_ids(entries: Any) -> set[str]:
    ids: set[str] = set()
    for entry in list(entries or []):
        ids.update(citation_ids_from_text(str(entry)))
    return ids


def _first_matching_audit_item(items: list[Any], gold: dict[str, Any]) -> Any | None:
    return next((item for item in items if evidence_matches_gold(item, gold)), None)


def _audit_item_summary(item: Any) -> dict[str, Any]:
    return {
        "citation_id": str(getattr(item, "citation_id", "") or ""),
        "chunk_id": str(getattr(item, "chunk_id", "") or ""),
        "page": getattr(item, "page", 0),
        "candidate_rank": getattr(item, "candidate_rank", None),
        "selected_rank": getattr(item, "selected_rank", None),
        "selection_status": str(getattr(item, "selection_status", "") or ""),
        "score": getattr(item, "score", 0.0),
        "score_source": str(getattr(item, "score_source", "") or ""),
        "quote": _shorten(str(getattr(item, "quote", "") or getattr(item, "text", "") or ""), 240),
    }


def evidence_matches_gold(item: Any, gold: dict[str, Any]) -> bool:
    document = str(gold.get("document") or gold.get("document_name") or "").strip()
    if document and not _evidence_matches_document(item, document):
        return False
    page = gold.get("page")
    if page is not None and not evidence_matches_page(item, int(page)):
        return False
    phrases = gold.get("text_contains") or gold.get("phrases") or []
    if isinstance(phrases, str):
        phrases = [phrases]
    phrase_values = clean_terms([str(phrase) for phrase in phrases])
    if not phrase_values:
        return True
    combined = normalize_eval_text(
        "\n".join(
            [
                str(getattr(item, "paper_name", "") or ""),
                str(getattr(item, "section", "") or ""),
                str(getattr(item, "quote", "") or ""),
                str(getattr(item, "text", "") or ""),
            ]
        )
    )
    return all(alternative_term_hit(phrase, combined) for phrase in phrase_values)


def evidence_matches_page(item: Any, page: int) -> bool:
    page_start = int(getattr(item, "page_start", getattr(item, "page", 0)) or 0)
    page_end = int(getattr(item, "page_end", getattr(item, "page", 0)) or 0)
    if page_start and page_end:
        return page_start <= page <= page_end
    return int(getattr(item, "page", 0) or 0) == page


EVAL_CONCEPT_ALIASES = [
    (
        "class imbalance",
        "class imbalance",
        "class imbalanced",
        "类别不平衡",
        "类别失衡",
        "类不平衡",
        "前景背景不平衡",
        "foreground background imbalance",
    ),
    (
        "down-weights",
        "down weights",
        "down-weights",
        "downweight",
        "downweights",
        "reduce the weight",
        "reduces the weight",
        "lower the weight",
        "lowers the weight",
        "降低权重",
        "下调权重",
        "降低损失权重",
        "损失权重",
        "减小损失",
    ),
    (
        "well-classified examples",
        "well-classified examples",
        "well classified examples",
        "easy examples",
        "easy samples",
        "easy negatives",
        "易分类",
        "容易分类",
        "已正确分类",
        "高置信度样本",
    ),
    (
        "hard examples",
        "hard examples",
        "hard samples",
        "difficult examples",
        "difficult samples",
        "难分类",
        "困难样本",
        "难样本",
    ),
    (
        "one-stage detector",
        "one-stage detector",
        "one stage detector",
        "single-stage detector",
        "single stage detector",
        "单阶段检测器",
        "一阶段检测器",
    ),
    (
        "surpass the accuracy",
        "surpass the accuracy",
        "surpassing the accuracy",
        "higher accuracy",
        "outperform",
        "outperforms",
        "better accuracy",
        "准确率超过",
        "精度超过",
        "准确率更高",
        "超越",
    ),
    (
        "two-stage detectors",
        "two-stage detectors",
        "two stage detectors",
        "two-stage detector",
        "two stage detector",
        "双阶段检测器",
        "两阶段检测器",
        "二阶段检测器",
    ),
    (
        "maximizing agreement",
        "maximizing agreement",
        "maximize agreement",
        "maximizes agreement",
        "maximizing similarity",
        "maximize similarity",
        "最大化一致性",
        "最大化正对之间的一致性",
        "最大化同一数据不同增强视图之间的一致性",
        "最大化同一数据不同增强视图",
        "一致性最大化",
        "最大化相似",
    ),
    (
        "augmented views",
        "augmented views",
        "augmented view",
        "different augmented views",
        "transformed views",
        "增强视图",
        "不同增强视图",
        "不同的数据增强视图",
        "数据增强后的视图",
        "增强后的视图",
        "数据增强视图",
    ),
    (
        "contrastive loss",
        "contrastive loss",
        "contrastive objective",
        "对比损失",
        "对比学习损失",
        "对比目标",
    ),
    (
        "data augmentations",
        "data augmentations",
        "data augmentation",
        "augmentations",
        "数据增强",
        "增强策略",
    ),
    (
        "nonlinear transformation",
        "nonlinear transformation",
        "non-linear transformation",
        "nonlinear projection",
        "non-linear projection",
        "projection head",
        "非线性变换",
        "非线性投影",
        "投影头",
    ),
    (
        "larger batch sizes",
        "larger batch sizes",
        "large batch sizes",
        "larger batch size",
        "bigger batch",
        "large batch",
        "更大 batch",
        "更大的 batch",
        "更大批量",
        "批量大小",
    ),
    (
        "more training steps",
        "more training steps",
        "longer training",
        "train longer",
        "training longer",
        "更多训练步",
        "更多训练步骤",
        "训练更久",
        "更长训练",
    ),
    (
        "freezing",
        "freezing",
        "freeze",
        "freezes",
        "frozen",
        "固定",
        "冻结",
        "保持不变",
    ),
    (
        "pre-trained model weights",
        "pre-trained model weights",
        "pretrained model weights",
        "pre trained model weights",
        "pre-trained weights",
        "pretrained weights",
        "预训练模型权重",
        "预训练权重",
    ),
    (
        "rank decomposition matrices",
        "rank decomposition matrices",
        "rank-decomposition matrices",
        "low-rank matrices",
        "low rank matrices",
        "低秩矩阵",
        "秩分解矩阵",
        "低秩分解矩阵",
    ),
    (
        "trainable parameters",
        "trainable parameters",
        "trained parameters",
        "可训练参数",
        "训练参数",
    ),
    (
        "gpu memory",
        "gpu memory",
        "gpu memory requirement",
        "gpu memory usage",
        "显存",
        "gpu 显存",
    ),
    (
        "10,000 times",
        "10000 times",
        "10000x",
        "10,000x",
        "10000 倍",
        "一万倍",
    ),
    (
        "3 times",
        "3 times",
        "3x",
        "3 倍",
        "三倍",
    ),
    (
        "cannot prove",
        "cannot prove",
        "not enough evidence",
        "does not support",
        "cannot determine",
        "不能证明",
        "证据不足",
        "不支持",
        "无法证明",
    ),
    (
        "object detection",
        "object detection",
        "dense object detection",
        "目标检测",
        "密集目标检测",
    ),
    (
        "low-rank adaptation",
        "low-rank adaptation",
        "low rank adaptation",
        "低秩适配",
        "低秩自适应",
    ),
    (
        "language model",
        "language model",
        "language models",
        "语言模型",
    ),
]


def alternative_term_hit(term: str, normalized_text: str) -> bool:
    alternatives = [part.strip() for part in str(term).split("|") if part.strip()]
    normalized_alternatives = [normalize_eval_text(part) for part in alternatives]
    if any(part and part in normalized_text for part in normalized_alternatives):
        return True
    for alias_group in EVAL_CONCEPT_ALIASES:
        normalized_group = [normalize_eval_text(alias) for alias in alias_group]
        if not any(part in normalized_group for part in normalized_alternatives):
            continue
        if any(alias and alias in normalized_text for alias in normalized_group):
            return True
    return False


def evidence_keyword_coverage(terms: list[str], evidence: Any) -> float:
    cleaned = clean_terms(terms)
    if not cleaned:
        return 1.0
    items = list(evidence or [])
    if not items:
        return 0.0
    combined = normalize_eval_text(
        "\n".join(
            "\n".join(
                [
                    str(getattr(item, "paper_name", "") or ""),
                    str(getattr(item, "section", "") or ""),
                    str(getattr(item, "quote", "") or ""),
                    str(getattr(item, "text", "") or ""),
                ]
            )
            for item in items
        )
    )
    return term_hit_rate(cleaned, combined)


def answer_has_refusal_signal(answer: str) -> bool:
    normalized = normalize_eval_text(answer)
    markers = [
        "证据不足",
        "无法证明",
        "不能证明",
        "没有证据",
        "未找到",
        "不支持",
        "不能得出",
        "无法判断",
        "not enough evidence",
        "cannot determine",
        "does not support",
        "no evidence",
    ]
    return any(normalize_eval_text(marker) in normalized for marker in markers)


def eval_case_metadata(case: EvaluationCase) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "case_type": case.case_type,
        "difficulty": case.difficulty,
        "tags": case.tags,
        "parser_sensitive": case.parser_sensitive,
        "expected_document": case.expected_document,
        "expected_documents": case.expected_documents,
        "required_document_count": case.required_document_count,
        "expected_keywords": case.expected_keywords,
        "expected_evidence_keywords": case.expected_evidence_keywords,
        "gold_evidence": case.gold_evidence,
        "expected_answer_points": case.expected_answer_points,
        "relation_keywords": case.relation_keywords,
        "expected_refusal": case.expected_refusal,
    }


def smoke_case_metadata(case: EvaluationCase) -> dict[str, Any]:
    return eval_case_metadata(case)


def build_smoke_grading_summary(
    *,
    results: list[EvaluationResult],
    status_counts: Counter[str],
    category_counts: Counter[str],
) -> dict[str, Any]:
    total = max(len(results), 1)
    pass_count = status_counts.get("pass", 0)
    fallback_count = sum(1 for result in results if result.embedding_used_fallback)
    return {
        "mode": eval_mode_for_results(results),
        "status_counts": dict(status_counts),
        "failure_category_counts": dict(category_counts),
        "pass_count": pass_count,
        "fail_count": len(results) - pass_count,
        "pass_rate": pass_count / total,
        "embedding_fallback_count": fallback_count,
        "checks": check_names_for_results(results),
        "avg_retrieval_recall_at_k": sum(result.context_recall for result in results) / total,
        "avg_citation_support_rate": sum(result.context_precision for result in results) / total,
        "avg_answer_point_coverage": sum(result.keyword_hit_rate for result in results) / total,
        "top_failure_categories": [
            {"category": category, "count": count}
            for category, count in category_counts.most_common(6)
        ],
    }


def failed_eval_result(*, case: EvaluationCase, error: Exception, latency_ms: int) -> EvaluationResult:
    message = str(error).strip() or error.__class__.__name__
    return EvaluationResult(
        case_id=case.id,
        question=case.question,
        answer="",
        error=message[:1000],
        evidence=[],
        trace_summary={
            "status": "blocked",
            "error": message[:1000],
            "case_metadata": eval_case_metadata(case),
            "checks": {
                "answer_completed": False,
                "evidence_present": False,
                "citation_linked": False,
                "document_match": False,
                "evidence_keywords": False,
                "answer_keywords": False,
                "refusal_when_expected": False,
                "embedding_not_fallback": True,
            },
        },
        evidence_count=0,
        citation_count=0,
        valid_citation_count=0,
        evidence_keyword_hit_rate=0.0,
        evidence_document_hit=False,
        retrieval_hit=False,
        citation_hit=False,
        keyword_hit_rate=0.0,
        context_precision=0.0,
        context_recall=0.0,
        document_coverage=0.0,
        citation_accuracy=0.0,
        embedding_used_fallback=False,
        score=0.0,
        score_breakdown={"answer_completed": 0.0},
        result_status="blocked",
        failure_categories=["answer_generation_failure"],
        grading_reasons=[f"评测用例执行失败：{message[:300]}"],
        grading_report={
            "status": "blocked",
            "reasons": [f"评测用例执行失败：{message[:300]}"],
        },
        latency_ms=latency_ms,
    )


def evaluation_evidence_summary(response: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in list(getattr(response, "evidence", []) or []):
        summary.append(
            {
                "citation_id": getattr(item, "citation_id", ""),
                "chunk_id": getattr(item, "chunk_id", ""),
                "document_id": getattr(item, "document_id", ""),
                "paper_name": getattr(item, "paper_name", ""),
                "page": getattr(item, "page", 0),
                "page_start": getattr(item, "page_start", None),
                "page_end": getattr(item, "page_end", None),
                "section": getattr(item, "section", None),
                "source": getattr(item, "source", ""),
                "score": getattr(item, "score", 0.0),
                "chunk_type": getattr(item, "chunk_type", "text"),
                "image_id": getattr(item, "image_id", None),
                "quote": _shorten(str(getattr(item, "quote", "") or getattr(item, "text", "") or ""), 600),
                "quality_label": getattr(item, "quality_label", ""),
                "selection_status": getattr(item, "selection_status", ""),
            }
        )
    return summary


def evaluation_trace_summary(response: Any) -> dict[str, Any]:
    trace = getattr(response, "rag_trace", None)
    if trace is None:
        return {}
    return {
        "retrieval_strategy": getattr(trace, "retrieval_strategy", ""),
        "retrieval_pipeline": getattr(trace, "retrieval_pipeline", ""),
        "ranking_method": getattr(trace, "ranking_method", ""),
        "retrieved_count": getattr(trace, "retrieved_count", 0),
        "top_k": getattr(trace, "top_k", 0),
        "final_prompt_evidence": _jsonable(getattr(trace, "final_prompt_evidence", [])),
        "embedding_requested_model": getattr(trace, "embedding_requested_model", ""),
        "embedding_provider": getattr(trace, "embedding_provider", ""),
        "embedding_used_fallback": bool(getattr(trace, "embedding_used_fallback", False)),
        "embedding_fallback_reason": getattr(trace, "embedding_fallback_reason", ""),
        "answer_strategy": getattr(trace, "answer_strategy", ""),
        "fallback_used": bool(getattr(trace, "fallback_used", False)),
        "evidence_quality": getattr(trace, "evidence_quality", ""),
        "evidence_coverage": _jsonable(getattr(trace, "evidence_coverage", {})),
        "verification": _jsonable(getattr(trace, "verification", {})),
        "multi_document_coverage": _jsonable(getattr(trace, "multi_document_coverage", {})),
    }


def score_version_for_cases(cases: list[EvaluationCase]) -> str:
    return "gold-rag-v1" if any(is_gold_case(case) for case in cases) else "smoke-evidence-v1"


def eval_mode_for_cases(cases: list[EvaluationCase]) -> str:
    return "gold_rag_benchmark" if any(is_gold_case(case) for case in cases) else "basic_smoke_evidence_check"


def check_names_for_cases(cases: list[EvaluationCase]) -> list[str]:
    return GOLD_CHECK_NAMES if any(is_gold_case(case) for case in cases) else SMOKE_CHECK_NAMES


def eval_mode_for_results(results: list[EvaluationResult]) -> str:
    for result in results:
        metadata = result.trace_summary.get("case_metadata", {})
        if metadata.get("gold_evidence") or metadata.get("expected_answer_points"):
            return "gold_rag_benchmark"
    return "basic_smoke_evidence_check"


def check_names_for_results(results: list[EvaluationResult]) -> list[str]:
    return GOLD_CHECK_NAMES if eval_mode_for_results(results) == "gold_rag_benchmark" else SMOKE_CHECK_NAMES


def expected_refusal_case(case: EvaluationCase) -> bool:
    if case.expected_refusal is not None:
        return bool(case.expected_refusal)
    normalized = normalize_eval_text(case.question)
    markers = ["证据不足", "无法证明", "没有证据", "无关结论", "unsupported", "not enough evidence"]
    return any(normalize_eval_text(marker) in normalized for marker in markers)


def expected_document_names(case: EvaluationCase) -> list[str]:
    values = [case.expected_document or "", *case.expected_documents]
    return clean_terms(values)


def expected_document_coverage(
    *,
    expected_documents: list[str],
    required_document_count: int | None,
    evidence: Any,
) -> float:
    items = list(evidence or [])
    if not expected_documents and not required_document_count:
        return 1.0
    if not items:
        return 0.0
    if expected_documents:
        hits = sum(1 for document in expected_documents if any(_evidence_matches_document(item, document) for item in items))
        return hits / max(len(expected_documents), 1)

    distinct_documents = {
        str(getattr(item, "document_id", "") or getattr(item, "paper_name", "") or getattr(item, "source", "") or "")
        for item in items
    }
    distinct_documents.discard("")
    required = max(int(required_document_count or 0), 1)
    return min(len(distinct_documents) / required, 1.0)


PDF_TEXT_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00ad": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}


def normalize_eval_text(text: str) -> str:
    normalized = str(text)
    for source, target in PDF_TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def term_hit_rate(terms: list[str], text: str) -> float:
    cleaned = clean_terms(terms)
    if not cleaned:
        return 1.0
    normalized = normalize_eval_text(text)
    hits = sum(1 for term in cleaned if normalize_eval_text(term) in normalized)
    return hits / max(len(cleaned), 1)


def clean_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    for term in terms:
        value = str(term).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _evidence_matches_document(item: Any, expected_document: str) -> bool:
    expected_values = _document_match_values(expected_document)
    candidate_values: set[str] = set()
    for value in [
        getattr(item, "document_id", ""),
        getattr(item, "paper_name", ""),
        getattr(item, "source", ""),
    ]:
        candidate_values.update(_document_match_values(str(value or "")))
    return bool(expected_values & candidate_values)


def _document_info_matches(document: Any, expected_document: str) -> bool:
    expected_values = _document_match_values(expected_document)
    candidate_values: set[str] = set()
    for value in [
        getattr(document, "id", ""),
        getattr(document, "file_name", ""),
        getattr(document, "source_path", ""),
    ]:
        candidate_values.update(_document_match_values(str(value or "")))
    return bool(expected_values & candidate_values)


def _document_match_values(value: str) -> set[str]:
    stripped = str(value or "").strip()
    if not stripped:
        return set()
    basename = Path(stripped).name
    return {normalize_eval_text(stripped), normalize_eval_text(basename)}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return str(value)


def _shorten(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)].rstrip() + "..."
