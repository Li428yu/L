from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.eval_grading import apply_eval_result_grading, build_eval_run_grading_summary
from backend.app.models import AskRequest, EvaluationCase, EvaluationResult, EvaluationRun
from backend.app.observability import ObservabilityClient, new_run_id, utc_now

if TYPE_CHECKING:
    from backend.app.agent import PaperAgentService


JUDGE_SYSTEM_PROMPT = (
    "Strict RAG judge. Score only from provided JSON; unsupported plausible claims get low scores. "
    "Return JSON only."
)

JUDGE_SCORE_KEYS = [
    "answer_relevance",
    "faithfulness",
    "citation_support",
    "context_usage",
    "multi_document_clarity",
    "visual_grounding",
    "claim_coverage",
    "refusal_correctness",
    "completeness",
    "no_hallucination",
]


def load_eval_suite(path: Path) -> list[EvaluationCase]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase(**item) for item in payload.get("cases", [])]


def run_eval_suite(
    *,
    suite_name: str,
    cases: list[EvaluationCase],
    agent: "PaperAgentService",
    document_ids: list[str],
    observer: ObservabilityClient | None = None,
    enable_judge: bool | None = None,
    model_preset: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    top_k: int | None = None,
    experiment_metadata: dict[str, Any] | None = None,
) -> EvaluationRun:
    results: list[EvaluationResult] = []
    for case in cases:
        started = time.perf_counter()
        try:
            response = agent.ask(
                AskRequest(
                    question=case.question,
                    document_ids=document_ids,
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
        metrics = score_eval_case(case=case, response=response)
        judge = score_with_llm_judge(
            case=case,
            response=response,
            agent=agent,
            enabled_override=enable_judge,
        )
        combined_score = combine_eval_scores(programmatic_score=float(metrics["score"]), judge=judge)
        final_score_cap = float(metrics["score_cap"])
        final_score = min(combined_score, final_score_cap)
        score_breakdown = dict(metrics["score_breakdown"])
        if judge["used"]:
            score_breakdown["llm_judge"] = float(judge["score"])
        score_breakdown["score_after_judge_before_caps"] = combined_score
        score_breakdown["final_score_cap"] = final_score_cap
        score_breakdown["final_score_capped"] = 1.0 if final_score < combined_score else 0.0
        trace_summary = evaluation_trace_summary(response)
        trace_summary["case_metadata"] = case_metadata(case)
        trace_summary["gold_evidence"] = gold_evidence_summary(case=case, metrics=metrics)
        trace_summary["citation_pages"] = metrics["citation_page_diagnostics"]
        results.append(
            apply_eval_result_grading(
                EvaluationResult(
                    case_id=case.id,
                    question=case.question,
                    answer=response.answer,
                    evidence=evaluation_evidence_summary(response),
                    trace_summary=trace_summary,
                    retrieval_hit=bool(metrics["retrieval_hit"]),
                    citation_hit=bool(metrics["citation_hit"]),
                    keyword_hit_rate=float(metrics["keyword_hit_rate"]),
                    context_precision=float(metrics["context_precision"]),
                    context_recall=float(metrics["context_recall"]),
                    gold_chunk_count=int(metrics["gold_chunk_count"]),
                    gold_chunk_recall_at_1=float(metrics["gold_chunk_recall_at_1"]),
                    gold_chunk_recall_at_3=float(metrics["gold_chunk_recall_at_3"]),
                    gold_chunk_recall_at_5=float(metrics["gold_chunk_recall_at_5"]),
                    gold_chunk_recall_at_k=float(metrics["gold_chunk_recall_at_k"]),
                    gold_chunk_hit_ids=list(metrics["gold_chunk_hit_ids"]),
                    gold_chunk_missed_ids=list(metrics["gold_chunk_missed_ids"]),
                    gold_chunk_candidate_recall_at_k=float(metrics["gold_chunk_candidate_recall_at_k"]),
                    gold_chunk_candidate_hit_ids=list(metrics["gold_chunk_candidate_hit_ids"]),
                    gold_chunk_candidate_missed_ids=list(metrics["gold_chunk_candidate_missed_ids"]),
                    document_coverage=float(metrics["document_coverage"]),
                    image_evidence_hit=bool(metrics["image_evidence_hit"]),
                    visual_evidence_hit=bool(metrics["visual_evidence_hit"]),
                    visual_summary_hit=bool(metrics["visual_summary_hit"]),
                    table_evidence_hit=bool(metrics["table_evidence_hit"]),
                    ocr_evidence_hit=bool(metrics["ocr_evidence_hit"]),
                    ocr_text_hit=bool(metrics["ocr_text_hit"]),
                    citation_accuracy=float(metrics["citation_accuracy"]),
                    answer_relevance=float(metrics["answer_relevance"]),
                    faithfulness_proxy=float(metrics["faithfulness_proxy"]),
                    claim_hit_rate=float(metrics["claim_hit_rate"]),
                    forbidden_claim_rate=float(metrics["forbidden_claim_rate"]),
                    refusal_correctness=float(metrics["refusal_correctness"]),
                    relation_hit=float(metrics["relation_hit"]),
                    visual_warning_count=int(metrics["visual_warning_count"]),
                    embedding_used_fallback=bool(response.rag_trace.embedding_used_fallback),
                    judge_used=bool(judge["used"]),
                    judge_score=float(judge["score"]),
                    judge_scores=dict(judge["scores"]),
                    judge_reason=str(judge["reason"]),
                    score=final_score,
                    score_breakdown=score_breakdown,
                    latency_ms=elapsed_ms,
                )
            )
        )

    total = max(len(results), 1)
    judge_used_count = sum(item.judge_used for item in results)
    gold_chunk_results = [item for item in results if item.gold_chunk_count > 0]
    gold_chunk_total = max(len(gold_chunk_results), 1)
    segment_metrics = build_segment_metrics(cases=cases, results=results)
    grading_summary = build_eval_run_grading_summary(results)
    run = EvaluationRun(
        run_id=new_run_id("eval"),
        suite_name=suite_name,
        created_at=utc_now(),
        document_ids=document_ids,
        case_count=len(results),
        judge_enabled=bool(enable_judge if enable_judge is not None else getattr(agent.settings, "enable_llm_judge", False)),
        score_version="rag-eval-v5-trust-gate",
        results=results,
        retrieval_hit_rate=sum(item.retrieval_hit for item in results) / total,
        citation_hit_rate=sum(item.citation_hit for item in results) / total,
        avg_keyword_hit_rate=sum(item.keyword_hit_rate for item in results) / total,
        avg_context_precision=sum(item.context_precision for item in results) / total,
        avg_context_recall=sum(item.context_recall for item in results) / total,
        gold_chunk_case_count=len(gold_chunk_results),
        avg_gold_chunk_recall_at_1=sum(item.gold_chunk_recall_at_1 for item in gold_chunk_results) / gold_chunk_total,
        avg_gold_chunk_recall_at_3=sum(item.gold_chunk_recall_at_3 for item in gold_chunk_results) / gold_chunk_total,
        avg_gold_chunk_recall_at_5=sum(item.gold_chunk_recall_at_5 for item in gold_chunk_results) / gold_chunk_total,
        avg_gold_chunk_recall_at_k=sum(item.gold_chunk_recall_at_k for item in gold_chunk_results) / gold_chunk_total,
        avg_gold_chunk_candidate_recall_at_k=sum(item.gold_chunk_candidate_recall_at_k for item in gold_chunk_results) / gold_chunk_total,
        avg_document_coverage=sum(item.document_coverage for item in results) / total,
        avg_image_evidence_hit_rate=sum(item.image_evidence_hit for item in results) / total,
        avg_visual_evidence_hit_rate=sum(item.visual_evidence_hit for item in results) / total,
        avg_visual_summary_hit_rate=sum(item.visual_summary_hit for item in results) / total,
        avg_table_evidence_hit_rate=sum(item.table_evidence_hit for item in results) / total,
        avg_ocr_evidence_hit_rate=sum(item.ocr_evidence_hit for item in results) / total,
        avg_ocr_text_hit_rate=sum(item.ocr_text_hit for item in results) / total,
        avg_citation_accuracy=sum(item.citation_accuracy for item in results) / total,
        avg_answer_relevance=sum(item.answer_relevance for item in results) / total,
        avg_faithfulness_proxy=sum(item.faithfulness_proxy for item in results) / total,
        avg_claim_hit_rate=sum(item.claim_hit_rate for item in results) / total,
        avg_forbidden_claim_rate=sum(item.forbidden_claim_rate for item in results) / total,
        avg_refusal_correctness=sum(item.refusal_correctness for item in results) / total,
        avg_relation_hit=sum(item.relation_hit for item in results) / total,
        avg_visual_warning_count=sum(item.visual_warning_count for item in results) / total,
        embedding_fallback_count=sum(item.embedding_used_fallback for item in results),
        embedding_fallback_rate=sum(item.embedding_used_fallback for item in results) / total,
        avg_judge_score=(
            sum(item.judge_score for item in results if item.judge_used) / max(judge_used_count, 1)
            if judge_used_count
            else 0.0
        ),
        judge_coverage=judge_used_count / total,
        segment_metrics=segment_metrics,
        experiment_metadata={
            "experiment_type": "rag_regression",
            "dataset_name": suite_name,
            "dataset_item_count": len(cases),
            "judge_mode": (
                "enabled"
                if bool(enable_judge if enable_judge is not None else getattr(agent.settings, "enable_llm_judge", False))
                else "programmatic_only"
            ),
            "score_version": "rag-eval-v5-trust-gate",
            "metric_groups": [
                "retrieval",
                "gold_chunk_recall",
                "gold_evidence",
                "grading",
                "citation",
                "claim",
                "refusal",
                "relation",
                "visual",
                "table",
                "ocr",
                "embedding_fallback",
                "judge",
            ],
            **(experiment_metadata or {}),
        },
        result_status_counts=dict(grading_summary["status_counts"]),
        failure_category_counts=dict(grading_summary["failure_category_counts"]),
        grading_summary=grading_summary,
        evaluation_trustworthy=bool(grading_summary["evaluation_trustworthy"]),
        trust_gate_status=str(grading_summary["trust_gate_status"]),
        trust_gate_failures=list(grading_summary["trust_gate_failures"]),
        avg_score=sum(item.score for item in results) / total,
        avg_latency_ms=int(sum(item.latency_ms for item in results) / total),
    )
    if observer is not None:
        observer.record_eval_run(run)
    return run


def failed_eval_result(*, case: EvaluationCase, error: Exception, latency_ms: int) -> EvaluationResult:
    message = str(error).strip() or error.__class__.__name__
    expected_chunk_ids = clean_terms(case.expected_chunk_ids)
    return apply_eval_result_grading(
        EvaluationResult(
            case_id=case.id,
            question=case.question,
            answer="",
            error=message[:1000],
            evidence=[],
            trace_summary={
                "status": "failed",
                "error": message[:1000],
                "case_metadata": case_metadata(case),
                "gold_evidence": {
                    "status": "failed_before_answer",
                    "expected_chunk_ids": expected_chunk_ids,
                    "answer_evidence": {
                        "hit_chunk_ids": [],
                        "missed_chunk_ids": expected_chunk_ids,
                        "recall_at_k": 0.0,
                    },
                    "candidate_trace": {
                        "hit_chunk_ids": [],
                        "missed_chunk_ids": expected_chunk_ids,
                        "recall_at_k": 0.0,
                    },
                },
            },
            retrieval_hit=False,
            citation_hit=False,
            keyword_hit_rate=0.0,
            context_precision=0.0,
            context_recall=0.0,
            gold_chunk_count=len(expected_chunk_ids),
            gold_chunk_recall_at_1=0.0,
            gold_chunk_recall_at_3=0.0,
            gold_chunk_recall_at_5=0.0,
            gold_chunk_recall_at_k=0.0,
            gold_chunk_hit_ids=[],
            gold_chunk_missed_ids=expected_chunk_ids,
            gold_chunk_candidate_recall_at_k=0.0,
            gold_chunk_candidate_hit_ids=[],
            gold_chunk_candidate_missed_ids=expected_chunk_ids,
            document_coverage=0.0,
            image_evidence_hit=False,
            visual_evidence_hit=False,
            visual_summary_hit=False,
            table_evidence_hit=False,
            ocr_evidence_hit=False,
            ocr_text_hit=False,
            citation_accuracy=0.0,
            answer_relevance=0.0,
            faithfulness_proxy=0.0,
            claim_hit_rate=0.0,
            forbidden_claim_rate=1.0,
            refusal_correctness=0.0,
            relation_hit=0.0,
            visual_warning_count=0,
            embedding_used_fallback=False,
            judge_used=False,
            judge_score=0.0,
            judge_scores={},
            judge_reason=f"case failed before judging: {message[:700]}",
            score=0.0,
            score_breakdown={"answer_generation_failed": 1.0},
            latency_ms=latency_ms,
        )
    )


def case_metadata(case: EvaluationCase) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "tags": case_tags(case),
        "expected_document": case.expected_document,
        "expected_documents": case.expected_documents,
        "expected_page": case.expected_page,
        "expected_pages": case.expected_pages,
        "expected_modalities": case.expected_modalities,
        "expected_chunk_ids": case.expected_chunk_ids,
        "required_document_count": case.required_document_count,
        "expected_claims": case.expected_claims,
        "forbidden_claims": case.forbidden_claims,
        "expected_relation": case.expected_relation,
        "expected_refusal": case.expected_refusal,
        "expected_answer": case.expected_answer,
        "expected_keywords": case.expected_keywords,
        "expected_evidence_keywords": case.expected_evidence_keywords,
        "relation_keywords": case.relation_keywords,
        "judge_rubric": case.judge_rubric,
    }


def case_tags(case: EvaluationCase) -> list[str]:
    tags: list[str] = []
    modalities = {str(item).strip().lower() for item in case.expected_modalities}
    if "table" in modalities:
        tags.append("table")
    if modalities & {"image", "figure", "chart"}:
        tags.append("image")
    if "vision" in modalities:
        tags.append("vision")
    if "ocr" in modalities:
        tags.append("ocr")
    if case.expected_documents or case.required_document_count:
        tags.append("multi_document")
    if case.expected_relation or case.relation_keywords:
        tags.append("relation")
    if case.expected_claims or case.forbidden_claims:
        tags.append("claim")
    if expected_refusal_case(case):
        tags.append("refusal")
    if not any(tag in tags for tag in ["table", "image", "vision", "ocr"]):
        tags.append("text")
    return tags


def expected_refusal_case(case: EvaluationCase) -> bool:
    if case.expected_refusal is not None:
        return bool(case.expected_refusal)
    text = f"{case.expected_answer}\n{case.judge_rubric}\n{case.question}"
    markers = ["拒绝", "拒答", "不能证明", "无证据", "证据不足", "不得", "没有直接证据"]
    return any(marker in text for marker in markers)


def build_segment_metrics(*, cases: list[EvaluationCase], results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    case_by_id = {case.id: case for case in cases}
    tagged_results: dict[str, list[EvaluationResult]] = {}
    for result in results:
        case = case_by_id.get(result.case_id)
        if not case:
            continue
        for tag in case_tags(case):
            tagged_results.setdefault(tag, []).append(result)

    return {
        tag: aggregate_segment_metrics(items)
        for tag, items in sorted(tagged_results.items())
        if items
    }


def aggregate_segment_metrics(items: list[EvaluationResult]) -> dict[str, float]:
    total = max(len(items), 1)
    judged = [item for item in items if item.judge_used]
    gold_chunk_items = [item for item in items if item.gold_chunk_count > 0]
    gold_chunk_total = max(len(gold_chunk_items), 1)
    return {
        "count": float(len(items)),
        "avg_score": sum(item.score for item in items) / total,
        "retrieval_hit_rate": sum(item.retrieval_hit for item in items) / total,
        "citation_hit_rate": sum(item.citation_hit for item in items) / total,
        "avg_context_precision": sum(item.context_precision for item in items) / total,
        "avg_context_recall": sum(item.context_recall for item in items) / total,
        "gold_chunk_case_count": float(len(gold_chunk_items)),
        "avg_gold_chunk_recall_at_1": sum(item.gold_chunk_recall_at_1 for item in gold_chunk_items) / gold_chunk_total,
        "avg_gold_chunk_recall_at_3": sum(item.gold_chunk_recall_at_3 for item in gold_chunk_items) / gold_chunk_total,
        "avg_gold_chunk_recall_at_5": sum(item.gold_chunk_recall_at_5 for item in gold_chunk_items) / gold_chunk_total,
        "avg_gold_chunk_recall_at_k": sum(item.gold_chunk_recall_at_k for item in gold_chunk_items) / gold_chunk_total,
        "avg_gold_chunk_candidate_recall_at_k": sum(item.gold_chunk_candidate_recall_at_k for item in gold_chunk_items) / gold_chunk_total,
        "avg_document_coverage": sum(item.document_coverage for item in items) / total,
        "avg_citation_accuracy": sum(item.citation_accuracy for item in items) / total,
        "avg_image_evidence_hit_rate": sum(item.image_evidence_hit for item in items) / total,
        "avg_visual_evidence_hit_rate": sum(item.visual_evidence_hit for item in items) / total,
        "avg_visual_summary_hit_rate": sum(item.visual_summary_hit for item in items) / total,
        "avg_table_evidence_hit_rate": sum(item.table_evidence_hit for item in items) / total,
        "avg_ocr_evidence_hit_rate": sum(item.ocr_evidence_hit for item in items) / total,
        "avg_ocr_text_hit_rate": sum(item.ocr_text_hit for item in items) / total,
        "avg_claim_hit_rate": sum(item.claim_hit_rate for item in items) / total,
        "avg_forbidden_claim_rate": sum(item.forbidden_claim_rate for item in items) / total,
        "avg_refusal_correctness": sum(item.refusal_correctness for item in items) / total,
        "avg_relation_hit": sum(item.relation_hit for item in items) / total,
        "avg_visual_warning_count": sum(item.visual_warning_count for item in items) / total,
        "embedding_fallback_count": float(sum(item.embedding_used_fallback for item in items)),
        "embedding_fallback_rate": sum(item.embedding_used_fallback for item in items) / total,
        "judge_coverage": len(judged) / total,
        "avg_judge_score": (
            sum(item.judge_score for item in judged) / max(len(judged), 1)
            if judged
            else 0.0
        ),
        "avg_latency_ms": sum(item.latency_ms for item in items) / total,
    }


def score_eval_case(*, case: EvaluationCase, response) -> dict[str, Any]:
    answer = response.answer
    answer_lower = answer.lower()
    evidence_text = "\n".join(
        f"{item.paper_name}\n{item.section or ''}\n{item.quote}\n{item.text}"
        for item in response.evidence
    )
    evidence_lower = evidence_text.lower()

    keyword_terms = clean_terms([*case.expected_keywords, *case.relation_keywords])
    keyword_hit_rate = term_hit_rate(keyword_terms, answer_lower)

    expected_documents = expected_document_names(case)
    document_coverage = expected_document_coverage(
        expected_documents=expected_documents,
        required_document_count=case.required_document_count,
        evidence=response.evidence,
    )
    retrieval_hit = document_coverage >= 1.0 if expected_documents or case.required_document_count else True

    expected_pages = expected_page_numbers(case)
    citation_hit_rate = expected_page_hit_rate(expected_pages=expected_pages, evidence=response.evidence)
    citation_hit = citation_hit_rate >= 1.0 if expected_pages else True
    citation_page_diagnostics = citation_page_diagnostics_for_eval(
        expected_pages=expected_pages,
        evidence=response.evidence,
    )

    evidence_terms = clean_terms(case.expected_evidence_keywords or case.expected_keywords)
    context_precision = context_precision_proxy(terms=evidence_terms, evidence=response.evidence)
    context_recall = term_hit_rate(evidence_terms, evidence_lower)
    gold_chunk_ids = clean_terms(case.expected_chunk_ids)
    gold_chunk_count = len(gold_chunk_ids)
    recall_k = int(getattr(response.rag_trace, "top_k", 0) or len(response.evidence) or 1)
    gold_chunk_answer_at_1 = gold_chunk_coverage(gold_chunk_ids=gold_chunk_ids, items=response.evidence, k=1)
    gold_chunk_answer_at_3 = gold_chunk_coverage(gold_chunk_ids=gold_chunk_ids, items=response.evidence, k=3)
    gold_chunk_answer_at_5 = gold_chunk_coverage(gold_chunk_ids=gold_chunk_ids, items=response.evidence, k=5)
    gold_chunk_answer_at_k = gold_chunk_coverage(gold_chunk_ids=gold_chunk_ids, items=response.evidence, k=recall_k)
    gold_chunk_candidate_at_k = gold_chunk_trace_coverage(
        gold_chunk_ids=gold_chunk_ids,
        trace=getattr(response.rag_trace, "evidence_quality_trace", []) or [],
        k=recall_k,
    )
    gold_chunk_recall_at_1 = float(gold_chunk_answer_at_1["recall"])
    gold_chunk_recall_at_3 = float(gold_chunk_answer_at_3["recall"])
    gold_chunk_recall_at_5 = float(gold_chunk_answer_at_5["recall"])
    gold_chunk_recall_at_k = float(gold_chunk_answer_at_k["recall"])
    gold_chunk_candidate_recall_at_k = float(gold_chunk_candidate_at_k["recall"])
    gold_chunk_hit_ids = list(gold_chunk_answer_at_k["hit_ids"])
    gold_chunk_missed_ids = list(gold_chunk_answer_at_k["missed_ids"])
    gold_chunk_candidate_hit_ids = list(gold_chunk_candidate_at_k["hit_ids"])
    gold_chunk_candidate_missed_ids = list(gold_chunk_candidate_at_k["missed_ids"])
    gold_chunk_dropped_after_retrieval_ids = [
        chunk_id for chunk_id in gold_chunk_candidate_hit_ids if chunk_id not in gold_chunk_hit_ids
    ]
    gold_chunk_not_retrieved_ids = [
        chunk_id for chunk_id in gold_chunk_missed_ids if chunk_id not in gold_chunk_candidate_hit_ids
    ]

    expected_modalities = {modality.lower() for modality in case.expected_modalities}
    image_expected = bool(expected_modalities & {"image", "figure", "chart"})
    visual_summary_expected = "vision" in expected_modalities
    visual_expected = image_expected or visual_summary_expected
    table_expected = "table" in expected_modalities
    ocr_expected = "ocr" in expected_modalities
    cited_image_evidence = any(evidence_has_image_signal(item) for item in response.evidence)
    cited_visual_summary = any(evidence_has_visual_summary(item) for item in response.evidence)
    cited_ocr_text = any(evidence_has_ocr_text(item) for item in response.evidence)
    image_evidence_hit = cited_image_evidence if image_expected else True
    visual_summary_hit = (cited_visual_summary and cited_image_evidence) if visual_summary_expected else True
    visual_evidence_hit = (image_evidence_hit and visual_summary_hit) if visual_expected else True
    table_evidence_hit = (
        any(evidence_has_table_signal(item) for item in response.evidence)
        if table_expected
        else True
    )
    ocr_text_hit = cited_ocr_text if ocr_expected else True
    ocr_evidence_hit = ocr_text_hit
    modality_evidence_hit = visual_evidence_hit and table_evidence_hit and ocr_evidence_hit
    citation_accuracy = citation_accuracy_proxy(answer=answer, evidence=response.evidence, trace=response.rag_trace)
    answer_relevance = answer_relevance_proxy(question=case.question, answer=answer)
    faithfulness = faithfulness_proxy_score(answer=answer, evidence=response.evidence)
    faithfulness_proxy = float(faithfulness["score"])
    claim_terms = expected_claim_terms(case)
    explicit_claims_used = bool(case.expected_claims)
    claim_hit_rate = claim_hit_rate_proxy(claims=claim_terms, answer=answer)
    forbidden_claim_rate = forbidden_claim_rate_proxy(claims=case.forbidden_claims, answer=answer)
    refusal_correctness = refusal_correctness_proxy(case=case, answer=answer)
    relation_hit = relation_hit_proxy(case=case, answer=answer, trace=response.rag_trace)
    wrong_refusal = wrong_refusal_proxy(
        case=case,
        answer=answer,
        claim_hit_rate=claim_hit_rate,
        keyword_hit_rate=keyword_hit_rate,
    )
    visual_warnings = getattr(response.rag_trace, "visual_ocr_warnings", []) or []
    visual_warning_count = sum(1 for warning in visual_warnings if str(warning.get("severity", "warn")) == "warn")

    score_breakdown = {
        "keyword_hit_rate": keyword_hit_rate,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "gold_chunk_count": float(gold_chunk_count),
        "gold_chunk_recall_at_1": gold_chunk_recall_at_1,
        "gold_chunk_recall_at_3": gold_chunk_recall_at_3,
        "gold_chunk_recall_at_5": gold_chunk_recall_at_5,
        "gold_chunk_recall_at_k": gold_chunk_recall_at_k,
        "gold_chunk_k": float(recall_k),
        "gold_chunk_answer_hit_count": float(len(gold_chunk_hit_ids)),
        "gold_chunk_answer_missed_count": float(len(gold_chunk_missed_ids)),
        "gold_chunk_answer_all_hit": 1.0 if gold_chunk_count > 0 and not gold_chunk_missed_ids else 0.0,
        "gold_chunk_candidate_recall_at_k": gold_chunk_candidate_recall_at_k,
        "gold_chunk_candidate_hit_count": float(len(gold_chunk_candidate_hit_ids)),
        "gold_chunk_candidate_missed_count": float(len(gold_chunk_candidate_missed_ids)),
        "gold_chunk_candidate_all_hit": 1.0 if gold_chunk_count > 0 and not gold_chunk_candidate_missed_ids else 0.0,
        "gold_chunk_dropped_after_retrieval_count": float(len(gold_chunk_dropped_after_retrieval_ids)),
        "gold_chunk_not_retrieved_count": float(len(gold_chunk_not_retrieved_ids)),
        "document_coverage": document_coverage,
        "citation_page_hit_rate": citation_hit_rate,
        "citation_page_expected_count": float(len(expected_pages)),
        "citation_page_missed_count": float(len(citation_page_diagnostics["missed_pages"])),
        "citation_accuracy": citation_accuracy,
        "answer_relevance": answer_relevance,
        "faithfulness_proxy": faithfulness_proxy,
        "faithfulness_sentence_count": float(faithfulness["sentence_count"]),
        "faithfulness_supported_sentence_rate": float(faithfulness["supported_sentence_rate"]),
        "faithfulness_uncited_sentence_count": float(faithfulness["uncited_sentence_count"]),
        "faithfulness_weak_sentence_count": float(faithfulness["weak_sentence_count"]),
        "image_evidence_hit": 1.0 if image_evidence_hit else 0.0,
        "visual_summary_hit": 1.0 if visual_summary_hit else 0.0,
        "ocr_text_hit": 1.0 if ocr_text_hit else 0.0,
        "cited_image_evidence": 1.0 if cited_image_evidence else 0.0,
        "cited_visual_summary": 1.0 if cited_visual_summary else 0.0,
        "cited_ocr_text": 1.0 if cited_ocr_text else 0.0,
        "visual_evidence_hit": 1.0 if visual_evidence_hit else 0.0,
        "table_evidence_hit": 1.0 if table_evidence_hit else 0.0,
        "ocr_evidence_hit": 1.0 if ocr_evidence_hit else 0.0,
        "modality_evidence_hit": 1.0 if modality_evidence_hit else 0.0,
        "claim_hit_rate": claim_hit_rate,
        "claim_term_count": float(len(claim_terms)),
        "explicit_claims_used": 1.0 if explicit_claims_used else 0.0,
        "forbidden_claim_rate": forbidden_claim_rate,
        "refusal_correctness": refusal_correctness,
        "wrong_refusal": 1.0 if wrong_refusal else 0.0,
        "relation_hit": relation_hit,
        "visual_warning_count": float(visual_warning_count),
    }
    raw_score = max(
        0.0,
        (
            keyword_hit_rate * 0.12
            + context_precision * 0.08
            + context_recall * 0.10
            + document_coverage * 0.12
            + citation_accuracy * 0.10
            + answer_relevance * 0.08
            + faithfulness_proxy * 0.06
            + (1.0 if modality_evidence_hit else 0.0) * 0.04
            + claim_hit_rate * 0.20
            + refusal_correctness * 0.06
            + relation_hit * 0.04
        )
        - forbidden_claim_rate * 0.10
    )
    score_cap = eval_score_cap(
        case=case,
        claim_terms=claim_terms,
        claim_hit_rate=claim_hit_rate,
        keyword_hit_rate=keyword_hit_rate,
        context_recall=context_recall,
        document_coverage=document_coverage,
        citation_hit_rate=citation_hit_rate,
        gold_chunk_count=gold_chunk_count,
        gold_chunk_recall_at_k=gold_chunk_recall_at_k,
        wrong_refusal=wrong_refusal,
        forbidden_claim_rate=forbidden_claim_rate,
    )
    score_breakdown["score_before_caps"] = raw_score
    score_breakdown["score_cap"] = score_cap
    score = min(raw_score, score_cap)
    return {
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "keyword_hit_rate": keyword_hit_rate,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "gold_chunk_count": gold_chunk_count,
        "gold_chunk_recall_at_1": gold_chunk_recall_at_1,
        "gold_chunk_recall_at_3": gold_chunk_recall_at_3,
        "gold_chunk_recall_at_5": gold_chunk_recall_at_5,
        "gold_chunk_recall_at_k": gold_chunk_recall_at_k,
        "gold_chunk_hit_ids": gold_chunk_hit_ids,
        "gold_chunk_missed_ids": gold_chunk_missed_ids,
        "gold_chunk_candidate_recall_at_k": gold_chunk_candidate_recall_at_k,
        "gold_chunk_candidate_hit_ids": gold_chunk_candidate_hit_ids,
        "gold_chunk_candidate_missed_ids": gold_chunk_candidate_missed_ids,
        "gold_chunk_dropped_after_retrieval_ids": gold_chunk_dropped_after_retrieval_ids,
        "gold_chunk_not_retrieved_ids": gold_chunk_not_retrieved_ids,
        "gold_chunk_k": recall_k,
        "document_coverage": document_coverage,
        "citation_page_diagnostics": citation_page_diagnostics,
        "image_evidence_hit": image_evidence_hit,
        "visual_evidence_hit": visual_evidence_hit,
        "visual_summary_hit": visual_summary_hit,
        "table_evidence_hit": table_evidence_hit,
        "ocr_evidence_hit": ocr_evidence_hit,
        "ocr_text_hit": ocr_text_hit,
        "citation_accuracy": citation_accuracy,
        "answer_relevance": answer_relevance,
        "faithfulness_proxy": faithfulness_proxy,
        "claim_hit_rate": claim_hit_rate,
        "forbidden_claim_rate": forbidden_claim_rate,
        "refusal_correctness": refusal_correctness,
        "relation_hit": relation_hit,
        "visual_warning_count": visual_warning_count,
        "score": score,
        "score_cap": score_cap,
        "score_breakdown": score_breakdown,
    }


def evaluation_evidence_summary(response) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "paper_name": item.paper_name,
            "page": item.page,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "section": item.section,
            "chunk_type": item.chunk_type,
            "image_id": item.image_id,
            "score": item.score,
            "quote": truncate_for_judge(item.quote or item.text, 420),
            "related_images": [
                {
                    "id": image.id,
                    "page_start": image.page_start,
                    "page_end": image.page_end,
                    "kind": image.kind,
                    "status": image.status,
                    "vision_summary": truncate_for_judge(image.vision_summary, 360),
                    "vision_error": truncate_for_judge(image.vision_error, 220),
                    "ocr_text": truncate_for_judge(image.ocr_text, 220),
                    "ocr_status": image.ocr_status,
                    "ocr_error": truncate_for_judge(image.ocr_error, 220),
                    "caption_text": truncate_for_judge(image.caption_text, 220),
                }
                for image in item.related_images[:3]
            ],
        }
        for item in response.evidence[:12]
    ]


def evaluation_trace_summary(response) -> dict[str, Any]:
    trace = response.rag_trace
    return {
        "retrieval_strategy": trace.retrieval_strategy,
        "retrieval_pipeline": trace.retrieval_pipeline,
        "ranking_method": trace.ranking_method,
        "retrieved_count": trace.retrieved_count,
        "top_k": trace.top_k,
        "embedding_requested_model": trace.embedding_requested_model,
        "embedding_provider": trace.embedding_provider,
        "embedding_used_fallback": trace.embedding_used_fallback,
        "embedding_fallback_reason": trace.embedding_fallback_reason,
        "embedding_document_fallback_count": trace.embedding_document_fallback_count,
        "embedding_document_providers": trace.embedding_document_providers,
        "final_prompt_evidence": trace.final_prompt_evidence,
        "evidence_quality": trace.evidence_quality,
        "evidence_coverage": trace.evidence_coverage,
        "evidence_quality_trace": trace.evidence_quality_trace,
        "multi_document_coverage": trace.multi_document_coverage,
        "document_relation_map": trace.document_relation_map,
        "verification": trace.verification,
        "visual_ocr_warnings": trace.visual_ocr_warnings,
    }


def gold_evidence_summary(*, case: EvaluationCase, metrics: dict[str, Any]) -> dict[str, Any]:
    expected_chunk_ids = clean_terms(case.expected_chunk_ids)
    if not expected_chunk_ids:
        return {"status": "not_configured", "expected_chunk_ids": []}

    answer_missed = list(metrics.get("gold_chunk_missed_ids") or [])
    candidate_missed = list(metrics.get("gold_chunk_candidate_missed_ids") or [])
    if not answer_missed:
        status = "pass"
    elif not candidate_missed:
        status = "dropped_after_retrieval"
    elif len(candidate_missed) < len(expected_chunk_ids):
        status = "partial_candidate_retrieval"
    else:
        status = "not_retrieved"

    return {
        "status": status,
        "expected_chunk_ids": expected_chunk_ids,
        "answer_evidence": {
            "source": "final_answer_evidence",
            "k": int(metrics.get("gold_chunk_k") or 0),
            "hit_chunk_ids": list(metrics.get("gold_chunk_hit_ids") or []),
            "missed_chunk_ids": answer_missed,
            "recall_at_1": float(metrics.get("gold_chunk_recall_at_1") or 0.0),
            "recall_at_3": float(metrics.get("gold_chunk_recall_at_3") or 0.0),
            "recall_at_5": float(metrics.get("gold_chunk_recall_at_5") or 0.0),
            "recall_at_k": float(metrics.get("gold_chunk_recall_at_k") or 0.0),
        },
        "candidate_trace": {
            "source": "evidence_quality_trace",
            "k": int(metrics.get("gold_chunk_k") or 0),
            "hit_chunk_ids": list(metrics.get("gold_chunk_candidate_hit_ids") or []),
            "missed_chunk_ids": candidate_missed,
            "recall_at_k": float(metrics.get("gold_chunk_candidate_recall_at_k") or 0.0),
        },
        "dropped_after_retrieval_ids": list(metrics.get("gold_chunk_dropped_after_retrieval_ids") or []),
        "not_retrieved_ids": list(metrics.get("gold_chunk_not_retrieved_ids") or []),
    }


def judge_evidence_payload(evidence) -> list[dict[str, Any]]:
    return [
        {
            "id": item.citation_id,
            "doc": item.paper_name,
            "page": item.page,
            "type": item.chunk_type,
            "image_id": item.image_id,
            "quote": truncate_for_judge(item.quote or item.text, 520),
        }
        for item in evidence[:6]
    ]


def judge_trace_payload(trace) -> dict[str, Any]:
    return {
        "retrieval_pipeline": trace.retrieval_pipeline,
        "ranking_method": trace.ranking_method,
        "retrieved_count": trace.retrieved_count,
        "final_prompt_evidence": list(trace.final_prompt_evidence[:8]),
        "evidence_coverage": trace.evidence_coverage,
        "evidence_quality_trace": [
            {
                "chunk_id": _payload_value(row, "chunk_id"),
                "status": _payload_value(row, "selection_status"),
                "label": _payload_value(row, "quality_label"),
                "rank": _payload_value(row, "candidate_rank"),
                "reason": truncate_for_judge(
                    _payload_value(row, "rejection_reason") or _payload_value(row, "judge_reason"),
                    160,
                ),
            }
            for row in list(trace.evidence_quality_trace or [])[:8]
        ],
        "multi_document_coverage": trace.multi_document_coverage,
        "visual_ocr_warnings": list((trace.visual_ocr_warnings or [])[:4]),
        "verification": trace.verification,
    }


def _payload_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, "")


def score_with_llm_judge(
    *,
    case: EvaluationCase,
    response,
    agent: "PaperAgentService",
    enabled_override: bool | None = None,
) -> dict[str, Any]:
    enabled = getattr(agent.settings, "enable_llm_judge", False) if enabled_override is None else enabled_override
    if not enabled:
        return empty_judge_result("LLM judge disabled.")
    if not getattr(agent.settings, "api_key", None):
        return empty_judge_result("LLM judge skipped because chat API key is not configured.")

    prompt_payload = {
        "case_id": case.id,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_keywords": case.expected_keywords,
        "expected_documents": case.expected_documents or ([case.expected_document] if case.expected_document else []),
        "expected_pages": case.expected_pages or ([case.expected_page] if case.expected_page is not None else []),
        "expected_modalities": case.expected_modalities,
        "relation_keywords": case.relation_keywords,
        "expected_claims": case.expected_claims,
        "forbidden_claims": case.forbidden_claims,
        "expected_relation": case.expected_relation,
        "expected_refusal": case.expected_refusal,
        "judge_rubric": case.judge_rubric,
        "answer": response.answer,
        "evidence": judge_evidence_payload(response.evidence),
        "trace_summary": judge_trace_payload(response.rag_trace),
    }
    system_prompt = JUDGE_SYSTEM_PROMPT
    user_prompt = f"""
Score keys: {", ".join(JUDGE_SCORE_KEYS)}. Use 1 when a key is not applicable.
Return:
{{
  "scores": {{key: 0.0}},
  "overall": 0.0,
  "reason": "short Chinese explanation"
}}

Payload:
{json.dumps(prompt_payload, ensure_ascii=False)}
""".strip()
    try:
        raw = agent.model_clients.chat_text(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            model=getattr(agent.settings, "judge_model", None),
        )
        payload = parse_judge_json(raw)
        scores = normalize_judge_scores(payload.get("scores", {}))
        overall = clamp01(float(payload.get("overall", weighted_judge_score(scores))))
        if not scores:
            overall = 0.0
        return {
            "used": True,
            "score": overall,
            "scores": scores,
            "reason": str(payload.get("reason") or "").strip()[:800],
        }
    except Exception as exc:
        return empty_judge_result(f"LLM judge failed: {exc}")


def empty_judge_result(reason: str) -> dict[str, Any]:
    return {"used": False, "score": 0.0, "scores": {}, "reason": reason}


def combine_eval_scores(*, programmatic_score: float, judge: dict[str, Any]) -> float:
    if not judge.get("used"):
        return programmatic_score
    return programmatic_score * 0.55 + float(judge.get("score", 0.0)) * 0.45


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def normalize_judge_scores(raw_scores: Any) -> dict[str, float]:
    if not isinstance(raw_scores, dict):
        return {}
    allowed = [
        "answer_relevance",
        "faithfulness",
        "citation_support",
        "context_usage",
        "multi_document_clarity",
        "visual_grounding",
        "claim_coverage",
        "refusal_correctness",
        "completeness",
        "no_hallucination",
    ]
    scores: dict[str, float] = {}
    for key in allowed:
        try:
            scores[key] = clamp01(float(raw_scores.get(key, 0.0)))
        except (TypeError, ValueError):
            scores[key] = 0.0
    return scores


def weighted_judge_score(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    weights = {
        "answer_relevance": 0.12,
        "faithfulness": 0.20,
        "citation_support": 0.16,
        "context_usage": 0.08,
        "multi_document_clarity": 0.08,
        "visual_grounding": 0.07,
        "claim_coverage": 0.12,
        "refusal_correctness": 0.07,
        "completeness": 0.06,
        "no_hallucination": 0.04,
    }
    return sum(scores.get(key, 0.0) * weight for key, weight in weights.items())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def truncate_for_judge(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def expected_document_names(case: EvaluationCase) -> list[str]:
    names = list(case.expected_documents)
    if case.expected_document:
        names.append(case.expected_document)
    return clean_terms(names)


def expected_page_numbers(case: EvaluationCase) -> list[int]:
    pages = list(case.expected_pages)
    if case.expected_page is not None:
        pages.append(case.expected_page)
    return sorted(set(pages))


def expected_document_coverage(
    *,
    expected_documents: list[str],
    required_document_count: int | None,
    evidence,
) -> float:
    if expected_documents:
        matched = 0
        for expected in expected_documents:
            expected_normalized = expected.lower()
            if any(
                expected_normalized in item.paper_name.lower()
                or expected_normalized == item.document_id.lower()
                or expected_normalized in item.document_id.lower()
                for item in evidence
            ):
                matched += 1
        return matched / max(len(expected_documents), 1)
    if required_document_count:
        unique_documents = {item.document_id for item in evidence if item.document_id}
        return min(1.0, len(unique_documents) / max(required_document_count, 1))
    return 1.0


def expected_page_hit_rate(*, expected_pages: list[int], evidence) -> float:
    if not expected_pages:
        return 1.0
    matched = 0
    for page in expected_pages:
        if any(_evidence_page_start(item) <= page <= _evidence_page_end(item) for item in evidence):
            matched += 1
    return matched / max(len(expected_pages), 1)


def citation_page_diagnostics_for_eval(*, expected_pages: list[int], evidence) -> dict[str, Any]:
    pages = sorted(set(int(page) for page in expected_pages if int(page) > 0))
    evidence_ranges = [
        {
            "citation_id": str(getattr(item, "citation_id", "") or ""),
            "chunk_id": str(getattr(item, "chunk_id", "") or ""),
            "page": int(getattr(item, "page", 0) or 0),
            "page_start": _evidence_page_start(item),
            "page_end": _evidence_page_end(item),
            "chunk_type": str(getattr(item, "chunk_type", "") or ""),
        }
        for item in evidence
    ]
    matched_pages: list[int] = []
    missed_pages: list[int] = []
    nearest: list[dict[str, Any]] = []
    for page in pages:
        covering = [
            row
            for row in evidence_ranges
            if int(row["page_start"]) <= page <= int(row["page_end"])
        ]
        if covering:
            matched_pages.append(page)
            continue
        missed_pages.append(page)
        nearest_row = _nearest_evidence_page(page=page, evidence_ranges=evidence_ranges)
        if nearest_row:
            nearest.append(nearest_row)

    if not pages:
        status = "not_configured"
    elif not evidence_ranges:
        status = "no_visible_evidence"
    elif not missed_pages:
        status = "pass"
    elif matched_pages:
        status = "partial_page_match"
    else:
        status = "page_mismatch"

    hint = ""
    if status == "page_mismatch" and nearest:
        min_distance = min(int(row["distance"]) for row in nearest)
        if min_distance >= 5:
            hint = "期望页码与可见证据页码相差较大，请检查评测集使用的是 PDF 物理页、文档印刷页，还是旧索引页码。"
        else:
            hint = "可见证据接近期望页，但未覆盖期望页码；请检查 chunk 页码区间或 expected_pages 是否过窄。"

    return {
        "status": status,
        "expected_pages": pages,
        "matched_pages": matched_pages,
        "missed_pages": missed_pages,
        "hit_rate": expected_page_hit_rate(expected_pages=pages, evidence=evidence),
        "evidence_ranges": evidence_ranges,
        "nearest_evidence": nearest,
        "hint": hint,
    }


def _nearest_evidence_page(*, page: int, evidence_ranges: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evidence_ranges:
        return None
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in evidence_ranges:
        start = int(row["page_start"])
        end = int(row["page_end"])
        if page < start:
            distance = start - page
            nearest_page = start
        elif page > end:
            distance = page - end
            nearest_page = end
        else:
            distance = 0
            nearest_page = page
        scored.append((distance, {**row, "expected_page": page, "nearest_page": nearest_page, "distance": distance}))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _evidence_page_start(item) -> int:
    page = int(getattr(item, "page", 0) or 0)
    page_start = int(getattr(item, "page_start", 0) or 0)
    return page_start or page


def _evidence_page_end(item) -> int:
    page = int(getattr(item, "page", 0) or 0)
    page_end = int(getattr(item, "page_end", 0) or 0)
    start = _evidence_page_start(item)
    end = page_end or page
    return max(start, end)


def gold_chunk_coverage(*, gold_chunk_ids: list[str], items, k: int) -> dict[str, Any]:
    expected_ids = clean_terms(gold_chunk_ids)
    top_k = max(int(k or 0), 1)
    retrieved_ids: list[str] = []
    rank_by_id: dict[str, int] = {}
    for rank, item in enumerate(list(items or [])[:top_k], start=1):
        chunk_id = _item_value(item, "chunk_id")
        if not chunk_id or chunk_id in rank_by_id:
            continue
        retrieved_ids.append(chunk_id)
        rank_by_id[chunk_id] = rank
    return _gold_chunk_coverage_from_ranked_ids(
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        rank_by_id=rank_by_id,
        k=top_k,
    )


def gold_chunk_trace_coverage(*, gold_chunk_ids: list[str], trace, k: int) -> dict[str, Any]:
    expected_ids = clean_terms(gold_chunk_ids)
    top_k = max(int(k or 0), 1)
    ranked_rows: list[tuple[int, Any]] = []
    for order, row in enumerate(list(trace or []), start=1):
        rank = _safe_int(_item_value(row, "candidate_rank"), default=order)
        ranked_rows.append((rank, row))

    retrieved_ids: list[str] = []
    rank_by_id: dict[str, int] = {}
    for rank, row in sorted(ranked_rows, key=lambda item: item[0]):
        if rank > top_k:
            continue
        chunk_id = _item_value(row, "chunk_id")
        if not chunk_id or chunk_id in rank_by_id:
            continue
        retrieved_ids.append(chunk_id)
        rank_by_id[chunk_id] = rank
    return _gold_chunk_coverage_from_ranked_ids(
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        rank_by_id=rank_by_id,
        k=top_k,
    )


def _gold_chunk_coverage_from_ranked_ids(
    *,
    expected_ids: list[str],
    retrieved_ids: list[str],
    rank_by_id: dict[str, int],
    k: int,
) -> dict[str, Any]:
    if not expected_ids:
        return {
            "k": k,
            "expected_ids": [],
            "retrieved_ids": retrieved_ids,
            "retrieved_rank_by_id": rank_by_id,
            "hit_ids": [],
            "missed_ids": [],
            "recall": 0.0,
        }
    hit_ids = [chunk_id for chunk_id in expected_ids if chunk_id in rank_by_id]
    missed_ids = [chunk_id for chunk_id in expected_ids if chunk_id not in rank_by_id]
    return {
        "k": k,
        "expected_ids": expected_ids,
        "retrieved_ids": retrieved_ids,
        "retrieved_rank_by_id": rank_by_id,
        "hit_ids": hit_ids,
        "missed_ids": missed_ids,
        "recall": len(hit_ids) / max(len(expected_ids), 1),
    }


def _item_value(item: Any, key: str) -> str:
    if isinstance(item, dict):
        value = item.get(key)
    else:
        value = getattr(item, key, "")
    return str(value or "").strip()


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def gold_chunk_recall(*, gold_chunk_ids: list[str], evidence, k: int) -> float:
    return float(gold_chunk_coverage(gold_chunk_ids=gold_chunk_ids, items=evidence, k=k)["recall"])


def context_precision_proxy(*, terms: list[str], evidence) -> float:
    if not evidence:
        return 0.0 if terms else 1.0
    if not terms:
        return 1.0
    hits = 0
    lowered_terms = [term.lower() for term in terms]
    for item in evidence:
        text = f"{item.paper_name}\n{item.section or ''}\n{item.quote}\n{item.text}".lower()
        if any(term in text for term in lowered_terms):
            hits += 1
    return hits / max(len(evidence), 1)


def evidence_has_image_signal(item) -> bool:
    chunk_type = (item.chunk_type or "").lower()
    if item.image_id:
        return True
    if any(marker in chunk_type for marker in ["image", "figure", "chart"]):
        return True
    return any(
        getattr(image, "id", "")
        and any(marker in (getattr(image, "kind", "") or "").lower() for marker in ["image", "figure", "chart"])
        for image in getattr(item, "related_images", [])
    )


def evidence_has_visual_summary(item) -> bool:
    if not evidence_has_image_signal(item):
        return False
    if _meaningful_eval_text(_visual_summary_text(item)):
        return True
    return any(
        _meaningful_eval_text(getattr(image, "vision_summary", ""))
        for image in getattr(item, "related_images", [])
    )


def evidence_has_ocr_text(item) -> bool:
    if _meaningful_eval_text(_labeled_evidence_text(item, "ocr")):
        return True
    if "ocr" in (item.chunk_type or "").lower() and _meaningful_eval_text(f"{item.quote}\n{item.text}"):
        return True
    return any(
        _meaningful_eval_text(getattr(image, "ocr_text", ""))
        for image in getattr(item, "related_images", [])
    )


def _labeled_evidence_text(item, label: str) -> str:
    text = f"{item.quote}\n{item.text}"
    match = re.search(
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.+?)(?=^\s*[A-Za-z][A-Za-z _-]{{0,30}}\s*:|\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def _visual_summary_text(item) -> str:
    text = f"{item.quote}\n{item.text}"
    labeled = _labeled_evidence_text(item, "summary")
    if _meaningful_eval_text(labeled):
        return labeled

    labels = [
        "vision summary",
        "visual summary",
        "summary",
        "图片实际内容",
        "图片说明",
        "实际内容",
        "图表属性",
        "支持的论文事实",
        "可支撑的论文事实",
    ]
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?ims)(?:^|\n|\s)({label_pattern})\s*[:：]?\s*(.+?)(?=\n\s*(?:[A-Za-z][A-Za-z _-]{{0,30}}|图片实际内容|图片说明|实际内容|图表属性|支持的论文事实|可支撑的论文事实|OCR)\s*[:：]|\Z)",
        text,
    )
    if match:
        return match.group(2).strip()
    return ""


def _meaningful_eval_text(text: str, *, min_chars: int = 6) -> bool:
    tokens = re.findall(r"[\w\u4e00-\u9fff]", normalize_eval_text(text))
    return len(tokens) >= min_chars


def evidence_has_table_signal(item) -> bool:
    chunk_type = (item.chunk_type or "").lower()
    text = f"{item.quote}\n{item.text}"
    normalized = normalize_eval_text(text)
    return (
        "table" in chunk_type
        or bool(re.search(r"\btable\s*\d+", normalized))
        or ("sequential operations" in normalized and "maximum path" in normalized)
        or ("bleu" in normalized and ("en-de" in normalized or "english-to-german" in normalized))
        or ("glue" in normalized and "average" in normalized)
    )


def citation_accuracy_proxy(*, answer: str, evidence, trace) -> float:
    citation_ids = set(citation_ids_from_text(answer))
    if not citation_ids:
        return 0.0 if evidence else 1.0
    valid_ids = {item.citation_id for item in evidence}
    base = len(citation_ids & valid_ids) / max(len(citation_ids), 1)
    verification = trace.verification or {}
    if verification.get("status") == "fail":
        return min(base, 0.25)
    if verification.get("status") == "warn":
        return min(base, 0.65)
    return base


def citation_ids_from_text(text: str) -> list[str]:
    matches = re.findall(r"\[E(\d+)\]|\bE(\d+)\b", text)
    ids: list[str] = []
    for bracketed, bare in matches:
        value = bracketed or bare
        citation_id = f"E{value}"
        if citation_id not in ids:
            ids.append(citation_id)
    return ids


def faithfulness_proxy_score(*, answer: str, evidence) -> dict[str, float]:
    if not str(answer).strip():
        return _faithfulness_result(score=0.0)
    if not evidence:
        return _faithfulness_result(score=1.0 if contains_refusal_signal_strict(answer) else 0.0)

    evidence_by_id = {item.citation_id: item for item in evidence}
    sentences = _faithfulness_candidate_sentences(answer)
    if not sentences:
        return _faithfulness_result(score=1.0 if contains_refusal_signal_strict(answer) else 0.0)

    sentence_scores: list[float] = []
    supported = 0
    weak = 0
    uncited = 0
    for sentence in sentences:
        cited_ids = citation_ids_from_text(sentence)
        if cited_ids:
            cited_evidence = [evidence_by_id[citation_id] for citation_id in cited_ids if citation_id in evidence_by_id]
            support_score = _sentence_support_score(sentence, cited_evidence)
        else:
            uncited += 1
            support_score = min(0.65, _sentence_support_score(sentence, evidence))

        if support_score >= 0.72:
            supported += 1
        elif support_score < 0.45:
            weak += 1
        sentence_scores.append(support_score)

    raw_score = sum(sentence_scores) / max(len(sentence_scores), 1)
    if uncited:
        raw_score = min(raw_score, 0.85)
    return _faithfulness_result(
        score=clamp01(raw_score),
        sentence_count=len(sentences),
        supported_sentence_rate=supported / max(len(sentences), 1),
        uncited_sentence_count=uncited,
        weak_sentence_count=weak,
    )


def _faithfulness_result(
    *,
    score: float,
    sentence_count: int = 0,
    supported_sentence_rate: float = 0.0,
    uncited_sentence_count: int = 0,
    weak_sentence_count: int = 0,
) -> dict[str, float]:
    return {
        "score": clamp01(score),
        "sentence_count": float(sentence_count),
        "supported_sentence_rate": clamp01(supported_sentence_rate),
        "uncited_sentence_count": float(uncited_sentence_count),
        "weak_sentence_count": float(weak_sentence_count),
    }


def _faithfulness_candidate_sentences(answer: str) -> list[str]:
    sentences = [
        part.strip(" \t\r\n-*")
        for part in re.split(r"(?<=[.!?。！？；;])\s+|\n+", str(answer))
        if part.strip(" \t\r\n-*")
    ]
    candidates: list[str] = []
    for sentence in sentences:
        if contains_refusal_signal_strict(sentence):
            continue
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", sentence):
            continue
        if len(_faithfulness_tokens(sentence)) < 2:
            continue
        candidates.append(sentence)
    return candidates


def _sentence_support_score(sentence: str, evidence) -> float:
    if not evidence:
        return 0.0
    sentence_tokens = _faithfulness_tokens(re.sub(r"\[E\d+\]|\bE\d+\b", "", sentence))
    if not sentence_tokens:
        return 1.0
    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens.update(_faithfulness_tokens(_evidence_text_for_faithfulness(item)))
    if not evidence_tokens:
        return 0.0
    hits = sum(1 for token in sentence_tokens if token in evidence_tokens)
    overlap = hits / max(len(sentence_tokens), 1)
    if hits < 2:
        return min(overlap, 0.35)
    return clamp01(overlap)


def _evidence_text_for_faithfulness(item) -> str:
    related = "\n".join(
        " ".join(
            part
            for part in [
                getattr(image, "caption_text", ""),
                getattr(image, "ocr_text", ""),
                getattr(image, "vision_summary", ""),
            ]
            if part
        )
        for image in getattr(item, "related_images", [])
    )
    return f"{item.paper_name}\n{item.section or ''}\n{item.quote}\n{item.text}\n{related}"


def _faithfulness_tokens(text: str) -> list[str]:
    normalized = normalize_eval_text(text)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "about",
        "paper",
        "says",
        "show",
        "shows",
        "uses",
        "using",
        "based",
        "evidence",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]{3,}", normalized)
        if token not in stopwords
    ]
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        if len(segment) == 2:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return list(dict.fromkeys(tokens))


def answer_relevance_proxy(*, question: str, answer: str) -> float:
    terms = question_terms(question)
    if not terms:
        return 1.0
    return term_hit_rate(terms, answer.lower())


def claim_hit_rate_proxy(*, claims: list[str], answer: str) -> float:
    cleaned = clean_terms(claims)
    if not cleaned:
        return 1.0
    return term_hit_rate(cleaned, normalize_eval_text(answer))


def expected_claim_terms(case: EvaluationCase) -> list[str]:
    explicit_claims = clean_terms(case.expected_claims)
    if explicit_claims:
        return explicit_claims
    return clean_terms(case.expected_keywords)


def forbidden_claim_rate_proxy(*, claims: list[str], answer: str) -> float:
    cleaned = clean_terms(claims)
    if not cleaned:
        return 0.0
    return term_hit_rate(cleaned, normalize_eval_text(answer))


def refusal_correctness_proxy(*, case: EvaluationCase, answer: str) -> float:
    if not expected_refusal_case(case):
        return 1.0
    return 1.0 if contains_refusal_signal(answer) or contains_refusal_signal_strict(answer) else 0.0


def wrong_refusal_proxy(
    *,
    case: EvaluationCase,
    answer: str,
    claim_hit_rate: float,
    keyword_hit_rate: float,
) -> bool:
    if expected_refusal_case(case):
        return False
    if not contains_refusal_signal(answer) and not contains_refusal_signal_strict(answer):
        return False
    if case.expected_claims:
        return claim_hit_rate < 0.75
    if case.expected_keywords:
        return keyword_hit_rate < 0.50
    return claim_hit_rate < 0.50


def eval_score_cap(
    *,
    case: EvaluationCase,
    claim_terms: list[str],
    claim_hit_rate: float,
    keyword_hit_rate: float,
    context_recall: float,
    document_coverage: float,
    citation_hit_rate: float,
    gold_chunk_count: int,
    gold_chunk_recall_at_k: float,
    wrong_refusal: bool,
    forbidden_claim_rate: float,
) -> float:
    cap = 1.0
    if claim_terms:
        if claim_hit_rate <= 0.0:
            cap = min(cap, 0.35)
        elif claim_hit_rate < 0.50:
            cap = min(cap, 0.55)
        elif claim_hit_rate < 0.75:
            cap = min(cap, 0.75)

    if case.expected_keywords:
        if keyword_hit_rate <= 0.0:
            cap = min(cap, 0.45)
        elif keyword_hit_rate < 0.50:
            cap = min(cap, 0.65)

    if case.expected_evidence_keywords and context_recall < 0.40:
        cap = min(cap, 0.70)

    if (case.expected_document or case.expected_documents or case.required_document_count) and document_coverage < 1.0:
        cap = min(cap, 0.65)

    if (case.expected_page is not None or case.expected_pages) and citation_hit_rate < 1.0:
        cap = min(cap, 0.75)

    if gold_chunk_count > 0:
        if gold_chunk_recall_at_k <= 0.0:
            cap = min(cap, 0.40)
        elif gold_chunk_recall_at_k < 0.50:
            cap = min(cap, 0.60)
        elif gold_chunk_recall_at_k < 1.0:
            cap = min(cap, 0.78)

    if wrong_refusal:
        cap = min(cap, 0.45)

    if forbidden_claim_rate >= 0.50:
        cap = min(cap, 0.60)
    elif forbidden_claim_rate > 0.0:
        cap = min(cap, 0.75)

    return cap


def relation_hit_proxy(*, case: EvaluationCase, answer: str, trace) -> float:
    relation_terms = clean_terms(case.relation_keywords)
    relation_aliases = relation_terms_for_label(case.expected_relation)
    if not relation_terms and not relation_aliases:
        return 1.0
    relation_payload = json.dumps(getattr(trace, "document_relation_map", []) or [], ensure_ascii=False)
    coverage_payload = json.dumps(getattr(trace, "multi_document_coverage", {}) or {}, ensure_ascii=False)
    text = normalize_eval_text(f"{answer}\n{relation_payload}\n{coverage_payload}")
    scores: list[float] = []
    if relation_terms:
        scores.append(term_hit_rate(relation_terms, text))
    if relation_aliases:
        normalized_aliases = [normalize_eval_text(alias) for alias in relation_aliases]
        scores.append(1.0 if any(alias and alias in text for alias in normalized_aliases) else 0.0)
    return max(scores) if scores else 1.0


def relation_terms_for_label(label: str) -> list[str]:
    normalized = label.strip().lower()
    if not normalized:
        return []
    aliases = {
        "support": ["支持", "基础", "应用", "继承", "建立在", "uses", "foundation"],
        "comparison": ["比较", "共同", "不同", "差异", "异同", "comparison"],
        "contrast": ["对比", "不同", "差异", "contrast"],
        "conflict": ["冲突", "矛盾", "不一致", "相反", "conflict"],
        "boundary": ["不能证明", "证据不足", "边界", "不等于", "没有直接证据"],
        "refusal": ["不能证明", "证据不足", "拒绝", "无证据"],
        "complementary": ["互补", "补充", "complementary"],
        "shared_topic": ["共同", "同一主题", "shared"],
        "same_term_different_context": ["同名", "不同", "领域", "语境"],
    }
    return aliases.get(normalized, [label])


def contains_refusal_signal(text: str) -> bool:
    normalized = normalize_eval_text(text)
    markers = ["不能证明", "无法证明", "证据不足", "没有证据", "无证据", "不能据此", "不足以", "不应得出"]
    return any(marker in normalized for marker in markers)


def normalize_eval_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def contains_refusal_signal_strict(text: str) -> bool:
    normalized = normalize_eval_text(text)
    markers = [
        "不能证明",
        "无法证明",
        "证据不足",
        "没有证据",
        "无证据",
        "不能据此",
        "无法据此",
        "不足以",
        "不应得出",
        "不能得出",
        "无法得出",
        "无法回答",
        "不能回答",
        "没有明确",
        "未明确",
        "未检索到",
        "没有检索到",
        "没有找到",
        "文档未提供",
        "当前证据",
        "没有任何内容支持",
        "没有内容支持",
        "没有直接支持",
        "不支持",
        "不能支持",
        "不涉及",
        "无关",
        "答案是否定",
        "是否定的",
    ]
    return any(marker in normalized for marker in markers)


def term_hit_rate(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    normalized = normalize_eval_text(text)
    hits = sum(1 for term in terms if normalize_eval_text(term) in normalized)
    return hits / max(len(terms), 1)


def clean_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    for term in terms:
        value = str(term).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def question_terms(question: str) -> list[str]:
    blocked = {"这个", "那个", "论文", "文献", "文章", "请问", "什么", "哪些", "如何", "是否"}
    tokens = re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", question.lower())
    terms: list[str] = []
    for token in tokens:
        if token in blocked or token in terms:
            continue
        terms.append(token)
        if len(terms) >= 8:
            break
    return terms
