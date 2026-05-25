from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.models import AskRequest, EvaluationCase, EvaluationResult, EvaluationRun
from backend.app.observability import ObservabilityClient, new_run_id, utc_now

if TYPE_CHECKING:
    from backend.app.agent import PaperAgentService


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
        final_score = combine_eval_scores(programmatic_score=float(metrics["score"]), judge=judge)
        score_breakdown = dict(metrics["score_breakdown"])
        if judge["used"]:
            score_breakdown["llm_judge"] = float(judge["score"])
        trace_summary = evaluation_trace_summary(response)
        trace_summary["case_metadata"] = case_metadata(case)
        results.append(
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
                document_coverage=float(metrics["document_coverage"]),
                image_evidence_hit=bool(metrics["image_evidence_hit"]),
                visual_evidence_hit=bool(metrics["visual_evidence_hit"]),
                table_evidence_hit=bool(metrics["table_evidence_hit"]),
                ocr_evidence_hit=bool(metrics["ocr_evidence_hit"]),
                citation_accuracy=float(metrics["citation_accuracy"]),
                answer_relevance=float(metrics["answer_relevance"]),
                faithfulness_proxy=float(metrics["faithfulness_proxy"]),
                claim_hit_rate=float(metrics["claim_hit_rate"]),
                forbidden_claim_rate=float(metrics["forbidden_claim_rate"]),
                refusal_correctness=float(metrics["refusal_correctness"]),
                relation_hit=float(metrics["relation_hit"]),
                visual_warning_count=int(metrics["visual_warning_count"]),
                judge_used=bool(judge["used"]),
                judge_score=float(judge["score"]),
                judge_scores=dict(judge["scores"]),
                judge_reason=str(judge["reason"]),
                score=final_score,
                score_breakdown=score_breakdown,
                latency_ms=elapsed_ms,
            )
        )

    total = max(len(results), 1)
    judge_used_count = sum(item.judge_used for item in results)
    segment_metrics = build_segment_metrics(cases=cases, results=results)
    run = EvaluationRun(
        run_id=new_run_id("eval"),
        suite_name=suite_name,
        created_at=utc_now(),
        document_ids=document_ids,
        case_count=len(results),
        judge_enabled=bool(enable_judge if enable_judge is not None else getattr(agent.settings, "enable_llm_judge", False)),
        score_version="rag-eval-v2-claim-modality",
        results=results,
        retrieval_hit_rate=sum(item.retrieval_hit for item in results) / total,
        citation_hit_rate=sum(item.citation_hit for item in results) / total,
        avg_keyword_hit_rate=sum(item.keyword_hit_rate for item in results) / total,
        avg_context_precision=sum(item.context_precision for item in results) / total,
        avg_context_recall=sum(item.context_recall for item in results) / total,
        avg_document_coverage=sum(item.document_coverage for item in results) / total,
        avg_image_evidence_hit_rate=sum(item.image_evidence_hit for item in results) / total,
        avg_visual_evidence_hit_rate=sum(item.visual_evidence_hit for item in results) / total,
        avg_table_evidence_hit_rate=sum(item.table_evidence_hit for item in results) / total,
        avg_ocr_evidence_hit_rate=sum(item.ocr_evidence_hit for item in results) / total,
        avg_citation_accuracy=sum(item.citation_accuracy for item in results) / total,
        avg_answer_relevance=sum(item.answer_relevance for item in results) / total,
        avg_faithfulness_proxy=sum(item.faithfulness_proxy for item in results) / total,
        avg_claim_hit_rate=sum(item.claim_hit_rate for item in results) / total,
        avg_forbidden_claim_rate=sum(item.forbidden_claim_rate for item in results) / total,
        avg_refusal_correctness=sum(item.refusal_correctness for item in results) / total,
        avg_relation_hit=sum(item.relation_hit for item in results) / total,
        avg_visual_warning_count=sum(item.visual_warning_count for item in results) / total,
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
            "score_version": "rag-eval-v2-claim-modality",
            "metric_groups": ["retrieval", "citation", "claim", "refusal", "relation", "visual", "table", "ocr", "judge"],
        },
        avg_score=sum(item.score for item in results) / total,
        avg_latency_ms=int(sum(item.latency_ms for item in results) / total),
    )
    if observer is not None:
        observer.record_eval_run(run)
    return run


def failed_eval_result(*, case: EvaluationCase, error: Exception, latency_ms: int) -> EvaluationResult:
    message = str(error).strip() or error.__class__.__name__
    return EvaluationResult(
        case_id=case.id,
        question=case.question,
        answer="",
        error=message[:1000],
        evidence=[],
        trace_summary={"status": "failed", "error": message[:1000], "case_metadata": case_metadata(case)},
        retrieval_hit=False,
        citation_hit=False,
        keyword_hit_rate=0.0,
        context_precision=0.0,
        context_recall=0.0,
        document_coverage=0.0,
        image_evidence_hit=False,
        visual_evidence_hit=False,
        table_evidence_hit=False,
        ocr_evidence_hit=False,
        citation_accuracy=0.0,
        answer_relevance=0.0,
        faithfulness_proxy=0.0,
        claim_hit_rate=0.0,
        forbidden_claim_rate=1.0,
        refusal_correctness=0.0,
        relation_hit=0.0,
        visual_warning_count=0,
        judge_used=False,
        judge_score=0.0,
        judge_scores={},
        judge_reason=f"case failed before judging: {message[:700]}",
        score=0.0,
        score_breakdown={"answer_generation_failed": 1.0},
        latency_ms=latency_ms,
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
    return {
        "count": float(len(items)),
        "avg_score": sum(item.score for item in items) / total,
        "retrieval_hit_rate": sum(item.retrieval_hit for item in items) / total,
        "citation_hit_rate": sum(item.citation_hit for item in items) / total,
        "avg_context_precision": sum(item.context_precision for item in items) / total,
        "avg_context_recall": sum(item.context_recall for item in items) / total,
        "avg_document_coverage": sum(item.document_coverage for item in items) / total,
        "avg_citation_accuracy": sum(item.citation_accuracy for item in items) / total,
        "avg_image_evidence_hit_rate": sum(item.image_evidence_hit for item in items) / total,
        "avg_visual_evidence_hit_rate": sum(item.visual_evidence_hit for item in items) / total,
        "avg_table_evidence_hit_rate": sum(item.table_evidence_hit for item in items) / total,
        "avg_ocr_evidence_hit_rate": sum(item.ocr_evidence_hit for item in items) / total,
        "avg_claim_hit_rate": sum(item.claim_hit_rate for item in items) / total,
        "avg_forbidden_claim_rate": sum(item.forbidden_claim_rate for item in items) / total,
        "avg_refusal_correctness": sum(item.refusal_correctness for item in items) / total,
        "avg_relation_hit": sum(item.relation_hit for item in items) / total,
        "avg_visual_warning_count": sum(item.visual_warning_count for item in items) / total,
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

    evidence_terms = clean_terms(case.expected_evidence_keywords or case.expected_keywords)
    context_precision = context_precision_proxy(terms=evidence_terms, evidence=response.evidence)
    context_recall = term_hit_rate(evidence_terms, evidence_lower)

    expected_modalities = {modality.lower() for modality in case.expected_modalities}
    visual_expected = bool(expected_modalities & {"image", "figure", "chart", "vision"})
    table_expected = "table" in expected_modalities
    ocr_expected = "ocr" in expected_modalities
    visual_evidence_hit = (
        any(
            item.image_id
            or any(marker in (item.chunk_type or "").lower() for marker in ["image", "figure", "chart"])
            for item in response.evidence
        )
        if visual_expected
        else True
    )
    table_evidence_hit = (
        any(evidence_has_table_signal(item) for item in response.evidence)
        if table_expected
        else True
    )
    ocr_evidence_hit = (
        any(
            any((image.ocr_text or "").strip() for image in getattr(item, "related_images", []))
            or "ocr" in (item.chunk_type or "").lower()
            or item.image_id
            for item in response.evidence
        )
        if ocr_expected
        else True
    )
    modality_evidence_hit = visual_evidence_hit and table_evidence_hit and ocr_evidence_hit
    citation_accuracy = citation_accuracy_proxy(answer=answer, evidence=response.evidence, trace=response.rag_trace)
    answer_relevance = answer_relevance_proxy(question=case.question, answer=answer)
    faithfulness_proxy = min(citation_accuracy, 1.0 if response.evidence else 0.0)
    claim_hit_rate = claim_hit_rate_proxy(claims=case.expected_claims, answer=answer)
    forbidden_claim_rate = forbidden_claim_rate_proxy(claims=case.forbidden_claims, answer=answer)
    refusal_correctness = refusal_correctness_proxy(case=case, answer=answer)
    relation_hit = relation_hit_proxy(case=case, answer=answer, trace=response.rag_trace)
    visual_warnings = getattr(response.rag_trace, "visual_ocr_warnings", []) or []
    visual_warning_count = sum(1 for warning in visual_warnings if str(warning.get("severity", "warn")) == "warn")

    score_breakdown = {
        "keyword_hit_rate": keyword_hit_rate,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "document_coverage": document_coverage,
        "citation_accuracy": citation_accuracy,
        "answer_relevance": answer_relevance,
        "faithfulness_proxy": faithfulness_proxy,
        "visual_evidence_hit": 1.0 if visual_evidence_hit else 0.0,
        "table_evidence_hit": 1.0 if table_evidence_hit else 0.0,
        "ocr_evidence_hit": 1.0 if ocr_evidence_hit else 0.0,
        "modality_evidence_hit": 1.0 if modality_evidence_hit else 0.0,
        "claim_hit_rate": claim_hit_rate,
        "forbidden_claim_rate": forbidden_claim_rate,
        "refusal_correctness": refusal_correctness,
        "relation_hit": relation_hit,
        "visual_warning_count": float(visual_warning_count),
    }
    score = max(
        0.0,
        (
            keyword_hit_rate * 0.14
            + context_precision * 0.10
            + context_recall * 0.14
            + document_coverage * 0.13
            + citation_accuracy * 0.14
            + answer_relevance * 0.07
            + faithfulness_proxy * 0.08
            + (1.0 if modality_evidence_hit else 0.0) * 0.04
            + claim_hit_rate * 0.08
            + refusal_correctness * 0.04
            + relation_hit * 0.04
        )
        - forbidden_claim_rate * 0.10
    )
    return {
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "keyword_hit_rate": keyword_hit_rate,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "document_coverage": document_coverage,
        "image_evidence_hit": modality_evidence_hit,
        "visual_evidence_hit": visual_evidence_hit,
        "table_evidence_hit": table_evidence_hit,
        "ocr_evidence_hit": ocr_evidence_hit,
        "citation_accuracy": citation_accuracy,
        "answer_relevance": answer_relevance,
        "faithfulness_proxy": faithfulness_proxy,
        "claim_hit_rate": claim_hit_rate,
        "forbidden_claim_rate": forbidden_claim_rate,
        "refusal_correctness": refusal_correctness,
        "relation_hit": relation_hit,
        "visual_warning_count": visual_warning_count,
        "score": score,
        "score_breakdown": score_breakdown,
    }


def evaluation_evidence_summary(response) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": item.citation_id,
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
                    "ocr_text": truncate_for_judge(image.ocr_text, 220),
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
        "final_prompt_evidence": trace.final_prompt_evidence,
        "evidence_quality": trace.evidence_quality,
        "multi_document_coverage": trace.multi_document_coverage,
        "document_relation_map": trace.document_relation_map,
        "verification": trace.verification,
        "visual_ocr_warnings": trace.visual_ocr_warnings,
    }


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

    evidence_payload = [
        {
            "citation_id": item.citation_id,
            "paper_name": item.paper_name,
            "page": item.page,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "section": item.section,
            "chunk_type": item.chunk_type,
            "image_id": item.image_id,
            "quote": truncate_for_judge(item.quote or item.text, 900),
            "text": truncate_for_judge(item.text, 900),
        }
        for item in response.evidence[:10]
    ]
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
        "evidence": evidence_payload,
        "trace_summary": {
            "retrieval_pipeline": response.rag_trace.retrieval_pipeline,
            "ranking_method": response.rag_trace.ranking_method,
            "retrieved_count": response.rag_trace.retrieved_count,
            "final_prompt_evidence": response.rag_trace.final_prompt_evidence,
            "multi_document_coverage": response.rag_trace.multi_document_coverage,
            "document_relation_map": response.rag_trace.document_relation_map,
            "visual_ocr_warnings": response.rag_trace.visual_ocr_warnings,
            "verification": response.rag_trace.verification,
        },
    }
    system_prompt = (
        "You are a strict RAG evaluation judge. Score only from the provided question, answer, evidence, and trace. "
        "Do not reward plausible claims that are not supported by evidence. "
        "Return JSON only, with scores between 0 and 1."
    )
    user_prompt = f"""
Evaluate this paper-reading assistant answer.

Scoring fields:
- answer_relevance: does the answer directly answer the user question?
- faithfulness: are claims grounded in the supplied evidence?
- citation_support: do cited evidence IDs actually support the cited claims?
- context_usage: did the answer use the most relevant retrieved evidence without overusing irrelevant context?
- multi_document_clarity: if multiple documents are involved, are documents and relationships kept clear? If not applicable, score 1.
- visual_grounding: if image/figure evidence is expected or used, is it used correctly? If not applicable, score 1.
- claim_coverage: are the expected claim-level facts covered, and are forbidden claims avoided?
- refusal_correctness: if the case expects refusal due to insufficient evidence, does the answer clearly refuse unsupported conclusions?
- completeness: does the answer cover the requested scope without major omissions?
- no_hallucination: 1 means no unsupported fabrication; 0 means severe fabrication.

Return exactly this JSON shape:
{{
  "scores": {{
    "answer_relevance": 0.0,
    "faithfulness": 0.0,
    "citation_support": 0.0,
    "context_usage": 0.0,
    "multi_document_clarity": 0.0,
    "visual_grounding": 0.0,
    "claim_coverage": 0.0,
    "refusal_correctness": 0.0,
    "completeness": 0.0,
    "no_hallucination": 0.0
  }},
  "overall": 0.0,
  "reason": "short Chinese explanation"
}}

Evaluation payload:
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
        if any((item.page_start or item.page) <= page <= (item.page_end or item.page) for item in evidence):
            matched += 1
    return matched / max(len(expected_pages), 1)


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


def forbidden_claim_rate_proxy(*, claims: list[str], answer: str) -> float:
    cleaned = clean_terms(claims)
    if not cleaned:
        return 0.0
    return term_hit_rate(cleaned, normalize_eval_text(answer))


def refusal_correctness_proxy(*, case: EvaluationCase, answer: str) -> float:
    if not expected_refusal_case(case):
        return 1.0
    return 1.0 if contains_refusal_signal(answer) else 0.0


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
