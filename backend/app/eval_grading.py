from __future__ import annotations

from collections import Counter
from typing import Any

from backend.app.models import EvaluationResult


PASS_SCORE = 0.75
FAIL_SCORE = 0.60


def apply_eval_result_grading(result: EvaluationResult) -> EvaluationResult:
    report = grade_eval_result(result)
    result.result_status = str(report["status"])
    result.failure_categories = list(report["failure_categories"])
    result.grading_reasons = list(report["reasons"])
    result.grading_report = report
    result.trace_summary["grading"] = report
    return result


def grade_eval_result(result: EvaluationResult) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    metadata = result.trace_summary.get("case_metadata", {}) if isinstance(result.trace_summary, dict) else {}
    expected_refusal = bool(metadata.get("expected_refusal")) if metadata.get("expected_refusal") is not None else False
    expected_modalities = {str(item).lower() for item in metadata.get("expected_modalities", []) or []}
    score_breakdown = result.score_breakdown or {}

    if result.error:
        _add_issue(
            issues,
            category="answer_generation_failure",
            severity="blocked",
            reason=f"评测用例执行失败：{result.error}",
        )

    _grade_gold_evidence(result=result, issues=issues)
    _grade_retrieval(result=result, issues=issues)
    _grade_citations(result=result, issues=issues)
    _grade_refusal(result=result, score_breakdown=score_breakdown, issues=issues)
    _grade_modalities(result=result, expected_modalities=expected_modalities, issues=issues)
    _grade_claims(
        result=result,
        score_breakdown=score_breakdown,
        expected_refusal=expected_refusal,
        issues=issues,
    )
    _grade_faithfulness(result=result, score_breakdown=score_breakdown, expected_refusal=expected_refusal, issues=issues)
    _grade_operational_signals(result=result, score_breakdown=score_breakdown, issues=issues)
    _grade_score(result=result, issues=issues)

    status = _status_from_issues(issues)
    categories = _ordered_unique(str(issue["category"]) for issue in issues if issue.get("severity") in {"blocked", "fail"})
    if status == "warn":
        categories = _ordered_unique(str(issue["category"]) for issue in issues)

    return {
        "status": status,
        "score": result.score,
        "pass_score": PASS_SCORE,
        "fail_score": FAIL_SCORE,
        "failure_categories": categories,
        "reasons": [str(issue["reason"]) for issue in issues],
        "issues": issues,
    }


def build_eval_run_grading_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    status_counts = Counter(result.result_status or "ungraded" for result in results)
    category_counts: Counter[str] = Counter()
    for result in results:
        category_counts.update(result.failure_categories)
    fail_like = status_counts.get("fail", 0) + status_counts.get("blocked", 0)
    embedding_fallback_count = sum(1 for result in results if result.embedding_used_fallback)
    trust_gate_failures: list[dict[str, Any]] = []
    if embedding_fallback_count:
        trust_gate_failures.append(
            {
                "category": "embedding_fallback",
                "severity": "blocked",
                "count": embedding_fallback_count,
                "reason": "本轮评测存在 embedding fallback，召回向量空间与目标模型不一致，分数不可作为可信基线或横向对比依据。",
            }
        )
    total = max(len(results), 1)
    return {
        "status_counts": dict(status_counts),
        "failure_category_counts": dict(category_counts),
        "evaluation_trustworthy": not trust_gate_failures,
        "trust_gate_status": "not_comparable" if trust_gate_failures else "passed",
        "trust_gate_failures": trust_gate_failures,
        "pass_rate": status_counts.get("pass", 0) / total,
        "warn_rate": status_counts.get("warn", 0) / total,
        "blocked_rate": status_counts.get("blocked", 0) / total,
        "fail_rate": fail_like / total,
        "top_failure_categories": [
            {"category": category, "count": count}
            for category, count in category_counts.most_common(8)
        ],
    }


def _grade_gold_evidence(*, result: EvaluationResult, issues: list[dict[str, Any]]) -> None:
    if result.gold_chunk_count <= 0:
        return
    if result.gold_chunk_candidate_recall_at_k <= 0.0:
        _add_issue(
            issues,
            category="retrieval_failure",
            severity="fail",
            reason="标准 gold chunk 没有进入候选召回轨迹。",
            metric="gold_chunk_candidate_recall_at_k",
            value=result.gold_chunk_candidate_recall_at_k,
        )
        return
    if result.gold_chunk_recall_at_k <= 0.0:
        _add_issue(
            issues,
            category="evidence_filtering_failure",
            severity="fail",
            reason="标准 gold chunk 曾进入候选召回，但没有进入最终回答证据。",
            metric="gold_chunk_recall_at_k",
            value=result.gold_chunk_recall_at_k,
        )
        return
    if result.gold_chunk_recall_at_k < 1.0:
        category = (
            "evidence_filtering_failure"
            if result.gold_chunk_candidate_recall_at_k > result.gold_chunk_recall_at_k
            else "retrieval_failure"
        )
        _add_issue(
            issues,
            category=category,
            severity="warning",
            reason="标准 gold chunk 只被部分覆盖。",
            metric="gold_chunk_recall_at_k",
            value=result.gold_chunk_recall_at_k,
        )


def _grade_retrieval(*, result: EvaluationResult, issues: list[dict[str, Any]]) -> None:
    if not result.retrieval_hit or result.document_coverage < 1.0:
        _add_issue(
            issues,
            category="retrieval_failure",
            severity="fail",
            reason="检索证据没有覆盖期望文档。",
            metric="document_coverage",
            value=result.document_coverage,
        )
    if result.context_precision < 0.25 and result.context_recall < 0.40:
        _add_issue(
            issues,
            category="retrieval_failure",
            severity="warning",
            reason="上下文精度和召回都偏低，证据相关性不足。",
            metric="context_precision/context_recall",
            value={"context_precision": result.context_precision, "context_recall": result.context_recall},
        )


def _grade_citations(*, result: EvaluationResult, issues: list[dict[str, Any]]) -> None:
    if not result.citation_hit or result.citation_accuracy < 0.75:
        _add_issue(
            issues,
            category="citation_failure",
            severity="fail",
            reason="引用准确率过低或未命中期望页码。",
            metric="citation_accuracy",
            value=result.citation_accuracy,
        )
    elif result.citation_accuracy < 0.90:
        _add_issue(
            issues,
            category="citation_failure",
            severity="warning",
            reason="引用准确率低于可信阈值。",
            metric="citation_accuracy",
            value=result.citation_accuracy,
        )


def _grade_refusal(*, result: EvaluationResult, score_breakdown: dict[str, float], issues: list[dict[str, Any]]) -> None:
    if result.refusal_correctness < 1.0:
        _add_issue(
            issues,
            category="refusal_failure",
            severity="fail",
            reason="期望拒答的 case 没有明确拒绝无证据结论。",
            metric="refusal_correctness",
            value=result.refusal_correctness,
        )
    if float(score_breakdown.get("wrong_refusal", 0.0)) > 0.0:
        _add_issue(
            issues,
            category="refusal_failure",
            severity="fail",
            reason="有足够答案目标时错误拒答。",
            metric="wrong_refusal",
            value=score_breakdown.get("wrong_refusal"),
        )


def _grade_modalities(
    *,
    result: EvaluationResult,
    expected_modalities: set[str],
    issues: list[dict[str, Any]],
) -> None:
    visual_expected = bool(expected_modalities & {"image", "figure", "chart", "vision"})
    if visual_expected and not result.visual_evidence_hit:
        _add_issue(
            issues,
            category="visual_ocr_failure",
            severity="fail",
            reason="视觉/图片类 case 没有命中可用视觉证据。",
            metric="visual_evidence_hit",
            value=result.visual_evidence_hit,
        )
    if "table" in expected_modalities and not result.table_evidence_hit:
        _add_issue(
            issues,
            category="table_evidence_failure",
            severity="fail",
            reason="表格类 case 没有命中表格证据。",
            metric="table_evidence_hit",
            value=result.table_evidence_hit,
        )
    if "ocr" in expected_modalities and not result.ocr_evidence_hit:
        _add_issue(
            issues,
            category="visual_ocr_failure",
            severity="fail",
            reason="OCR 类 case 没有命中 OCR 文本证据。",
            metric="ocr_evidence_hit",
            value=result.ocr_evidence_hit,
        )
    if result.visual_warning_count > 0:
        _add_issue(
            issues,
            category="visual_ocr_failure",
            severity="warning",
            reason="视觉/OCR 链路存在警告，需要检查图像处理状态。",
            metric="visual_warning_count",
            value=result.visual_warning_count,
        )


def _grade_claims(
    *,
    result: EvaluationResult,
    score_breakdown: dict[str, float],
    expected_refusal: bool,
    issues: list[dict[str, Any]],
) -> None:
    if result.forbidden_claim_rate > 0.0:
        _add_issue(
            issues,
            category="claim_failure",
            severity="fail",
            reason="回答包含禁止出现的主张。",
            metric="forbidden_claim_rate",
            value=result.forbidden_claim_rate,
        )
    if expected_refusal:
        return
    has_claim_target = float(score_breakdown.get("claim_term_count", 0.0)) > 0.0
    if has_claim_target and result.claim_hit_rate < 0.50:
        _add_issue(
            issues,
            category="claim_failure",
            severity="fail",
            reason="答案没有覆盖关键主张。",
            metric="claim_hit_rate",
            value=result.claim_hit_rate,
        )
    elif has_claim_target and result.claim_hit_rate < 0.75:
        _add_issue(
            issues,
            category="claim_failure",
            severity="warning",
            reason="答案关键主张覆盖不足。",
            metric="claim_hit_rate",
            value=result.claim_hit_rate,
        )


def _grade_faithfulness(
    *,
    result: EvaluationResult,
    score_breakdown: dict[str, float],
    expected_refusal: bool,
    issues: list[dict[str, Any]],
) -> None:
    if expected_refusal:
        return
    weak_sentences = float(score_breakdown.get("faithfulness_weak_sentence_count", 0.0))
    if result.faithfulness_proxy < 0.45 or weak_sentences > 0.0:
        _add_issue(
            issues,
            category="faithfulness_failure",
            severity="fail",
            reason="回答存在证据支撑不足的句子。",
            metric="faithfulness_proxy",
            value=result.faithfulness_proxy,
        )
    elif result.faithfulness_proxy < 0.65:
        _add_issue(
            issues,
            category="faithfulness_failure",
            severity="warning",
            reason="回答忠实度代理分偏低。",
            metric="faithfulness_proxy",
            value=result.faithfulness_proxy,
        )


def _grade_operational_signals(
    *,
    result: EvaluationResult,
    score_breakdown: dict[str, float],
    issues: list[dict[str, Any]],
) -> None:
    if result.embedding_used_fallback:
        _add_issue(
            issues,
            category="embedding_fallback",
            severity="blocked",
            reason="该 case 使用了 embedding fallback，召回向量空间与目标模型不一致，评测分数不可作为可信基线。",
            metric="embedding_used_fallback",
            value=True,
        )
    if float(score_breakdown.get("final_score_capped", 0.0)) > 0.0:
        _add_issue(
            issues,
            category="score_cap",
            severity="warning",
            reason="最终分数被程序化硬上限压低。",
            metric="final_score_cap",
            value=score_breakdown.get("final_score_cap"),
        )


def _grade_score(*, result: EvaluationResult, issues: list[dict[str, Any]]) -> None:
    if result.error:
        return
    if result.score < FAIL_SCORE:
        _add_issue(
            issues,
            category="overall_score_failure",
            severity="fail",
            reason="最终分数低于失败阈值。",
            metric="score",
            value=result.score,
        )
    elif result.score < PASS_SCORE:
        _add_issue(
            issues,
            category="overall_score_warning",
            severity="warning",
            reason="最终分数低于通过阈值。",
            metric="score",
            value=result.score,
        )


def _status_from_issues(issues: list[dict[str, Any]]) -> str:
    severities = {str(issue.get("severity", "")) for issue in issues}
    if "blocked" in severities:
        return "blocked"
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warn"
    return "pass"


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    category: str,
    severity: str,
    reason: str,
    metric: str = "",
    value: Any = None,
) -> None:
    issue = {
        "category": category,
        "severity": severity,
        "reason": reason,
    }
    if metric:
        issue["metric"] = metric
    if value is not None:
        issue["value"] = value
    issues.append(issue)


def _ordered_unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
