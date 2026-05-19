from __future__ import annotations

import json
import time
from pathlib import Path

from backend.app.agent import PaperAgentService
from backend.app.models import AskRequest, EvaluationCase, EvaluationResult, EvaluationRun


def load_eval_suite(path: Path) -> list[EvaluationCase]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase(**item) for item in payload.get("cases", [])]


def run_eval_suite(
    *,
    suite_name: str,
    cases: list[EvaluationCase],
    agent: PaperAgentService,
    document_ids: list[str],
) -> EvaluationRun:
    results: list[EvaluationResult] = []
    for case in cases:
        started = time.perf_counter()
        response = agent.ask(
            AskRequest(
                question=case.question,
                document_ids=document_ids,
            )
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        answer_lower = response.answer.lower()
        keyword_hits = [
            keyword
            for keyword in case.expected_keywords
            if keyword.lower() in answer_lower
        ]
        retrieval_hit = True
        if case.expected_document:
            retrieval_hit = any(
                case.expected_document.lower() in item.paper_name.lower()
                for item in response.evidence
            )
        citation_hit = True
        if case.expected_page is not None:
            citation_hit = any(item.page == case.expected_page for item in response.evidence)

        results.append(
            EvaluationResult(
                case_id=case.id,
                question=case.question,
                answer=response.answer,
                retrieval_hit=retrieval_hit,
                citation_hit=citation_hit,
                keyword_hit_rate=len(keyword_hits) / max(len(case.expected_keywords), 1),
                latency_ms=elapsed_ms,
            )
        )

    total = max(len(results), 1)
    return EvaluationRun(
        suite_name=suite_name,
        results=results,
        retrieval_hit_rate=sum(item.retrieval_hit for item in results) / total,
        citation_hit_rate=sum(item.citation_hit for item in results) / total,
        avg_keyword_hit_rate=sum(item.keyword_hit_rate for item in results) / total,
        avg_latency_ms=int(sum(item.latency_ms for item in results) / total),
    )

