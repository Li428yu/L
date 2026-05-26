from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.app.config import Settings
from backend.app.models import AskRequest, AskResponse, EvaluationRun


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservabilityClient:
    """Best-effort local tracing for RAG runs and eval runs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_dir = settings.observability_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def record_rag_run(
        self,
        *,
        request: AskRequest,
        response: AskResponse,
        elapsed_ms: int,
    ) -> str:
        run_id = new_run_id("rag")
        payload = {
            "run_id": run_id,
            "type": "rag_run",
            "created_at": utc_now(),
            "conversation_id": response.conversation_id,
            "question": request.question,
            "document_ids": request.document_ids,
            "model_preset": request.model_preset,
            "chat_model": request.chat_model,
            "embedding_model": request.embedding_model,
            "top_k": response.rag_trace.top_k,
            "answer": response.answer,
            "elapsed_ms": elapsed_ms,
            "evidence": [
                {
                    "citation_id": item.citation_id,
                    "document_id": item.document_id,
                    "paper_name": item.paper_name,
                    "page": item.page,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "section": item.section,
                    "score": item.score,
                    "score_source": item.score_source,
                    "chunk_type": item.chunk_type,
                    "image_id": item.image_id,
                    "quote": item.quote,
                }
                for item in response.evidence
            ],
            "trace": response.rag_trace.model_dump(mode="json"),
            "scores": self._rag_scores(response=response, elapsed_ms=elapsed_ms),
        }
        self._append_jsonl("rag_runs.jsonl", payload)
        return run_id

    def record_eval_run(self, run: EvaluationRun) -> None:
        payload = run.model_dump(mode="json")
        self._write_json(f"{run.run_id or new_run_id('eval')}.json", payload, subdir="eval_runs")
        self._append_jsonl("eval_runs.jsonl", {"type": "eval_run", **payload})
        for result in run.results:
            self._record_eval_case(run=run, result=result)

    def _record_eval_case(self, *, run: EvaluationRun, result) -> None:
        payload = {
            "type": "eval_case",
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "created_at": run.created_at,
            **result.model_dump(mode="json"),
        }
        self._append_jsonl("eval_case_runs.jsonl", payload)

    def _rag_scores(self, *, response: AskResponse, elapsed_ms: int) -> dict[str, float]:
        verification = response.rag_trace.verification or {}
        citation_count = len(citation_ids_from_text(response.answer))
        available_citations = {item.citation_id for item in response.evidence}
        valid_citations = {
            citation_id for citation_id in citation_ids_from_text(response.answer)
            if citation_id in available_citations
        }
        citation_accuracy = len(valid_citations) / max(citation_count, 1)
        if verification.get("status") == "fail":
            citation_accuracy = min(citation_accuracy, 0.25)
        elif verification.get("status") == "warn":
            citation_accuracy = min(citation_accuracy, 0.65)

        evidence_quality_score = {
            "strong": 1.0,
            "medium": 0.65,
            "weak": 0.3,
            "none": 0.0,
            "insufficient": 0.0,
            "fallback": 0.35,
            "unavailable": 0.0,
        }.get(response.rag_trace.evidence_quality, 0.5)
        image_evidence_count = sum(
            1
            for item in response.evidence
            if item.image_id or "image" in (item.chunk_type or "")
        )
        return {
            "retrieved_count": float(response.rag_trace.retrieved_count),
            "prompt_evidence_count": float(len(response.rag_trace.final_prompt_evidence)),
            "citation_accuracy": citation_accuracy,
            "evidence_quality": evidence_quality_score,
            "image_evidence_count": float(image_evidence_count),
            "latency_ms": float(elapsed_ms),
        }

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.local_dir / filename
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_json(self, filename: str, payload: dict[str, Any], *, subdir: str = "") -> None:
        if subdir == "eval_runs":
            directory = self.settings.eval_results_dir
        elif subdir:
            directory = self.local_dir / subdir
        else:
            directory = self.local_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def citation_ids_from_text(text: str) -> list[str]:
    ids: list[str] = []
    for bracketed, bare in re.findall(r"\[E(\d+)\]|\bE(\d+)\b", text):
        value = bracketed or bare
        citation_id = f"E{value}"
        if citation_id not in ids:
            ids.append(citation_id)
    return ids


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"
