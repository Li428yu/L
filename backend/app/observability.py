from __future__ import annotations

import json
import re
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from backend.app.config import Settings
from backend.app.models import AskRequest, AskResponse, EvaluationRun


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservabilityClient:
    """Best-effort local and Langfuse tracing for RAG runs and eval runs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_dir = settings.observability_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.langfuse = self._build_langfuse_client()

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
        self._record_langfuse_trace(
            name="paper-assistant-rag",
            trace_id=run_id,
            input_payload={
                "question": request.question,
                "document_ids": request.document_ids,
            },
            output_payload={"answer": response.answer},
            metadata={
                "conversation_id": response.conversation_id,
                "retrieval_pipeline": response.rag_trace.retrieval_pipeline,
                "ranking_method": response.rag_trace.ranking_method,
                "answer_strategy": response.rag_trace.answer_strategy,
                "evidence_quality": response.rag_trace.evidence_quality,
                "elapsed_ms": elapsed_ms,
            },
            scores=payload["scores"],
        )
        return run_id

    def record_eval_run(self, run: EvaluationRun) -> None:
        payload = run.model_dump(mode="json")
        self._write_json(f"{run.run_id or new_run_id('eval')}.json", payload, subdir="eval_runs")
        self._append_jsonl("eval_runs.jsonl", {"type": "eval_run", **payload})
        self._write_langfuse_dataset_exports(run)
        for result in run.results:
            self._record_eval_case(run=run, result=result)
        self._record_langfuse_trace(
            name=f"paper-assistant-eval-{run.suite_name}",
            trace_id=run.run_id or new_run_id("eval"),
            input_payload={
                "suite_name": run.suite_name,
                "case_count": len(run.results),
                "document_ids": run.document_ids,
            },
            output_payload={
                "avg_score": run.avg_score,
                "avg_judge_score": run.avg_judge_score,
                "retrieval_hit_rate": run.retrieval_hit_rate,
                "citation_hit_rate": run.citation_hit_rate,
                "segment_metrics": run.segment_metrics,
            },
            metadata={
                "created_at": run.created_at,
                "judge_enabled": run.judge_enabled,
                "score_version": run.score_version,
                "experiment_metadata": run.experiment_metadata,
                "segment_metrics": run.segment_metrics,
            },
            scores={
                "avg_score": run.avg_score,
                "retrieval_hit_rate": run.retrieval_hit_rate,
                "citation_hit_rate": run.citation_hit_rate,
                "avg_keyword_hit_rate": run.avg_keyword_hit_rate,
                "avg_context_precision": run.avg_context_precision,
                "avg_context_recall": run.avg_context_recall,
                "avg_document_coverage": run.avg_document_coverage,
                "avg_citation_accuracy": run.avg_citation_accuracy,
                "avg_visual_evidence_hit_rate": run.avg_visual_evidence_hit_rate,
                "avg_table_evidence_hit_rate": run.avg_table_evidence_hit_rate,
                "avg_ocr_evidence_hit_rate": run.avg_ocr_evidence_hit_rate,
                "avg_claim_hit_rate": run.avg_claim_hit_rate,
                "avg_forbidden_claim_rate": run.avg_forbidden_claim_rate,
                "avg_refusal_correctness": run.avg_refusal_correctness,
                "avg_relation_hit": run.avg_relation_hit,
                "avg_visual_warning_count": run.avg_visual_warning_count,
                "avg_judge_score": run.avg_judge_score,
                "judge_coverage": run.judge_coverage,
                "avg_latency_ms": float(run.avg_latency_ms),
            },
        )

    def _record_eval_case(self, *, run: EvaluationRun, result) -> None:
        payload = {
            "type": "eval_case",
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "created_at": run.created_at,
            **result.model_dump(mode="json"),
        }
        self._append_jsonl("eval_case_runs.jsonl", payload)
        score_payload = {
            "score": result.score,
            "retrieval_hit": 1.0 if result.retrieval_hit else 0.0,
            "citation_hit": 1.0 if result.citation_hit else 0.0,
            "keyword_hit_rate": result.keyword_hit_rate,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "document_coverage": result.document_coverage,
            "image_evidence_hit": 1.0 if result.image_evidence_hit else 0.0,
            "visual_evidence_hit": 1.0 if result.visual_evidence_hit else 0.0,
            "table_evidence_hit": 1.0 if result.table_evidence_hit else 0.0,
            "ocr_evidence_hit": 1.0 if result.ocr_evidence_hit else 0.0,
            "citation_accuracy": result.citation_accuracy,
            "answer_relevance": result.answer_relevance,
            "faithfulness_proxy": result.faithfulness_proxy,
            "claim_hit_rate": result.claim_hit_rate,
            "forbidden_claim_rate": result.forbidden_claim_rate,
            "refusal_correctness": result.refusal_correctness,
            "relation_hit": result.relation_hit,
            "visual_warning_count": float(result.visual_warning_count),
            "judge_score": result.judge_score,
            "latency_ms": float(result.latency_ms),
        }
        for key, value in result.judge_scores.items():
            score_payload[f"judge_{key}"] = value
        self._record_langfuse_trace(
            name=f"paper-assistant-eval-case-{run.suite_name}",
            trace_id=f"{run.run_id}_{safe_trace_suffix(result.case_id)}",
            input_payload={
                "case_id": result.case_id,
                "question": result.question,
                "document_ids": run.document_ids,
            },
            output_payload={
                "answer": result.answer,
                "evidence": result.evidence,
                "judge_reason": result.judge_reason,
            },
            metadata={
                "suite_name": run.suite_name,
                "run_id": run.run_id,
                "judge_used": result.judge_used,
                "trace_summary": result.trace_summary,
            },
            scores=score_payload,
        )

    def _write_langfuse_dataset_exports(self, run: EvaluationRun) -> None:
        dataset_dir = self.local_dir / "langfuse_datasets"
        experiment_dir = self.local_dir / "langfuse_experiments"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = dataset_dir / f"{safe_trace_suffix(run.suite_name)}.jsonl"
        with dataset_path.open("w", encoding="utf-8") as file:
            for result in run.results:
                metadata = dict(result.trace_summary.get("case_metadata") or {})
                dataset_item = {
                    "dataset_name": run.suite_name,
                    "item_id": result.case_id,
                    "input": {"question": result.question, "document_ids": run.document_ids},
                    "expected_output": {
                        "answer": metadata.get("expected_answer", ""),
                        "keywords": metadata.get("expected_keywords", []),
                        "evidence_keywords": metadata.get("expected_evidence_keywords", []),
                        "documents": metadata.get("expected_documents") or (
                            [metadata.get("expected_document")] if metadata.get("expected_document") else []
                        ),
                        "pages": metadata.get("expected_pages") or (
                            [metadata.get("expected_page")] if metadata.get("expected_page") else []
                        ),
                        "modalities": metadata.get("expected_modalities", []),
                        "claims": metadata.get("expected_claims", []),
                        "forbidden_claims": metadata.get("forbidden_claims", []),
                        "relation": metadata.get("expected_relation", ""),
                        "refusal": metadata.get("expected_refusal"),
                    },
                    "metadata": {
                        "tags": metadata.get("tags", []),
                        "judge_rubric": metadata.get("judge_rubric", ""),
                        "required_document_count": metadata.get("required_document_count"),
                    },
                }
                file.write(json.dumps(dataset_item, ensure_ascii=False) + "\n")

        experiment_payload = {
            "experiment_name": run.run_id,
            "dataset_name": run.suite_name,
            "created_at": run.created_at,
            "score_version": run.score_version,
            "metadata": run.experiment_metadata,
            "summary_scores": {
                "avg_score": run.avg_score,
                "retrieval_hit_rate": run.retrieval_hit_rate,
                "citation_hit_rate": run.citation_hit_rate,
                "avg_context_precision": run.avg_context_precision,
                "avg_context_recall": run.avg_context_recall,
                "avg_document_coverage": run.avg_document_coverage,
                "avg_visual_evidence_hit_rate": run.avg_visual_evidence_hit_rate,
                "avg_table_evidence_hit_rate": run.avg_table_evidence_hit_rate,
                "avg_ocr_evidence_hit_rate": run.avg_ocr_evidence_hit_rate,
                "avg_claim_hit_rate": run.avg_claim_hit_rate,
                "avg_forbidden_claim_rate": run.avg_forbidden_claim_rate,
                "avg_refusal_correctness": run.avg_refusal_correctness,
                "avg_relation_hit": run.avg_relation_hit,
                "avg_judge_score": run.avg_judge_score,
                "judge_coverage": run.judge_coverage,
            },
            "segment_metrics": run.segment_metrics,
            "results": [
                {
                    "case_id": result.case_id,
                    "score": result.score,
                    "retrieval_hit": result.retrieval_hit,
                    "citation_hit": result.citation_hit,
                    "document_coverage": result.document_coverage,
                    "claim_hit_rate": result.claim_hit_rate,
                    "forbidden_claim_rate": result.forbidden_claim_rate,
                    "refusal_correctness": result.refusal_correctness,
                    "relation_hit": result.relation_hit,
                    "visual_warning_count": result.visual_warning_count,
                    "judge_used": result.judge_used,
                    "judge_score": result.judge_score,
                    "judge_scores": result.judge_scores,
                    "judge_reason": result.judge_reason,
                }
                for result in run.results
            ],
        }
        self._write_json(f"{run.run_id}.json", experiment_payload, subdir="langfuse_experiments")

    def _build_langfuse_client(self):
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse import Langfuse
        except Exception:
            return None

        kwargs: dict[str, str] = {}
        if self.settings.langfuse_public_key:
            kwargs["public_key"] = self.settings.langfuse_public_key
        if self.settings.langfuse_secret_key:
            kwargs["secret_key"] = self.settings.langfuse_secret_key
        if self.settings.langfuse_host:
            kwargs["host"] = self.settings.langfuse_host
        try:
            return Langfuse(**kwargs)
        except Exception:
            return None

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

    def _record_langfuse_trace(
        self,
        *,
        name: str,
        trace_id: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        metadata: dict[str, Any],
        scores: dict[str, float],
    ) -> None:
        if self.langfuse is None:
            return

        trace = None
        with suppress(Exception):
            trace = self.langfuse.trace(
                id=trace_id,
                name=name,
                input=input_payload,
                output=output_payload,
                metadata=metadata,
            )
        if trace is None:
            with suppress(Exception):
                trace = self.langfuse.trace(
                    name=name,
                    input=input_payload,
                    output=output_payload,
                    metadata={**metadata, "local_trace_id": trace_id},
                )
        for score_name, value in scores.items():
            self._record_langfuse_score(trace=trace, trace_id=trace_id, name=score_name, value=value)
        with suppress(Exception):
            self.langfuse.flush()

    def _record_langfuse_score(
        self,
        *,
        trace: Any,
        trace_id: str,
        name: str,
        value: float,
    ) -> None:
        if trace is not None and hasattr(trace, "score"):
            with suppress(Exception):
                trace.score(name=name, value=value)
                return
        if self.langfuse is not None and hasattr(self.langfuse, "score"):
            with suppress(Exception):
                self.langfuse.score(trace_id=trace_id, name=name, value=value)

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


def safe_trace_suffix(value: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return suffix[:80] or uuid.uuid4().hex[:8]
