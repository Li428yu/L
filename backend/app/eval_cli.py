from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from backend.app.agent import PaperAgentService
from backend.app.config import settings
from backend.app.evaluation import load_eval_suite, run_eval_suite
from backend.app.llm_clients import ModelClients
from backend.app.memory import MemoryManager
from backend.app.observability import ObservabilityClient
from backend.app.storage import MetadataStore
from backend.app.vector_store import ChromaPaperStore


def main() -> None:
    load_dotenv()
    settings.ensure_dirs()
    args = parse_args()

    store = MetadataStore(settings.sqlite_path)
    model_clients = ModelClients(settings)
    vector_store = ChromaPaperStore(settings.chroma_dir)
    observer = ObservabilityClient(settings)
    agent = PaperAgentService(
        settings=settings,
        store=store,
        vector_store=vector_store,
        model_clients=model_clients,
        memory=MemoryManager(store),
        observer=observer,
    )

    suite_path = resolve_suite_path(args.suite)
    cases = load_eval_suite(suite_path)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.id in wanted]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise SystemExit("没有可运行的评测用例。")

    document_ids = resolve_document_ids(store, args.documents)
    if not document_ids:
        raise SystemExit("没有可用于评测的已就绪文档。")

    run = run_eval_suite(
        suite_name=suite_path.stem,
        cases=cases,
        agent=agent,
        document_ids=document_ids,
        observer=observer,
        enable_judge=not args.no_judge,
    )
    result_path = settings.eval_results_dir / f"{run.run_id}.json"
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "suite_name": run.suite_name,
                "case_count": run.case_count,
                "document_ids": run.document_ids,
                "avg_score": run.avg_score,
                "avg_judge_score": run.avg_judge_score,
                "judge_coverage": run.judge_coverage,
                "retrieval_hit_rate": run.retrieval_hit_rate,
                "citation_hit_rate": run.citation_hit_rate,
                "avg_context_precision": run.avg_context_precision,
                "avg_context_recall": run.avg_context_recall,
                "avg_document_coverage": run.avg_document_coverage,
                "avg_citation_accuracy": run.avg_citation_accuracy,
                "avg_image_evidence_hit_rate": run.avg_image_evidence_hit_rate,
                "avg_visual_evidence_hit_rate": run.avg_visual_evidence_hit_rate,
                "avg_table_evidence_hit_rate": run.avg_table_evidence_hit_rate,
                "avg_ocr_evidence_hit_rate": run.avg_ocr_evidence_hit_rate,
                "avg_claim_hit_rate": run.avg_claim_hit_rate,
                "avg_forbidden_claim_rate": run.avg_forbidden_claim_rate,
                "avg_refusal_correctness": run.avg_refusal_correctness,
                "avg_relation_hit": run.avg_relation_hit,
                "avg_visual_warning_count": run.avg_visual_warning_count,
                "segment_metrics": run.segment_metrics,
                "avg_latency_ms": run.avg_latency_ms,
                "result_path": str(result_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行论文阅读助手评测集")
    parser.add_argument(
        "--suite",
        default="current_network_programming_eval_set",
        help="evals 目录下的评测集名称或 JSON 路径",
    )
    parser.add_argument(
        "--documents",
        default="all-ready",
        help="all-ready 或逗号分隔的 document_id 列表",
    )
    parser.add_argument("--case", action="append", help="只运行指定 case_id，可重复传入")
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 条 case")
    parser.add_argument("--no-judge", action="store_true", help="关闭 LLM-as-judge，只跑程序化指标")
    return parser.parse_args()


def resolve_suite_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".json")
    if not candidate.is_absolute():
        candidate = settings.project_root / "evals" / candidate
    resolved = candidate.resolve()
    eval_root = (settings.project_root / "evals").resolve()
    try:
        allowed = resolved.is_relative_to(eval_root)
    except AttributeError:
        allowed = str(resolved).startswith(str(eval_root))
    if not allowed:
        raise SystemExit("评测集必须放在 evals 目录下。")
    return resolved


def resolve_document_ids(store: MetadataStore, value: str) -> list[str]:
    if value.strip().lower() == "all-ready":
        return [document.id for document in store.list_documents() if document.status == "ready"]
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
