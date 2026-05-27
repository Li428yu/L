from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from backend.app.agent import PaperAgentService
from backend.app.config import settings
from backend.app.eval_baselines import (
    baseline_metadata,
    load_eval_baselines,
    resolve_eval_document_ids,
    resolve_eval_suite_path as resolve_baseline_suite_path,
)
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
    if args.list_baselines:
        print_baselines()
        return

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

    suite_path, baseline = resolve_suite_path(suite=args.suite, baseline_id=args.baseline)
    cases = load_eval_suite(suite_path)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.id in wanted]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise SystemExit("没有可运行的评测用例。")

    document_ids = resolve_document_ids(
        store=store,
        cases=cases,
        value=args.documents,
        document_policy=baseline.document_policy if baseline else "expected_ready",
    )
    if not document_ids:
        baseline_hint = f" 基线 {baseline.id} 需要的 expected_document 当前没有 ready 文档。" if baseline else ""
        raise SystemExit(f"没有可用于评测的已就绪文档。{baseline_hint}")

    run = run_eval_suite(
        suite_name=baseline.suite_name if baseline else suite_path.stem,
        cases=cases,
        agent=agent,
        document_ids=document_ids,
        observer=observer,
        model_preset=args.model_preset,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        experiment_metadata=baseline_metadata(baseline),
    )
    result_path = settings.eval_results_dir / f"{run.run_id}.json"
    metadata = baseline_metadata(baseline)
    case_summaries = [
        {
            "case_id": result.case_id,
            "status": result.result_status,
            "evidence_count": result.evidence_count,
            "citation_count": result.citation_count,
            "valid_citation_count": result.valid_citation_count,
            "evidence_keyword_hit_rate": result.evidence_keyword_hit_rate,
            "failure_categories": result.failure_categories,
            "reasons": result.grading_reasons,
        }
        for result in run.results
    ]
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "suite_name": run.suite_name,
                **metadata,
                "case_count": run.case_count,
                "pass_count": run.pass_count,
                "fail_count": run.fail_count,
                "pass_rate": run.pass_rate,
                "document_ids": run.document_ids,
                "result_status_counts": run.result_status_counts,
                "failure_category_counts": run.failure_category_counts,
                "evaluation_trustworthy": run.evaluation_trustworthy,
                "retrieval_hit_rate": run.retrieval_hit_rate,
                "citation_hit_rate": run.citation_hit_rate,
                "avg_document_coverage": run.avg_document_coverage,
                "avg_citation_accuracy": run.avg_citation_accuracy,
                "embedding_fallback_count": run.embedding_fallback_count,
                "embedding_fallback_rate": run.embedding_fallback_rate,
                "avg_latency_ms": run.avg_latency_ms,
                "cases": case_summaries,
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
        default=None,
        help="evals 目录下的评测集名称或 JSON 路径；未指定时使用默认 active baseline",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="评测基线 ID，例如 pdf_gold_current；未指定时使用 evals/baselines.json 的 default_baseline",
    )
    parser.add_argument(
        "--documents",
        default="baseline",
        help="baseline、all-ready 或逗号分隔的 document_id 列表",
    )
    parser.add_argument("--case", action="append", help="只运行指定 case_id，可重复传入")
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 条 case")
    parser.add_argument("--model-preset", default=None, help="指定阅读模式，例如 balanced/careful/quick")
    parser.add_argument("--chat-model", default=None, help="覆盖对话模型")
    parser.add_argument("--embedding-model", default=None, help="覆盖 embedding 模型")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖检索 top-k")
    parser.add_argument("--list-baselines", action="store_true", help="列出可用评测基线后退出")
    return parser.parse_args()


def resolve_suite_path(*, suite: str | None, baseline_id: str | None):
    try:
        return resolve_baseline_suite_path(
            eval_dir=settings.project_root / "evals",
            suite_name=suite,
            baseline_id=baseline_id,
        )
    except ValueError:
        raise SystemExit("评测集必须放在 evals 目录下。")


def resolve_document_ids(
    *,
    store: MetadataStore,
    cases,
    value: str,
    document_policy: str,
) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all-ready":
        return [document.id for document in store.list_documents() if document.status == "ready"]
    if normalized in {"baseline", "expected-ready", "expected_ready"}:
        return resolve_eval_document_ids(
            documents=store.list_documents(),
            cases=cases,
            document_policy=document_policy,
        )
    return [item.strip() for item in value.split(",") if item.strip()]


def print_baselines() -> None:
    manifest = load_eval_baselines(settings.project_root / "evals")
    print(
        json.dumps(
            {
                "default_baseline": manifest.default_baseline,
                "baselines": [
                    {
                        "id": baseline.id,
                        "label": baseline.label,
                        "tier": baseline.tier,
                        "status": baseline.status,
                        "suite_name": baseline.suite_name,
                        "suite_path": baseline.suite_path,
                        "document_policy": baseline.document_policy,
                        "pause_reason": baseline.pause_reason,
                    }
                    for baseline in manifest.baselines
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
