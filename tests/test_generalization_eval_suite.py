from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.app.eval_baselines import load_eval_baselines, resolve_eval_suite_path
from backend.app.eval_cli import render_eval_markdown_report
from backend.app.evaluation import (
    answer_point_coverage,
    audit_gold_evidence_stages,
    eval_mode_for_cases,
    is_gold_case,
    load_eval_suite,
)
from backend.app.models import EvaluationCase, EvaluationResult, EvaluationRun, EvidenceItem


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evals"


def test_generalization_baseline_is_registered_and_resolves() -> None:
    manifest = load_eval_baselines(EVAL_DIR)
    baseline = manifest.by_id()["generalization_gold_v1"]

    suite_path, resolved = resolve_eval_suite_path(eval_dir=EVAL_DIR, baseline_id="generalization_gold_v1")

    assert resolved is not None
    assert resolved.id == baseline.id
    assert baseline.tier == "gold"
    assert baseline.document_policy == "expected_ready"
    assert suite_path.name == "generalization_gold_eval_set.json"


def test_generalization_suite_is_gold_and_uses_unseen_documents() -> None:
    cases = load_eval_suite(EVAL_DIR / "generalization_gold_eval_set.json")
    documents = {case.expected_document for case in cases if case.expected_document}
    for case in cases:
        documents.update(case.expected_documents)

    assert len(cases) == 10
    assert eval_mode_for_cases(cases) == "gold_rag_benchmark"
    assert all(is_gold_case(case) for case in cases)
    assert documents == {"efficientnet.pdf", "focal-loss.pdf", "simclr.pdf", "lora.pdf"}
    assert any(case.expected_refusal for case in cases)
    assert any(case.required_document_count == 2 for case in cases)

    suite_text = (EVAL_DIR / "generalization_gold_eval_set.json").read_text(encoding="utf-8").lower()
    for forbidden in ["attention-is-all-you-need", "bert.pdf", "unet.pdf", "wmt 2014", "bookscorpus"]:
        assert forbidden not in suite_text


def test_markdown_eval_report_renders_key_quality_metrics() -> None:
    result = EvaluationResult(
        case_id="case-1",
        question="Question?",
        answer="Answer [E1]",
        evidence=[],
        trace_summary={},
        evidence_count=1,
        citation_count=1,
        valid_citation_count=1,
        evidence_keyword_hit_rate=1.0,
        evidence_document_hit=True,
        retrieval_hit=True,
        citation_hit=True,
        keyword_hit_rate=1.0,
        context_precision=1.0,
        context_recall=1.0,
        document_coverage=1.0,
        citation_accuracy=1.0,
        embedding_used_fallback=False,
        score=1.0,
        score_breakdown={},
        result_status="pass",
        failure_categories=[],
        grading_reasons=[],
        grading_report={},
        latency_ms=12,
    )
    run = EvaluationRun(
        run_id="eval_test",
        suite_name="generalization_gold_v1",
        created_at="2026-05-28T00:00:00Z",
        document_ids=["doc-1"],
        case_count=1,
        pass_count=1,
        fail_count=0,
        pass_rate=1.0,
        score_version="gold-rag-v1",
        results=[result],
        retrieval_hit_rate=1.0,
        citation_hit_rate=1.0,
        avg_keyword_hit_rate=1.0,
        avg_context_precision=1.0,
        avg_context_recall=1.0,
        avg_document_coverage=1.0,
        avg_citation_accuracy=1.0,
        embedding_fallback_count=0,
        embedding_fallback_rate=0.0,
        result_status_counts={"pass": 1},
        failure_category_counts={},
        grading_summary={},
        evaluation_trustworthy=True,
        experiment_metadata={},
        avg_score=1.0,
        avg_latency_ms=12,
    )

    report = render_eval_markdown_report(
        run,
        metadata={"baseline_label": "Generalization"},
        result_path=Path("data/eval_runs/eval_test.json"),
    )

    assert "Evaluation Report: generalization_gold_v1" in report
    assert "Avg retrieval recall at k: 1.000" in report
    assert "Avg citation support rate: 1.000" in report
    assert "| case-1 | pass | 1.000 | 1 | 1/1 | - |" in report


def test_gold_evidence_audit_tracks_candidate_to_citation_stages() -> None:
    case = EvaluationCase(
        id="audit",
        question="How does the method scale?",
        gold_evidence=[
            {
                "document": "paper.pdf",
                "text_contains": ["compound coefficient", "width, depth, and resolution"],
            }
        ],
    )
    evidence = EvidenceItem(
        citation_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_name="paper.pdf",
        page=1,
        source="paper.pdf",
        file_hash="hash",
        score=0.9,
        text="The method uses a compound coefficient to scale network width, depth, and resolution.",
        quote="compound coefficient to scale network width, depth, and resolution",
    )
    response = SimpleNamespace(
        answer="It uses a compound coefficient [E1].",
        evidence=[evidence],
        rag_trace=SimpleNamespace(
            final_prompt_evidence=["[E1] paper.pdf p.1 score=0.900"],
            evidence_quality_trace=[
                {
                    "citation_id": "E1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "paper_name": "paper.pdf",
                    "page": 1,
                    "candidate_rank": 1,
                    "selected_rank": 1,
                    "selection_status": "selected_by_retrieval_filter",
                    "score": 0.9,
                    "score_source": "rrf_fusion",
                    "quote": "compound coefficient to scale network width, depth, and resolution",
                }
            ],
        ),
    )
    agent = SimpleNamespace(
        vector_store=SimpleNamespace(
            get_document_chunks=lambda document_id, limit=1000: [
                {
                    "id": "chunk-1",
                    "text": evidence.text,
                    "metadata": {
                        "chunk_id": "chunk-1",
                        "paper_name": "paper.pdf",
                        "page": 1,
                    },
                }
            ]
        )
    )

    audit = audit_gold_evidence_stages(case=case, response=response, agent=agent)

    assert audit["summary"] == {
        "candidate": 1,
        "retrieval_selected": 1,
        "prompt": 1,
        "visible": 1,
        "cited": 1,
        "gold_count": 1,
    }


def test_answer_point_coverage_accepts_generic_bilingual_concepts() -> None:
    answer = (
        "该方法处理类别不平衡，并降低易分类样本的损失权重，"
        "让模型更关注难分类样本。"
    )

    coverage = answer_point_coverage(
        ["class imbalance", "down-weights", "well-classified examples", "hard examples"],
        answer,
    )

    assert coverage == 1.0


def test_answer_point_coverage_keeps_numeric_and_resource_concepts_strict() -> None:
    answer = "训练时可训练参数减少一万倍，并把显存需求降低三倍。"

    coverage = answer_point_coverage(["10,000 times", "GPU memory", "3 times"], answer)

    assert coverage == 1.0
