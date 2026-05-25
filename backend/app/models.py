from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["queued", "processing", "ready", "failed"]
TaskStatus = Literal["queued", "running", "completed", "failed"]


class ChunkStrategy(BaseModel):
    chunk_size: int
    overlap: int
    splitter: str
    language: str
    page_count: int
    paragraph_count: int
    char_count: int
    token_count: int = 0
    size_unit: str = "characters"
    parent_chunk_size: int = 0
    parent_overlap: int = 0
    reasons: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    id: str
    file_name: str
    file_hash: str
    status: DocumentStatus
    page_count: int = 0
    chunk_count: int = 0
    source_path: str
    embedding_model: str | None = None
    chunk_strategy: ChunkStrategy | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class TaskInfo(BaseModel):
    id: str
    document_id: str | None = None
    stage: str
    status: TaskStatus
    progress: float = Field(ge=0, le=1)
    message: str
    error: str | None = None
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    document: DocumentInfo
    task: TaskInfo


class RelatedImageInfo(BaseModel):
    id: str
    document_id: str
    page_start: int
    page_end: int
    kind: str
    caption_text: str = ""
    ocr_text: str = ""
    vision_summary: str = ""
    status: str = ""


class EvidenceItem(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    paper_name: str
    page: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source: str
    file_hash: str
    score: float
    vector_score: float | None = None
    sparse_score: float | None = None
    rule_score: float | None = None
    rrf_score: float | None = None
    final_score: float | None = None
    score_source: str = ""
    text: str
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    chunk_type: str = "text"
    parent_id: str | None = None
    image_id: str | None = None
    image_path: str | None = None
    bbox_json: str | None = None
    related_images: list[RelatedImageInfo] = Field(default_factory=list)


class RuntimeStep(BaseModel):
    node: str
    title: str
    detail: str


class RetrievalDebugItem(BaseModel):
    citation_id: str
    chunk_id: str
    page: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    score: float
    vector_score: float | None = None
    sparse_score: float | None = None
    rule_score: float | None = None
    rrf_score: float | None = None
    final_score: float | None = None
    score_source: str = ""
    retrieval_strategy: str
    selected_by: str
    matched_keywords: list[str] = Field(default_factory=list)
    reason: str
    used_in_answer: bool = False
    used_in_prompt: bool = False
    quote: str


class RagTrace(BaseModel):
    model_profile: str
    vector_store: str
    vector_record_count: int
    top_k: int
    filter_document_ids: list[str]
    retrieved_count: int
    final_prompt_evidence: list[str]
    intent: str = ""
    retrieval_strategy: str = ""
    retrieval_pipeline: str = ""
    ranking_method: str = ""
    answer_strategy: str = ""
    fallback_used: bool = False
    evidence_quality: str = ""
    diagnosis: str = ""
    retrieval_debug: list[RetrievalDebugItem] = Field(default_factory=list)
    compound_tasks: list[str] = Field(default_factory=list)
    task_parse_reason: str = ""
    evidence_judgments: list[dict[str, Any]] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)
    multi_document_cards: list[dict[str, Any]] = Field(default_factory=list)
    document_relation_map: list[dict[str, Any]] = Field(default_factory=list)
    multi_document_coverage: dict[str, Any] = Field(default_factory=dict)
    visual_ocr_warnings: list[dict[str, Any]] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    model_preset: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    evidence: list[EvidenceItem]
    runtime: list[RuntimeStep]
    rag_trace: RagTrace
    memory_used: dict[str, Any] = Field(default_factory=dict)


class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    created_at: str


class ConversationDetail(BaseModel):
    conversation: ConversationInfo
    messages: list[ConversationMessage] = Field(default_factory=list)
    memory_used: dict[str, Any] = Field(default_factory=dict)


class ModelPreset(BaseModel):
    id: str
    label: str
    description: str
    chat_model: str
    embedding_model: str
    top_k: int


class ModelCatalog(BaseModel):
    presets: list[ModelPreset]
    default_preset: str
    chat_model_options: list[str] = Field(default_factory=list)
    embedding_model_options: list[str] = Field(default_factory=list)
    default_chat_model: str
    default_embedding_model: str
    default_top_k: int


class ChunkPreview(BaseModel):
    chunk_id: str
    page: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    chunk_type: str = "text"
    token_count: int | None = None
    text: str


class DocumentImageInfo(BaseModel):
    id: str
    document_id: str
    image_hash: str
    page_start: int
    page_end: int
    bbox_json: str
    image_path: str
    thumbnail_path: str
    width: int
    height: int
    kind: str
    ocr_text: str
    vision_summary: str
    caption_text: str
    status: str
    created_at: str
    updated_at: str


class EvaluationCase(BaseModel):
    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_answer: str = ""
    expected_document: str | None = None
    expected_page: int | None = None
    expected_documents: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    expected_evidence_keywords: list[str] = Field(default_factory=list)
    expected_modalities: list[str] = Field(default_factory=list)
    relation_keywords: list[str] = Field(default_factory=list)
    required_document_count: int | None = None
    expected_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_relation: str = ""
    expected_refusal: bool | None = None
    judge_rubric: str = ""


class EvaluationRunRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    suite_name: str | None = None
    suite_path: str | None = None
    case_ids: list[str] = Field(default_factory=list)
    limit: int | None = None
    enable_judge: bool | None = None
    model_preset: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    top_k: int | None = None


class EvaluationResult(BaseModel):
    case_id: str
    question: str
    answer: str
    error: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    retrieval_hit: bool
    citation_hit: bool
    keyword_hit_rate: float
    context_precision: float = 0.0
    context_recall: float = 0.0
    document_coverage: float = 1.0
    image_evidence_hit: bool = True
    visual_evidence_hit: bool = True
    table_evidence_hit: bool = True
    ocr_evidence_hit: bool = True
    citation_accuracy: float = 1.0
    answer_relevance: float = 0.0
    faithfulness_proxy: float = 0.0
    claim_hit_rate: float = 1.0
    forbidden_claim_rate: float = 0.0
    refusal_correctness: float = 1.0
    relation_hit: float = 1.0
    visual_warning_count: int = 0
    judge_used: bool = False
    judge_score: float = 0.0
    judge_scores: dict[str, float] = Field(default_factory=dict)
    judge_reason: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    latency_ms: int


class EvaluationRun(BaseModel):
    run_id: str = ""
    suite_name: str
    created_at: str = ""
    document_ids: list[str] = Field(default_factory=list)
    case_count: int = 0
    judge_enabled: bool = False
    score_version: str = "rag-eval-v1"
    results: list[EvaluationResult]
    retrieval_hit_rate: float
    citation_hit_rate: float
    avg_keyword_hit_rate: float
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_document_coverage: float = 1.0
    avg_image_evidence_hit_rate: float = 1.0
    avg_visual_evidence_hit_rate: float = 1.0
    avg_table_evidence_hit_rate: float = 1.0
    avg_ocr_evidence_hit_rate: float = 1.0
    avg_citation_accuracy: float = 1.0
    avg_answer_relevance: float = 0.0
    avg_faithfulness_proxy: float = 0.0
    avg_claim_hit_rate: float = 1.0
    avg_forbidden_claim_rate: float = 0.0
    avg_refusal_correctness: float = 1.0
    avg_relation_hit: float = 1.0
    avg_visual_warning_count: float = 0.0
    avg_judge_score: float = 0.0
    judge_coverage: float = 0.0
    segment_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    experiment_metadata: dict[str, Any] = Field(default_factory=dict)
    avg_score: float = 0.0
    avg_latency_ms: int
