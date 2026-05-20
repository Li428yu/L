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


class EvidenceItem(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    paper_name: str
    page: int
    section: str | None = None
    source: str
    file_hash: str
    score: float
    text: str
    quote: str
    char_start: int | None = None
    char_end: int | None = None


class RuntimeStep(BaseModel):
    node: str
    title: str
    detail: str


class RetrievalDebugItem(BaseModel):
    citation_id: str
    chunk_id: str
    page: int
    section: str | None = None
    score: float
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
    answer_strategy: str = ""
    fallback_used: bool = False
    evidence_quality: str = ""
    diagnosis: str = ""
    retrieval_debug: list[RetrievalDebugItem] = Field(default_factory=list)
    compound_tasks: list[str] = Field(default_factory=list)
    task_parse_reason: str = ""
    evidence_judgments: list[dict[str, Any]] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)


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
    section: str | None = None
    text: str


class EvaluationCase(BaseModel):
    id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_document: str | None = None
    expected_page: int | None = None


class EvaluationResult(BaseModel):
    case_id: str
    question: str
    answer: str
    retrieval_hit: bool
    citation_hit: bool
    keyword_hit_rate: float
    latency_ms: int


class EvaluationRun(BaseModel):
    suite_name: str
    results: list[EvaluationResult]
    retrieval_hit_rate: float
    citation_hit_rate: float
    avg_keyword_hit_rate: float
    avg_latency_ms: int
