from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from backend.app.models import EvidenceItem, RuntimeStep


class PaperAgentState(TypedDict, total=False):
    question: str
    conversation_id: str
    document_ids: list[str]
    top_k: int
    chat_model: str
    embedding_model: str
    memory_facts: dict[str, str]
    memory_prompt: str
    recent_messages: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    answer: str
    runtime: list[RuntimeStep]
    final_prompt_evidence: list[str]
    needs_retrieval: bool
    intent: str
    retrieval_strategy: str
    answer_strategy: str
    fallback_used: bool
    evidence_quality: str
    diagnosis: str
    soft_intent: dict[str, Any]
    compound_tasks: list[str]
    task_parse_reason: str
    evidence_judgments: list[dict[str, Any]]
    verification: dict[str, Any]


@dataclass
class DocumentProfile:
    document_id: str
    name: str
    title: str
    kind: str
    method: str
    main_claim: str
    has_empirical_data: bool
    has_references: bool
    is_generated_sample: bool


@dataclass(frozen=True)
class ParsedTask:
    task_type: str
    label: str
    position: int
    trigger: str


@dataclass(frozen=True)
class ReadingTaskContract:
    operation: str
    scope: str
    depth: str
    style: str
    target: str
    exclude_roles: tuple[str, ...] = ()
    role_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPlan:
    mode: str
    answer_strategy: str
    runtime_detail: str
    fallback_used: bool = False
    system_prompt: str = ""
    user_prompt: str = ""
    local_answer: str = ""
    final_prompt_evidence: list[str] = field(default_factory=list)
    prompt_evidence: list[EvidenceItem] = field(default_factory=list)
