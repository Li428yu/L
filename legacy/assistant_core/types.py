from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass
class Chunk:
    chunk_id: int
    paper_id: str
    paper_name: str
    page: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class PaperOverview:
    paper_id: str
    paper_name: str
    page_count: int
    chunk_count: int


@dataclass
class ToolTrace:
    tool_name: str
    tool_args: dict[str, Any]
    tool_output: str


@dataclass
class NodeExecutionLog:
    step_index: int
    node_name: str
    summary: str


@dataclass
class AgentTurnResult:
    answer: str
    evidence: list[RetrievedChunk]
    tool_traces: list[ToolTrace]
    messages: list[BaseMessage]
    runtime_node_logs: list[NodeExecutionLog]
    traversed_nodes: list[str]
    graph_mermaid: str
