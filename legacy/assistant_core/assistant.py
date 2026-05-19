from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

from assistant_core.documents import PaperDocumentMixin
from assistant_core.graph import PaperAgentGraphMixin
from assistant_core.planner import PaperPlannerMixin
from assistant_core.retrieval import PaperRetrievalMixin
from assistant_core.tracing import AgentTraceMixin


class PaperAssistant(
    PaperAgentGraphMixin,
    PaperPlannerMixin,
    AgentTraceMixin,
    PaperRetrievalMixin,
    PaperDocumentMixin,
):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("API_BASE_URL")

        self.api_key = resolved_api_key
        self.base_url = (resolved_base_url or "").rstrip("/")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

        self.llm = ChatOpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=self.llm_model,
            temperature=0,
        )
        self.embedding_client = OpenAIEmbeddings(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=self.embedding_model,
            check_embedding_ctx_length=False,
        )
        self.agent_memory = InMemorySaver()
