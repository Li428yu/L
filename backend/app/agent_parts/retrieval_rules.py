from __future__ import annotations

from backend.app.agent_parts.retrieval_field_rules import AgentRetrievalFieldRuleMixin
from backend.app.agent_parts.retrieval_overview_rules import AgentRetrievalOverviewRuleMixin
from backend.app.agent_parts.retrieval_special_rules import AgentRetrievalSpecialRuleMixin


class AgentRetrievalRuleMixin(
    AgentRetrievalFieldRuleMixin,
    AgentRetrievalOverviewRuleMixin,
    AgentRetrievalSpecialRuleMixin,
):
    pass
