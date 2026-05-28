from __future__ import annotations

from backend.app.agent import PaperAgentService
from backend.app.models import RuntimeStep


def make_service() -> PaperAgentService:
    return PaperAgentService.__new__(PaperAgentService)


def test_main_agent_records_routing_to_retrieval_agent() -> None:
    service = make_service()
    service._plan = lambda state: {
        **state,
        "needs_retrieval": True,
        "intent": "specific_question",
        "retrieval_strategy": "hybrid_soft",
        "runtime": [
            *state.get("runtime", []),
            RuntimeStep(node="main_agent", title="主 Agent 判断路径", detail="route"),
        ],
    }
    service._route_after_planner = lambda state: "retrieve"

    result = service._main_agent({"question": "q", "runtime": [], "agent_calls": []})

    assert result["active_agent"] == "retrieval_agent"
    assert result["agent_calls"][-1]["agent"] == "main_agent"
    assert result["agent_calls"][-1]["action"] == "route_request"
    assert result["agent_calls"][-1]["next_agent"] == "retrieval_agent"


def test_sub_agents_record_retrieval_answer_and_audit_calls() -> None:
    service = make_service()
    service._retrieve = lambda state: {
        **state,
        "retrieval_strategy": "hybrid_soft",
        "retrieval_pipeline": "dense + sparse",
        "evidence": [object(), object()],
    }
    service._answer = lambda state: {
        **state,
        "answer": "answer [E1]",
        "answer_strategy": "model_answer",
        "final_prompt_evidence": ["[E1] paper p.1 score=1.000"],
        "fallback_used": False,
    }
    service._verify_answer = lambda state: {
        **state,
        "verification": {
            "status": "pass",
            "citation_count": 1,
            "weak_citations": [],
        },
    }
    service._route_after_answer = lambda state: "verify"

    state = {"question": "q", "runtime": [], "agent_calls": []}
    state = service._retrieval_agent(state)
    state = service._answer_agent(state)
    state = service._audit_agent(state)

    assert [call["agent"] for call in state["agent_calls"]] == [
        "retrieval_agent",
        "answer_agent",
        "audit_agent",
    ]
    assert [call["action"] for call in state["agent_calls"]] == [
        "retrieve_evidence",
        "compose_answer",
        "verify_answer",
    ]
    assert state["active_agent"] == "memory_writer"


def test_missing_evidence_refusal_with_citations_still_reaches_audit_agent() -> None:
    service = make_service()
    service._citation_ids_from_answer = lambda answer: ["E1"]

    route = service._route_after_answer(
        {
            "needs_retrieval": True,
            "answer_strategy": "missing_evidence_refusal",
            "answer": "not enough evidence [E1]",
            "evidence": [object()],
        }
    )

    assert route == "verify"
