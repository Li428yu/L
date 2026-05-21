from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    type: str
    payload: object


def status_event(message: str) -> AgentEvent:
    return AgentEvent(type="status", payload=message)


def token_event(text: str) -> AgentEvent:
    return AgentEvent(type="token", payload=text)


def final_event(payload: object) -> AgentEvent:
    return AgentEvent(type="final", payload=payload)


def error_event(message: str) -> AgentEvent:
    return AgentEvent(type="error", payload=message)
