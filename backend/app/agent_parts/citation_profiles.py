from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationStabilityProfile:
    name: str
    question_groups: tuple[tuple[str, ...], ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    quote_terms: tuple[str, ...]
    quote_bonus: tuple[str, ...] = ()
    avoid_terms: tuple[str, ...] = ()
    quote_limit: int = 620


def citation_stability_profiles() -> tuple[CitationStabilityProfile, ...]:
    return ()
