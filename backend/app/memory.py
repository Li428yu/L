from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.app.storage import MetadataStore


STOP_CHARS = r"\uff0c\u3002\uff1b\uff1a\u3001,.!?;:\n"


@dataclass(frozen=True)
class ExtractedMemory:
    memory_type: str
    key: str
    value: str
    confidence: float = 0.86
    source: str = "rule"


PROFILE_PATTERNS: list[tuple[re.Pattern[str], str, str, float]] = [
    (
        re.compile(
            rf"(?:\u6211(?:\u7684)?(?:\u80cc\u666f|\u8eab\u4efd)\u662f|\u6211(?:\u662f|\u662f\u4e00\u540d|\u662f\u4e00\u4e2a))"
            rf"(?P<value>[^{STOP_CHARS}]{{2,40}}(?:\u5b66\u751f|\u8001\u5e08|\u6559\u5e08|\u7814\u7a76\u751f|\u672c\u79d1\u751f|\u5de5\u7a0b\u5e08|\u533b\u751f))"
        ),
        "profile",
        "user_profile",
        0.92,
    ),
    (
        re.compile(
            rf"(?:\u6211\u7684\u4e13\u4e1a\u662f|\u6211\u5b66\u7684\u662f|\u6211\u6765\u81ea)"
            rf"(?P<value>[^{STOP_CHARS}]{{2,24}})"
        ),
        "profile",
        "major",
        0.9,
    ),
    (
        re.compile(
            rf"\u6211\u662f(?P<value>[^{STOP_CHARS}]{{2,24}}\u4e13\u4e1a)"
            rf"(?:\u5b66\u751f|\u7814\u7a76\u751f|\u672c\u79d1\u751f)?"
        ),
        "profile",
        "major",
        0.9,
    ),
]

PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str, str, float]] = [
    (
        re.compile(
            rf"(?:\u8bf7|\u5e0c\u671b\u4f60|\u4f60)(?:\u5c3d\u91cf)?\u7528"
            rf"(?P<value>\u901a\u4fd7|\u7b80\u5355|\u4e13\u4e1a|\u4e25\u8c28|\u8be6\u7ec6|\u4e2d\u6587|\u82f1\u6587|\u975e\u6280\u672f\u8bed\u8a00)"
            rf"[^{STOP_CHARS}]{{0,30}}(?:\u89e3\u91ca|\u56de\u7b54|\u8bf4)"
        ),
        "preference",
        "explanation_style",
        0.9,
    ),
    (
        re.compile(rf"\u6211(?:\u66f4)?\u559c\u6b22(?P<value>[^{STOP_CHARS}]{{2,40}})"),
        "preference",
        "preference",
        0.84,
    ),
    (
        re.compile(
            rf"\u6211(?:\u5e0c\u671b|\u60f3|\u9700\u8981)(?:\u4f60)?"
            rf"(?P<value>[^{STOP_CHARS}]{{2,48}})"
        ),
        "goal",
        "reading_goal",
        0.78,
    ),
]


class MemoryManager:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store

    def remember_from_user_text(
        self,
        conversation_id: str,
        text: str,
        *,
        source_message_id: int | None = None,
    ) -> dict[str, str]:
        remembered: dict[str, str] = {}
        for memory in self._extract_rule_memories(text):
            self.store.upsert_memory_item(
                conversation_id=conversation_id,
                memory_type=memory.memory_type,
                key=memory.key,
                value=memory.value,
                source=memory.source,
                source_message_id=source_message_id,
                confidence=memory.confidence,
            )
            remembered[memory.key] = memory.value
        return remembered

    def build_memory_context(
        self,
        conversation_id: str,
        *,
        question: str | None = None,
    ) -> dict[str, Any]:
        items = self.store.list_memory_items(conversation_id)
        facts: dict[str, Any] = self.store.list_memory_facts(conversation_id)
        facts.update({
            str(item["key"]): str(item["value"])
            for item in items
            if str(item.get("memory_type")) in {"profile", "preference", "goal", "task_state"}
        })

        summary = self.store.get_conversation_summary(conversation_id)
        if summary and str(summary.get("summary", "")).strip():
            facts["conversation_summary"] = str(summary["summary"]).strip()

        relevant = self._select_relevant_memories(items, question or "")
        if relevant:
            facts["relevant_memory"] = "\n".join(
                f"- {item['key']}: {item['value']}" for item in relevant
            )
            self.store.touch_memory_items([str(item["id"]) for item in relevant])
        return facts

    def render_memory_prompt(self, facts: dict[str, Any]) -> str:
        if not facts:
            return "\u6682\u65e0\u957f\u671f\u7528\u6237\u753b\u50cf\u6216\u504f\u597d\u3002"

        labels = {
            "user_profile": "\u7528\u6237\u753b\u50cf",
            "major": "\u4e13\u4e1a\u80cc\u666f",
            "explanation_style": "\u89e3\u91ca\u98ce\u683c\u504f\u597d",
            "preference": "\u957f\u671f\u504f\u597d",
            "reading_goal": "\u9605\u8bfb\u76ee\u6807",
            "conversation_summary": "\u4f1a\u8bdd\u9636\u6bb5\u6458\u8981",
            "relevant_memory": "\u76f8\u5173\u5386\u53f2\u8bb0\u5fc6",
        }
        ordered_keys = [
            "user_profile",
            "major",
            "explanation_style",
            "preference",
            "reading_goal",
            "conversation_summary",
            "relevant_memory",
        ]
        lines: list[str] = []
        for key in ordered_keys:
            value = facts.get(key)
            if value:
                lines.append(f"- {labels[key]}: {value}")
        for key, value in facts.items():
            if key not in labels and value:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def refresh_conversation_summary(
        self,
        conversation_id: str,
        *,
        min_messages: int = 40,
        refresh_every: int = 30,
    ) -> str:
        message_count = self.store.count_messages(conversation_id)
        existing = self.store.get_conversation_summary(conversation_id)
        covered = int(existing.get("covered_message_count", 0)) if existing else 0
        if message_count < min_messages or message_count - covered < refresh_every:
            return str(existing.get("summary", "")) if existing else ""

        messages = self.store.get_latest_messages(conversation_id, limit=80)
        summary = self._build_heuristic_summary(
            existing_summary=str(existing.get("summary", "")) if existing else "",
            messages=messages,
        )
        if summary:
            self.store.upsert_conversation_summary(
                conversation_id=conversation_id,
                summary=summary,
                covered_message_count=message_count,
            )
        return summary

    def _extract_rule_memories(self, text: str) -> list[ExtractedMemory]:
        memories: list[ExtractedMemory] = []
        for pattern, memory_type, key, confidence in [*PROFILE_PATTERNS, *PREFERENCE_PATTERNS]:
            match = pattern.search(text)
            if not match:
                continue
            value = self._clean_value(match.group("value"))
            if not self._is_stable_memory_value(value):
                continue
            memories.append(
                ExtractedMemory(
                    memory_type=memory_type,
                    key=key,
                    value=value,
                    confidence=confidence,
                )
            )
        return self._dedupe_memories(memories)

    def _select_relevant_memories(
        self,
        items: list[dict[str, Any]],
        question: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        if not question:
            return []
        question_tokens = set(self._tokens(question))
        if not question_tokens:
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            key = str(item.get("key", ""))
            value = str(item.get("value", ""))
            memory_type = str(item.get("memory_type", ""))
            if memory_type in {"profile", "preference", "goal"}:
                continue
            overlap = len(question_tokens & set(self._tokens(f"{key} {value}")))
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _build_heuristic_summary(
        self,
        *,
        existing_summary: str,
        messages: list[dict[str, Any]],
    ) -> str:
        user_messages = [
            str(row.get("content", "")).strip()
            for row in messages
            if row.get("role") == "user" and str(row.get("content", "")).strip()
        ]
        if not user_messages:
            return existing_summary

        extracted: dict[str, str] = {}
        for text in user_messages:
            for memory in self._extract_rule_memories(text):
                extracted[memory.key] = memory.value

        recent_topics = [self._compact_sentence(text) for text in user_messages[-8:]]
        recent_topics = [text for text in recent_topics if text][:5]

        parts: list[str] = []
        if existing_summary:
            parts.append(self._truncate(existing_summary, 600))
        if extracted:
            profile_bits = "; ".join(f"{key}={value}" for key, value in extracted.items())
            parts.append(f"Stable profile/preference updates: {profile_bits}.")
        if recent_topics:
            parts.append("Recent focus: " + " | ".join(recent_topics) + ".")

        summary = " ".join(parts)
        return self._truncate(summary, 1200)

    def _clean_value(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \t\r\n\uff0c\u3002\uff1b;,.")

    def _is_stable_memory_value(self, value: str) -> bool:
        if len(value) < 2 or len(value) > 80:
            return False
        volatile_markers = [
            "\u4eca\u5929",
            "\u660e\u5929",
            "\u521a\u624d",
            "\u8fd9\u4e00\u6b21",
            "\u672c\u8f6e",
        ]
        return not any(marker in value for marker in volatile_markers)

    def _dedupe_memories(self, memories: list[ExtractedMemory]) -> list[ExtractedMemory]:
        by_key: dict[tuple[str, str], ExtractedMemory] = {}
        for memory in memories:
            identity = (memory.memory_type, memory.key)
            existing = by_key.get(identity)
            if existing is None or memory.confidence >= existing.confidence:
                by_key[identity] = memory
        return list(by_key.values())

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]", lowered)
        cjk = [word for word in words if re.fullmatch(r"[\u4e00-\u9fff]", word)]
        if len(cjk) >= 2:
            words.extend("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
        return words[:400]

    def _compact_sentence(self, text: str) -> str:
        cleaned = self._clean_value(text)
        cleaned = re.sub(r"\[[Ee]\d+\]", "", cleaned)
        return self._truncate(cleaned, 120)

    def _truncate(self, text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
