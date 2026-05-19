from __future__ import annotations

import re

from backend.app.storage import MetadataStore


PROFILE_PATTERNS = [
    re.compile(r"我(?:是|是一名|是一个)(?P<value>[^，。,.!\n]{2,30}(?:学生|专业|工程师|研究生|本科生|医生|教师))"),
    re.compile(r"我(?:是|是一名|是一个)(?P<value>[^，。,.!\n]{2,30})"),
    re.compile(r"我的背景是(?P<value>[^，。,.!\n]{2,40})"),
]

PREFERENCE_PATTERNS = [
    (re.compile(r"请用(?P<value>通俗|简单|专业|严谨|详细|中文|英文)[^，。,.!\n]*解释"), "explanation_style"),
    (re.compile(r"我更喜欢(?P<value>[^，。,.!\n]{2,40})"), "preference"),
]


class MemoryManager:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store

    def remember_from_user_text(self, conversation_id: str, text: str) -> dict[str, str]:
        remembered: dict[str, str] = {}
        for pattern in PROFILE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group("value").strip()
                self.store.upsert_memory_fact(
                    conversation_id=conversation_id,
                    key="user_profile",
                    value=value,
                    scope="long_term",
                )
                remembered["user_profile"] = value
                break

        for pattern, key in PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group("value").strip()
                self.store.upsert_memory_fact(
                    conversation_id=conversation_id,
                    key=key,
                    value=value,
                    scope="long_term",
                )
                remembered[key] = value
        return remembered

    def build_memory_context(self, conversation_id: str) -> dict[str, str]:
        return self.store.list_memory_facts(conversation_id)

    def render_memory_prompt(self, facts: dict[str, str]) -> str:
        if not facts:
            return "暂无长期用户画像或偏好。"
        lines = []
        if profile := facts.get("user_profile"):
            lines.append(f"- 用户画像：{profile}")
        if style := facts.get("explanation_style"):
            lines.append(f"- 解释风格偏好：{style}")
        if preference := facts.get("preference"):
            lines.append(f"- 长期偏好：{preference}")
        for key, value in facts.items():
            if key not in {"user_profile", "explanation_style", "preference"}:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
