from __future__ import annotations

import uuid
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage


class PaperPlannerMixin:
    def _resolve_paper_names(
        self,
        requested_names: Sequence[str],
        available_names: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        unmatched: list[str] = []
        available_by_lower = {name.lower(): name for name in available_names}

        for requested_name in requested_names:
            cleaned = requested_name.strip()
            if not cleaned:
                continue

            exact_match = available_by_lower.get(cleaned.lower())
            if exact_match:
                if exact_match not in matched:
                    matched.append(exact_match)
                continue

            partial_matches = [
                name
                for name in available_names
                if cleaned.lower() in name.lower() or name.lower() in cleaned.lower()
            ]
            if len(partial_matches) == 1:
                if partial_matches[0] not in matched:
                    matched.append(partial_matches[0])
            else:
                unmatched.append(cleaned)

        return matched, unmatched

    def _build_planner_tool_call(
        self,
        messages: Sequence[BaseMessage],
        available_names: Sequence[str],
        default_scope_names: Sequence[str],
    ) -> dict[str, Any] | None:
        latest_question = self._latest_human_message_text(messages)
        if not latest_question:
            return None

        explicit_names, _ = self._resolve_paper_names([latest_question], available_names)
        recent_names = self._find_recent_paper_names(messages[:-1], available_names)

        if self._looks_like_list_request(latest_question):
            return self._make_tool_call("list_indexed_papers", {})

        if self._looks_like_digest_request(latest_question):
            target_name = (
                explicit_names[0]
                if explicit_names
                else (recent_names[0] if recent_names else None)
            )
            if target_name is None and len(default_scope_names) == 1:
                target_name = default_scope_names[0]
            if target_name is None and len(available_names) == 1:
                target_name = available_names[0]
            if target_name is not None:
                return self._make_tool_call(
                    "generate_paper_digest_tool",
                    {"paper_name": target_name},
                )
            return self._make_tool_call("list_indexed_papers", {})

        retrieval_names = explicit_names or recent_names
        retrieval_query = self._augment_query_with_context(latest_question, retrieval_names)
        args: dict[str, Any] = {"query": retrieval_query}
        if retrieval_names:
            args["paper_names"] = retrieval_names
        return self._make_tool_call("retrieve_paper_evidence", args)

    def _latest_human_message_text(self, messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return self._message_to_text(message)
        return ""

    def _find_recent_paper_names(
        self,
        messages: Sequence[BaseMessage],
        available_names: Sequence[str],
        limit: int = 2,
    ) -> list[str]:
        found: list[str] = []
        for message in reversed(messages):
            text = self._message_to_text(message)
            for available_name in available_names:
                if available_name in text and available_name not in found:
                    found.append(available_name)
                    if len(found) >= limit:
                        return found
        return found

    def _looks_like_list_request(self, text: str) -> bool:
        keywords = [
            "有哪些论文",
            "当前有哪些论文",
            "论文列表",
            "列出论文",
            "当前索引",
            "看一下论文",
        ]
        return any(keyword in text for keyword in keywords)

    def _looks_like_digest_request(self, text: str) -> bool:
        keywords = [
            "摘要卡片",
            "阅读卡片",
            "结构化摘要",
            "生成摘要",
            "生成卡片",
            "总结这篇",
        ]
        return any(keyword in text for keyword in keywords)

    def _augment_query_with_context(
        self,
        question: str,
        paper_names: Sequence[str],
    ) -> str:
        if not paper_names:
            return question
        joined_names = "、".join(paper_names)
        return f"{question}\n重点关注论文：{joined_names}"

    def _make_tool_call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"call_{uuid.uuid4().hex}",
            "name": tool_name,
            "args": args,
            "type": "tool_call",
        }

    def _scope_names_from_ids(
        self,
        paper_name_to_id: dict[str, str],
        paper_ids: set[str] | None,
    ) -> list[str]:
        return [
            paper_name
            for paper_name, paper_id in paper_name_to_id.items()
            if paper_ids is None or paper_id in paper_ids
        ]
