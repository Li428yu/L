from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from assistant_core.types import Chunk, NodeExecutionLog, RetrievedChunk, ToolTrace


class AgentTraceMixin:
    def _get_thread_messages(self, agent, config: dict[str, Any]) -> list[BaseMessage]:
        try:
            snapshot = agent.get_state(config)
        except Exception:
            return []

        if snapshot is None:
            return []

        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", [])
        return list(messages)

    def _extract_final_answer(
        self,
        new_messages: Sequence[BaseMessage],
        all_messages: Sequence[BaseMessage],
    ) -> str:
        for message in reversed(new_messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                text = self._message_to_text(message)
                if text:
                    return text

        for message in reversed(all_messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                text = self._message_to_text(message)
                if text:
                    return text

        return "本轮没有生成可用回答。"

    def _extract_latest_evidence(self, messages: Sequence[BaseMessage]) -> list[RetrievedChunk]:
        latest_payload: dict[str, Any] | None = None

        for message in messages:
            if isinstance(message, ToolMessage):
                payload = self._parse_tool_payload(message)
                if payload.get("type") == "retrieval_result":
                    latest_payload = payload

        if latest_payload is None:
            return []

        evidence: list[RetrievedChunk] = []
        for item in latest_payload.get("evidence", []):
            try:
                evidence.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            chunk_id=int(item.get("chunk_id", -1)),
                            paper_id=str(item.get("paper_id", "")),
                            paper_name=str(item.get("paper_name", "")),
                            page=int(item.get("page", 0)),
                            text=str(item.get("text", "")),
                        ),
                        score=float(item.get("score", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                continue

        return evidence

    def _extract_tool_traces(self, messages: Sequence[BaseMessage]) -> list[ToolTrace]:
        pending_calls: dict[str, dict[str, Any]] = {}
        traces: list[ToolTrace] = []

        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    pending_calls[str(tool_call.get("id", ""))] = {
                        "name": str(tool_call.get("name", "tool")),
                        "args": tool_call.get("args", {}),
                    }
                continue

            if isinstance(message, ToolMessage):
                call_meta = pending_calls.get(
                    message.tool_call_id,
                    {"name": message.name or "tool", "args": {}},
                )
                traces.append(
                    ToolTrace(
                        tool_name=str(message.name or call_meta["name"]),
                        tool_args=self._normalize_tool_args(call_meta.get("args", {})),
                        tool_output=self._tool_message_to_text(message),
                    )
                )

        return traces

    def _extract_runtime_logs_from_event(
        self,
        event: dict[str, Any],
        start_index: int,
    ) -> list[NodeExecutionLog]:
        logs: list[NodeExecutionLog] = []

        for offset, (node_name, payload) in enumerate(event.items()):
            logs.append(
                NodeExecutionLog(
                    step_index=start_index + offset,
                    node_name=node_name,
                    summary=self._summarize_node_payload(node_name, payload),
                )
            )

        return logs

    def _summarize_node_payload(self, node_name: str, payload: Any) -> str:
        if not isinstance(payload, dict):
            return "Node updated state."

        messages = payload.get("messages", [])
        if not messages:
            return "Node updated state."

        latest_message = messages[-1]
        if node_name == "planner":
            return self._summarize_planner_message(latest_message)
        if node_name == "tools":
            return self._summarize_tool_messages(messages)
        if node_name == "assistant":
            return self._summarize_assistant_message(latest_message)

        return self._summarize_generic_message(latest_message)

    def _summarize_planner_message(self, message: BaseMessage) -> str:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_names = [str(tool_call.get("name", "tool")) for tool_call in message.tool_calls]
            return f"Planner routed this turn to tool node: {', '.join(tool_names)}."
        return "Planner sent the turn directly to the assistant node."

    def _summarize_tool_messages(self, messages: Sequence[BaseMessage]) -> str:
        tool_names: list[str] = []
        tool_error: str | None = None
        for message in messages:
            if isinstance(message, ToolMessage):
                tool_names.append(str(message.name or "tool"))
                payload = self._parse_tool_payload(message)
                if payload.get("type") == "tool_error":
                    tool_error = str(payload.get("error", "tool error"))

        if not tool_names:
            return "Tool node executed, but no tool result message was captured."

        if tool_error:
            return f"Tool node executed but hit a temporary error: {tool_error}"

        return f"Tool node executed: {', '.join(tool_names)}."

    def _summarize_assistant_message(self, message: BaseMessage) -> str:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_names = [str(tool_call.get("name", "tool")) for tool_call in message.tool_calls]
            return f"Assistant requested another tool call: {', '.join(tool_names)}."

        text = self._message_to_text(message)
        if not text:
            return "Assistant produced an empty response."

        preview = text.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = f"{preview[:120]}..."
        return f"Assistant produced the final answer: {preview}"

    def _summarize_generic_message(self, message: BaseMessage) -> str:
        text = self._message_to_text(message)
        if not text:
            return "Node updated messages."
        preview = text.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = f"{preview[:120]}..."
        return preview

    def _normalize_tool_args(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"value": value}

    def _parse_tool_payload(self, message: ToolMessage) -> dict[str, Any]:
        text = self._tool_message_to_text(message)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _tool_message_to_text(self, message: ToolMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(part.strip() for part in parts if part).strip()
        return str(content).strip()

    def _message_to_text(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(message, AIMessage):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "\n".join(part.strip() for part in text_parts if part).strip()
        return str(content).strip()

    def _to_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)
