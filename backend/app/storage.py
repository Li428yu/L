from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.app.models import ChunkStrategy, DocumentInfo, TaskInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class MetadataStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists documents (
                    id text primary key,
                    file_name text not null,
                    file_hash text not null unique,
                    status text not null,
                    page_count integer not null default 0,
                    chunk_count integer not null default 0,
                    source_path text not null,
                    embedding_model text,
                    chunk_strategy_json text,
                    error text,
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists tasks (
                    id text primary key,
                    document_id text,
                    stage text not null,
                    status text not null,
                    progress real not null,
                    message text not null,
                    error text,
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists conversations (
                    id text primary key,
                    title text not null,
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists messages (
                    id integer primary key autoincrement,
                    conversation_id text not null,
                    role text not null,
                    content text not null,
                    evidence_json text,
                    created_at text not null
                );

                create table if not exists memory_facts (
                    id text primary key,
                    conversation_id text not null,
                    key text not null,
                    value text not null,
                    scope text not null,
                    created_at text not null,
                    updated_at text not null,
                    unique(conversation_id, key)
                );
                """
            )

    def upsert_document(
        self,
        *,
        document_id: str,
        file_name: str,
        file_hash: str,
        source_path: str,
        status: str,
        page_count: int = 0,
        chunk_count: int = 0,
        embedding_model: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        error: str | None = None,
    ) -> DocumentInfo:
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "select created_at from documents where id = ?", (document_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                insert into documents (
                    id, file_name, file_hash, status, page_count, chunk_count,
                    source_path, embedding_model, chunk_strategy_json, error,
                    created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    file_name = excluded.file_name,
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    source_path = excluded.source_path,
                    embedding_model = excluded.embedding_model,
                    chunk_strategy_json = excluded.chunk_strategy_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    file_name,
                    file_hash,
                    status,
                    page_count,
                    chunk_count,
                    source_path,
                    embedding_model,
                    chunk_strategy.model_dump_json() if chunk_strategy else None,
                    error,
                    created_at,
                    now,
                ),
            )
        return self.get_document(document_id)  # type: ignore[return-value]

    def update_document(
        self,
        *,
        document_id: str,
        file_name: str,
        file_hash: str,
        source_path: str,
        status: str,
        page_count: int = 0,
        chunk_count: int = 0,
        embedding_model: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        error: str | None = None,
    ) -> DocumentInfo | None:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                update documents
                set file_name = ?,
                    file_hash = ?,
                    status = ?,
                    page_count = ?,
                    chunk_count = ?,
                    source_path = ?,
                    embedding_model = ?,
                    chunk_strategy_json = ?,
                    error = ?,
                    updated_at = ?
                where id = ?
                """,
                (
                    file_name,
                    file_hash,
                    status,
                    page_count,
                    chunk_count,
                    source_path,
                    embedding_model,
                    chunk_strategy.model_dump_json() if chunk_strategy else None,
                    error,
                    utc_now(),
                    document_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_document(document_id)

    def list_documents(self) -> list[DocumentInfo]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from documents order by updated_at desc"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> DocumentInfo | None:
        with self.connect() as conn:
            row = conn.execute(
                "select * from documents where id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> DocumentInfo | None:
        with self.connect() as conn:
            row = conn.execute(
                "select * from documents where file_hash = ?", (file_hash,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def delete_document(self, document_id: str) -> None:
        with self.connect() as conn:
            conn.execute("delete from documents where id = ?", (document_id,))

    def create_task(self, *, document_id: str | None, message: str) -> TaskInfo:
        task_id = new_id("task")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                insert into tasks (
                    id, document_id, stage, status, progress, message,
                    error, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, document_id, "queued", "queued", 0.0, message, None, now, now),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def update_task(
        self,
        task_id: str,
        *,
        stage: str,
        status: str,
        progress: float,
        message: str,
        error: str | None = None,
    ) -> TaskInfo:
        with self.connect() as conn:
            conn.execute(
                """
                update tasks
                set stage = ?, status = ?, progress = ?, message = ?, error = ?, updated_at = ?
                where id = ?
                """,
                (stage, status, progress, message, error, utc_now(), task_id),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self.connect() as conn:
            row = conn.execute("select * from tasks where id = ?", (task_id,)).fetchone()
        if not row:
            return None
        return TaskInfo(**dict(row))

    def ensure_conversation(self, conversation_id: str | None, title: str) -> ConversationInfoRow:
        now = utc_now()
        if conversation_id:
            with self.connect() as conn:
                row = conn.execute(
                    "select * from conversations where id = ?", (conversation_id,)
                ).fetchone()
                if row:
                    conn.execute(
                        "update conversations set updated_at = ? where id = ?",
                        (now, conversation_id),
                    )
                    return ConversationInfoRow(**dict(row))

        new_conversation_id = new_id("conv")
        clean_title = title.strip()[:40] or "新的论文对话"
        with self.connect() as conn:
            conn.execute(
                """
                insert into conversations (id, title, created_at, updated_at)
                values (?, ?, ?, ?)
                """,
                (new_conversation_id, clean_title, now, now),
            )
        return ConversationInfoRow(
            id=new_conversation_id,
            title=clean_title,
            created_at=now,
            updated_at=now,
        )

    def list_conversations(self, limit: int = 30) -> list[ConversationInfoRow]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select * from conversations
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [ConversationInfoRow(**dict(row)) for row in rows]

    def get_conversation(self, conversation_id: str) -> ConversationInfoRow | None:
        with self.connect() as conn:
            row = conn.execute(
                "select * from conversations where id = ?",
                (conversation_id,),
            ).fetchone()
        return ConversationInfoRow(**dict(row)) if row else None

    def delete_conversation(self, conversation_id: str) -> None:
        with self.connect() as conn:
            conn.execute("delete from messages where conversation_id = ?", (conversation_id,))
            conn.execute("delete from memory_facts where conversation_id = ?", (conversation_id,))
            conn.execute("delete from conversations where id = ?", (conversation_id,))

    def save_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        evidence: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into messages (conversation_id, role, content, evidence_json, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(evidence or [], ensure_ascii=False),
                    utc_now(),
                ),
            )

    def get_recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select role, content, evidence_json, created_at
                from messages
                where conversation_id = ?
                order by id desc
                limit ?
                """,
                (conversation_id, limit),
            ).fetchall()
        items = [dict(row) for row in rows]
        items.reverse()
        return items

    def get_messages(self, conversation_id: str, limit: int = 80) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select id, conversation_id, role, content, evidence_json, created_at
                from messages
                where conversation_id = ?
                order by id asc
                limit ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_memory_fact(
        self,
        *,
        conversation_id: str,
        key: str,
        value: str,
        scope: str = "long_term",
    ) -> None:
        now = utc_now()
        fact_id = new_id("mem")
        with self.connect() as conn:
            conn.execute(
                """
                insert into memory_facts (id, conversation_id, key, value, scope, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(conversation_id, key) do update set
                    value = excluded.value,
                    scope = excluded.scope,
                    updated_at = excluded.updated_at
                """,
                (fact_id, conversation_id, key, value, scope, now, now),
            )

    def list_memory_facts(self, conversation_id: str) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute(
                "select key, value from memory_facts where conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _row_to_document(self, row: sqlite3.Row) -> DocumentInfo:
        data = dict(row)
        strategy_json = data.pop("chunk_strategy_json", None)
        strategy = ChunkStrategy.model_validate_json(strategy_json) if strategy_json else None
        return DocumentInfo(**data, chunk_strategy=strategy)


class ConversationInfoRow:
    def __init__(self, id: str, title: str, created_at: str, updated_at: str) -> None:
        self.id = id
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
