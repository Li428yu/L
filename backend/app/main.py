from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.agent import PaperAgentService
from backend.app.config import settings
from backend.app.eval_baselines import (
    baseline_metadata,
    resolve_eval_document_ids,
    resolve_eval_suite_path as resolve_baseline_suite_path,
)
from backend.app.evaluation import load_eval_suite, run_eval_suite
from backend.app.indexer import DocumentIndexer
from backend.app.llm_clients import ModelClients
from backend.app.memory import MemoryManager
from backend.app.models import (
    AskRequest,
    ChunkPreview,
    ConversationDetail,
    ConversationInfo,
    ConversationMessage,
    DocumentInfo,
    DocumentImageInfo,
    EvidenceItem,
    EvaluationRun,
    EvaluationRunRequest,
    ModelCatalog,
    ModelPreset,
    TaskInfo,
    UploadResponse,
)
from backend.app.observability import ObservabilityClient
from backend.app.storage import MetadataStore, new_id
from backend.app.vector_store import ChromaPaperStore


load_dotenv()
settings.ensure_dirs()

store = MetadataStore(settings.sqlite_path)
model_clients = ModelClients(settings)
vector_store = ChromaPaperStore(settings.chroma_dir)
memory = MemoryManager(store)
observer = ObservabilityClient(settings)
indexer = DocumentIndexer(
    settings=settings,
    store=store,
    vector_store=vector_store,
    model_clients=model_clients,
)
agent = PaperAgentService(
    settings=settings,
    store=store,
    vector_store=vector_store,
    model_clients=model_clients,
    memory=memory,
    observer=observer,
)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "app": settings.app_name,
        "vector_store": vector_store.name,
        "vector_records": vector_store.count(),
    }


@app.get("/api/models", response_model=ModelCatalog)
def models() -> ModelCatalog:
    presets = settings.public_model_presets()
    return ModelCatalog(
        presets=[
            ModelPreset(
                id=item.id,
                label=item.label,
                description=item.description,
                chat_model=item.chat_model,
                embedding_model=item.embedding_model,
                top_k=item.top_k,
            )
            for item in presets
        ],
        default_preset=settings.resolve_model_preset(settings.default_model_preset).id,
        chat_model_options=settings.chat_model_options,
        embedding_model_options=settings.embedding_model_options,
        default_chat_model=settings.default_chat_model,
        default_embedding_model=settings.default_embedding_model,
        default_top_k=settings.default_top_k,
    )


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="请上传 PDF 或 DOCX 文档。")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(file_bytes) > 80 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件超过 80MB，请先压缩或拆分后上传。")

    file_hash, path, source_path = indexer.save_upload(
        file_name=file.filename or "paper.pdf",
        file_bytes=file_bytes,
    )
    existing = store.get_document_by_hash(file_hash)
    document_id = existing.id if existing else new_id("doc")
    document = store.upsert_document(
        document_id=document_id,
        file_name=file.filename or path.name,
        file_hash=file_hash,
        source_path=source_path,
        status="queued",
    )
    task = store.create_task(document_id=document.id, message="文件已上传，等待后台索引。")
    background_tasks.add_task(indexer.index_document, task_id=task.id, document_id=document.id)
    return UploadResponse(document=document, task=task)


@app.get("/api/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return task


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return store.list_documents()


@app.get("/api/conversations", response_model=list[ConversationInfo])
def list_conversations() -> list[ConversationInfo]:
    return [
        ConversationInfo(
            id=item.id,
            title=item.title,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in store.list_conversations(limit=30)
    ]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation = store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在。")

    messages: list[ConversationMessage] = []
    for row in store.get_messages(conversation_id, limit=100):
        evidence_payload = []
        if row.get("evidence_json"):
            try:
                evidence_payload = json.loads(row["evidence_json"])
            except json.JSONDecodeError:
                evidence_payload = []
        evidence_items = [EvidenceItem.model_validate(item) for item in evidence_payload]
        messages.append(
            ConversationMessage(
                id=int(row["id"]),
                conversation_id=str(row["conversation_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                evidence=agent.attach_related_images(evidence_items),
                created_at=str(row["created_at"]),
            )
        )

    return ConversationDetail(
        conversation=ConversationInfo(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        ),
        messages=messages,
        memory_used=memory.build_memory_context(conversation_id),
    )


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, str]:
    conversation = store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在。")
    store.delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    vector_store.delete_document(document_id)
    store.delete_document(document_id)
    _delete_uploaded_file_if_safe(document.source_path)
    return {"status": "deleted"}


def _delete_uploaded_file_if_safe(source_path: str) -> None:
    upload_root = settings.uploads_dir.resolve()
    path = Path(source_path).resolve()
    if not path.is_relative_to(upload_root):
        return
    with suppress(FileNotFoundError):
        path.unlink()


@app.post("/api/documents/{document_id}/reindex", response_model=TaskInfo)
def reindex_document(document_id: str, background_tasks: BackgroundTasks) -> TaskInfo:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    document = store.update_document(
        document_id=document.id,
        file_name=document.file_name,
        file_hash=document.file_hash,
        source_path=document.source_path,
        status="queued",
    )
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    task = store.create_task(document_id=document.id, message="重新索引任务已创建。")
    background_tasks.add_task(indexer.index_document, task_id=task.id, document_id=document.id)
    return task


@app.get("/api/documents/{document_id}/chunks", response_model=list[ChunkPreview])
def preview_chunks(document_id: str) -> list[ChunkPreview]:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    rows = vector_store.get_document_chunks(document_id, limit=80)
    return [
        ChunkPreview(
            chunk_id=str(row["id"]),
            page=int(row["metadata"].get("page", 0) or 0),
            page_start=int(row["metadata"].get("page_start", row["metadata"].get("page", 0)) or 0),
            page_end=int(row["metadata"].get("page_end", row["metadata"].get("page", 0)) or 0),
            section=str(row["metadata"].get("section") or ""),
            chunk_type=str(row["metadata"].get("chunk_type") or "text"),
            token_count=int(row["metadata"].get("token_count", 0) or 0),
            text=str(row["text"]),
        )
        for row in rows
    ]


@app.get("/api/documents/{document_id}/images", response_model=list[DocumentImageInfo])
def list_document_images(document_id: str) -> list[DocumentImageInfo]:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document_not_found")
    return [DocumentImageInfo(**row) for row in store.list_document_images(document_id)]


@app.get("/api/documents/{document_id}/images/{image_id}/file")
def get_document_image_file(
    document_id: str,
    image_id: str,
    thumbnail: bool = False,
) -> FileResponse:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document_not_found")
    image = next(
        (row for row in store.list_document_images(document_id) if str(row["id"]) == image_id),
        None,
    )
    if not image:
        raise HTTPException(status_code=404, detail="image_not_found")

    selected_path = str(image.get("thumbnail_path") or "") if thumbnail else ""
    if not selected_path:
        selected_path = str(image.get("image_path") or "")
    path = Path(selected_path).resolve()
    image_root = settings.images_dir.resolve()
    if not path.exists() or not path.is_relative_to(image_root):
        raise HTTPException(status_code=404, detail="image_file_not_found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: str) -> FileResponse:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    path = Path(document.source_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="原文件不存在。")
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if path.suffix.lower() == ".docx"
        else "application/pdf"
    )
    disposition = "attachment" if path.suffix.lower() == ".docx" else "inline"
    return FileResponse(
        path,
        media_type=media_type,
        filename=document.file_name,
        content_disposition_type=disposition,
    )


@app.post("/api/chat/stream")
def chat_stream(request: AskRequest) -> StreamingResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空。")

    def event_stream():
        try:
            for event in agent.stream(request):
                yield _stream_event(event.type, event.payload)
        except RuntimeError as exc:
            yield _stream_event("error", str(exc))
        except Exception:
            yield _stream_event("error", "回答生成失败，请稍后重试。")

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _stream_event(event_type: str, payload: object) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False) + "\n"


@app.post("/api/evaluation/run", response_model=EvaluationRun, response_model_exclude_defaults=True)
def run_evaluation(request: EvaluationRunRequest | list[str] = Body(...)) -> EvaluationRun:
    if isinstance(request, list):
        request = EvaluationRunRequest(document_ids=request)
    suite_path, baseline = _resolve_eval_suite_path(request)
    cases = load_eval_suite(suite_path)
    if request.case_ids:
        wanted = set(request.case_ids)
        cases = [case for case in cases if case.id in wanted]
    if request.limit is not None:
        cases = cases[: max(request.limit, 0)]
    if not cases:
        raise HTTPException(status_code=404, detail="没有找到评测集。")
    document_ids = resolve_eval_document_ids(
        documents=store.list_documents(),
        cases=cases,
        requested_document_ids=request.document_ids,
        document_policy=baseline.document_policy if baseline else "expected_ready",
    )
    if not document_ids:
        raise HTTPException(status_code=400, detail="没有可用于评测的已就绪文档。")
    return run_eval_suite(
        suite_name=baseline.suite_name if baseline else (request.suite_name or suite_path.stem),
        cases=cases,
        agent=agent,
        document_ids=document_ids,
        observer=observer,
        model_preset=request.model_preset,
        chat_model=request.chat_model,
        embedding_model=request.embedding_model,
        top_k=request.top_k,
        experiment_metadata=baseline_metadata(baseline),
    )


def _resolve_eval_suite_path(request: EvaluationRunRequest):
    try:
        return resolve_baseline_suite_path(
            eval_dir=settings.project_root / "evals",
            suite_name=request.suite_name,
            suite_path=request.suite_path,
            baseline_id=request.baseline_id,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="评测集必须放在 evals 目录下。")
