from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.config import Settings
from backend.app.document_processing import (
    choose_chunk_strategy,
    compute_file_hash,
    extract_document_text,
    safe_upload_name,
    split_pages_into_chunks,
)
from backend.app.image_processing import (
    enrich_images_with_vision,
    extract_docx_images,
    extract_pdf_images,
    image_records_to_chunks,
    ocr_status_counts,
    vision_status_counts,
)
from backend.app.llm_clients import ModelClients
from backend.app.models import ChunkStrategy, DocumentInfo
from backend.app.storage import MetadataStore
from backend.app.vector_store import ChromaPaperStore


@dataclass
class BatchEmbeddingResult:
    vectors: list[list[float]]
    provider: str
    used_fallback: bool


class DocumentIndexer:
    def __init__(
        self,
        *,
        settings: Settings,
        store: MetadataStore,
        vector_store: ChromaPaperStore,
        model_clients: ModelClients,
    ) -> None:
        self.settings = settings
        self.store = store
        self.vector_store = vector_store
        self.model_clients = model_clients

    def save_upload(self, *, file_name: str, file_bytes: bytes) -> tuple[str, Path, str]:
        file_hash = compute_file_hash(file_bytes)
        target_path = self.settings.uploads_dir / safe_upload_name(file_hash, file_name)
        target_path.write_bytes(file_bytes)
        return file_hash, target_path, str(target_path)

    def index_document(self, *, task_id: str, document_id: str) -> None:
        document = self.store.get_document(document_id)
        if document is None:
            self.store.update_task(
                task_id,
                stage="failed",
                status="failed",
                progress=1,
                message="文档记录不存在。",
                error="document_not_found",
            )
            return

        try:
            current_stage = "parse"
            if not self._update_existing_document(
                task_id=task_id,
                document=document,
                status="processing",
                embedding_model=self.settings.default_embedding_model,
            ):
                return
            self.store.update_task(
                task_id,
                stage="parse",
                status="running",
                progress=0.12,
                message="正在读取文档内容。",
            )
            source_path = Path(document.source_path)
            pages = extract_document_text(source_path)
            text_pages = [page for page in pages if page.text.strip()]
            image_chunks = []
            images = []
            ocr_counts: dict[str, int] = {}
            vision_counts: dict[str, int] = {}
            if source_path.suffix.lower() in {".pdf", ".docx"}:
                current_stage = "images"
                if self._stop_if_document_deleted(task_id=task_id, document_id=document.id):
                    return
                self.store.update_task(
                    task_id,
                    stage="images",
                    status="running",
                    progress=0.24,
                    message="正在抽取文档图片和图片文字。",
                )
                if source_path.suffix.lower() == ".pdf":
                    images = extract_pdf_images(
                        pdf_path=source_path,
                        output_dir=self.settings.images_dir / document.id,
                        document_id=document.id,
                    )
                else:
                    images = extract_docx_images(
                        docx_path=source_path,
                        output_dir=self.settings.images_dir / document.id,
                        document_id=document.id,
                    )
                ocr_counts = ocr_status_counts(images)
                if images:
                    self.store.update_task(
                        task_id,
                        stage="images",
                        status="running",
                        progress=0.26,
                        message=self._ocr_status_message(ocr_counts),
                    )
                if images and self.settings.enable_vision_analysis:
                    self.store.update_task(
                        task_id,
                        stage="vision",
                        status="running",
                        progress=0.28,
                        message="正在用视觉模型理解论文图片。",
                    )
                    images = enrich_images_with_vision(
                        images=images,
                        analyze_image=self._analyze_image_with_vision,
                        max_images=self.settings.max_vision_images,
                    )
                    vision_counts = vision_status_counts(images)
                    self.store.update_task(
                        task_id,
                        stage="vision",
                        status="running",
                        progress=0.32,
                        message=self._vision_status_message(vision_counts),
                    )
                self.store.replace_document_images(
                    document_id=document.id,
                    images=[image.to_storage_dict() for image in images],
                )
                image_chunks = image_records_to_chunks(
                    images=images,
                    document_id=document.id,
                    paper_name=document.file_name,
                    source=document.source_path,
                    file_hash=document.file_hash,
                )
            if not text_pages and not image_chunks:
                raise ValueError("没有读取到可用文字或图片证据。扫描版 PDF 需要 OCR，空白 DOCX 也无法索引。")

            current_stage = "chunk"
            if self._stop_if_document_deleted(task_id=task_id, document_id=document.id):
                return
            self.store.update_task(
                task_id,
                stage="chunk",
                status="running",
                progress=0.32,
                message="正在按文档结构自动整理段落。",
            )
            strategy = choose_chunk_strategy(text_pages or pages)
            chunks = split_pages_into_chunks(
                pages=text_pages,
                document_id=document.id,
                paper_name=document.file_name,
                file_hash=document.file_hash,
                source=document.source_path,
                strategy=strategy,
            )
            chunks.extend(image_chunks)
            if not chunks:
                raise ValueError("文档文字太少或结构异常，没有生成可检索的段落。")

            current_stage = "embedding"
            if self._stop_if_document_deleted(task_id=task_id, document_id=document.id):
                return
            self.store.update_task(
                task_id,
                stage="embedding",
                status="running",
                progress=0.58,
                message=f"正在为 {len(chunks)} 个段落建立检索索引。",
            )
            embedding_result = self._embed_in_batches(
                [chunk.text for chunk in chunks],
                model=self.settings.default_embedding_model,
            )
            embeddings = embedding_result.vectors

            current_stage = "vector_store"
            if self._stop_if_document_deleted(task_id=task_id, document_id=document.id):
                return
            self.store.update_task(
                task_id,
                stage="vector_store",
                status="running",
                progress=0.82,
                message="正在保存检索索引。",
            )
            self.vector_store.delete_document(document.id)
            self.vector_store.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embedding_result.provider,
            )
            if not self._update_existing_document(
                task_id=task_id,
                document=document,
                status="ready",
                page_count=len(pages),
                chunk_count=len(chunks),
                embedding_model=embedding_result.provider,
                chunk_strategy=strategy,
            ):
                self.vector_store.delete_document(document.id)
                return
            self.store.update_task(
                task_id,
                stage="completed",
                status="completed",
                progress=1,
                message=self._completed_message(
                    embedding_used_fallback=embedding_result.used_fallback,
                    ocr_counts=ocr_counts,
                    vision_counts=vision_counts,
                ),
            )
        except Exception as exc:
            if self.store.get_document(document.id) is None:
                self._finish_deleted_document_task(task_id=task_id, document_id=document.id)
                return
            friendly_error = self._friendly_error(exc, current_stage)
            if not self._update_existing_document(
                task_id=task_id,
                document=document,
                status="failed",
                error=friendly_error,
            ):
                return
            self.store.update_task(
                task_id,
                stage="failed",
                status="failed",
                progress=1,
                message="文档已保存，但暂时还不能提问。",
                error=friendly_error,
            )

    def _update_existing_document(
        self,
        *,
        task_id: str,
        document: DocumentInfo,
        status: str,
        page_count: int = 0,
        chunk_count: int = 0,
        embedding_model: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        error: str | None = None,
    ) -> bool:
        updated = self.store.update_document(
            document_id=document.id,
            file_name=document.file_name,
            file_hash=document.file_hash,
            source_path=document.source_path,
            status=status,
            page_count=page_count,
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            error=error,
        )
        if updated is not None:
            return True
        self._finish_deleted_document_task(task_id=task_id, document_id=document.id)
        return False

    def _embed_in_batches(
        self,
        texts: list[str],
        model: str,
        batch_size: int = 64,
    ) -> BatchEmbeddingResult:
        vectors: list[list[float]] = []
        provider = model
        used_fallback = False
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            result = self.model_clients.embed_documents_with_info(batch, model=model)
            vectors.extend(result.vectors)
            provider = result.provider
            used_fallback = used_fallback or result.used_fallback
        return BatchEmbeddingResult(
            vectors=vectors,
            provider=provider,
            used_fallback=used_fallback,
        )

    def _analyze_image_with_vision(self, image_path: Path, prompt: str) -> str:
        return self.model_clients.vision_text(
            image_path=image_path,
            prompt=prompt,
            model=self.settings.vision_model,
        )

    def _vision_status_message(self, counts: dict[str, int]) -> str:
        total = sum(counts.values())
        ready = counts.get("vision_ready", 0)
        failed = counts.get("vision_failed", 0)
        skipped = counts.get("vision_skipped", 0)
        return (
            f"视觉分析完成：共 {total} 张图片，成功 {ready} 张，"
            f"失败 {failed} 张，跳过 {skipped} 张。"
        )

    def _ocr_status_message(self, counts: dict[str, int]) -> str:
        total = sum(counts.values())
        ready = counts.get("ocr_ready", 0)
        empty = counts.get("ocr_empty", 0)
        failed = counts.get("ocr_failed", 0)
        skipped = counts.get("ocr_skipped", 0)
        return (
            f"OCR 完成：共 {total} 张图片，识别成功 {ready} 张，空结果 {empty} 张，"
            f"失败 {failed} 张，跳过 {skipped} 张。"
        )

    def _completed_message(
        self,
        *,
        embedding_used_fallback: bool,
        ocr_counts: dict[str, int],
        vision_counts: dict[str, int],
    ) -> str:
        parts = ["文档已准备好，可以开始提问。"]
        if ocr_counts:
            parts.append(self._ocr_status_message(ocr_counts))
        if vision_counts:
            parts.append(self._vision_status_message(vision_counts))
        if embedding_used_fallback:
            parts.append("模型服务暂时不可用，本次先使用本地备用检索。")
        return " ".join(parts)

    def _friendly_error(self, exc: Exception, stage: str) -> str:
        raw = str(exc).lower()
        if stage == "embedding" or "api" in raw or "model" in raw or "quota" in raw:
            return (
                "文档已上传并读取成功，但建立检索索引时模型服务暂时不可用。"
                "请稍后点击“重新准备”，或检查模型配置、额度和网络。"
            )
        if stage == "parse":
            return (
                "文档已上传，但没有读取到可用文字。"
                "如果是扫描版 PDF，需要先做 OCR；如果是 DOCX，请确认里面有正文。"
            )
        if stage == "vector_store":
            return "文档已读取成功，但保存检索索引时失败。请稍后重新准备。"
        return "文档已上传，但准备过程中遇到问题。请稍后重试，或换一个文件试试。"

    def _stop_if_document_deleted(self, *, task_id: str, document_id: str) -> bool:
        if self.store.get_document(document_id) is not None:
            return False
        self._finish_deleted_document_task(task_id=task_id, document_id=document_id)
        return True

    def _finish_deleted_document_task(self, *, task_id: str, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.store.update_task(
            task_id,
            stage="completed",
            status="completed",
            progress=1,
            message="文档已删除，已停止准备。",
        )
