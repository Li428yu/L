from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _csv_env(name: str, fallback: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or fallback


@dataclass(frozen=True)
class ResolvedModelPreset:
    id: str
    label: str
    description: str
    chat_model: str
    embedding_model: str
    top_k: int


@dataclass(frozen=True)
class Settings:
    app_name: str = "Paper Reading Assistant"
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    api_key: str | None = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base_url: str | None = os.getenv("API_BASE_URL")
    default_chat_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    default_embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    model_timeout_seconds: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "8"))
    model_max_retries: int = int(os.getenv("MODEL_MAX_RETRIES", "0"))

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir.is_absolute():
            return self.data_dir
        return self.project_root / self.data_dir

    @property
    def uploads_dir(self) -> Path:
        return self.resolved_data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.resolved_data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.resolved_data_dir / "assistant.sqlite3"

    @property
    def chat_model_options(self) -> list[str]:
        return _csv_env("CHAT_MODEL_OPTIONS", [self.default_chat_model, "gpt-4.1"])

    @property
    def embedding_model_options(self) -> list[str]:
        return _csv_env(
            "EMBEDDING_MODEL_OPTIONS",
            [self.default_embedding_model, "text-embedding-3-large"],
        )

    @property
    def default_model_preset(self) -> str:
        return os.getenv("DEFAULT_MODEL_PRESET", "balanced")

    def public_model_presets(self) -> list[ResolvedModelPreset]:
        return [
            self.resolve_model_preset("balanced"),
            self.resolve_model_preset("careful"),
            self.resolve_model_preset("quick"),
        ]

    def resolve_model_preset(self, preset_id: str | None) -> ResolvedModelPreset:
        resolved_id = preset_id or self.default_model_preset
        presets = {
            "balanced": ResolvedModelPreset(
                id="balanced",
                label="日常阅读",
                description="适合大多数论文问答，速度和回答质量比较均衡。",
                chat_model=os.getenv("BALANCED_CHAT_MODEL", self.default_chat_model),
                embedding_model=os.getenv(
                    "BALANCED_EMBEDDING_MODEL",
                    self.default_embedding_model,
                ),
                top_k=int(os.getenv("BALANCED_TOP_K", str(self.default_top_k))),
            ),
            "careful": ResolvedModelPreset(
                id="careful",
                label="精读模式",
                description="适合方法细节、实验结论和证据核对，会多取一些原文证据。",
                chat_model=os.getenv("CAREFUL_CHAT_MODEL", self.default_chat_model),
                embedding_model=os.getenv(
                    "CAREFUL_EMBEDDING_MODEL",
                    self.default_embedding_model,
                ),
                top_k=int(os.getenv("CAREFUL_TOP_K", "7")),
            ),
            "quick": ResolvedModelPreset(
                id="quick",
                label="快速浏览",
                description="适合先看大意和快速总结，取证据更少、响应更轻。",
                chat_model=os.getenv("QUICK_CHAT_MODEL", self.default_chat_model),
                embedding_model=os.getenv("QUICK_EMBEDDING_MODEL", self.default_embedding_model),
                top_k=int(os.getenv("QUICK_TOP_K", "4")),
            ),
        }
        return presets.get(resolved_id, presets["balanced"])

    def ensure_dirs(self) -> None:
        self.resolved_data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
