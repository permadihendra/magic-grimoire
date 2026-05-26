from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Telegram ──────────────────────────────────────────
    telegram_token: str = ""
    telegram_webhook_url: str = ""
    telegram_webhook_secret: str = ""
    allowed_chat_ids: list[int] = []

    # ── Ollama ────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen3:4b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 60.0

    # ── RAG ───────────────────────────────────────────────
    docs_dir: str = "app/docs"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # ── Database ──────────────────────────────────────────
    db_path: str = "data/magic-grimoire.db"

    # ── AI Personality ────────────────────────────────────
    ai_personality: str = "friendly, helpful, knowledgeable tutor"

    @field_validator("allowed_chat_ids", mode="before")
    @classmethod
    def coerce_chat_ids(cls, v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v


# Module-level singleton
settings = Settings()


# Module-level singleton
settings = Settings()
