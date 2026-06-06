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

    # ── Gemini / LLM Gateway (Agent Brain) ───────────────
    gemini_api_key: str = ""
    llm_provider: str = "gateway"       # "gemini" (direct API) or "gateway" (llm-gateway proxy)
    llm_model: str = "smart-router"     # Model name (gemini-2.5-flash-lite or smart-router)
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    llm_gateway_url: str = "http://localhost:4000"  # llm-gateway base URL
    llm_context_window: int = 2048  # Must match Modelfile num_ctx (2048 saves ~1.2GB KV cache)

    # ── Ollama (RAG Engine) ───────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "magic-grimoire:3b"  # was "qwen2.5:7b" — optimized Modelfile
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 300.0

    # ── RAG ───────────────────────────────────────────────
    docs_dir: str = "app/docs"
    chunk_size: int = 256
    chunk_overlap: int = 30
    retrieval_top_k: int = 20        # Over-retrieve for reranker (was 3)

    # ── RAG v2: Query Refinement + Reranking ──────────────
    use_query_refiner: bool = True       # Gemini query refinement before retrieval
    use_reranker: bool = True            # Cross-encoder reranking after retrieval
    use_bm25_fallback: bool = True       # BM25 keyword search if rerank fails
    reranker_top_k: int = 3             # Chunks after reranking (3 = less VRAM pressure)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_threshold: float = 0.0     # Min rerank score (logits; 0.0 = keep positive)
    reranker_device: str = "cuda"       # "cuda" or "cpu" (fallback on OOM)

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

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"gemini", "gateway"}
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of: {allowed}")
        return v


# Module-level singleton
settings = Settings()
