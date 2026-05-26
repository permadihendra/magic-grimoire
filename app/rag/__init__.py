"""Magic Grimoire — RAG Study Engine.

Document ingestion, indexing, and querying with LlamaIndex + Ollama.
"""

from app.rag.models import get_llm, get_embed_model, configure_settings
from app.rag.engine import RAGEngine
from app.rag.indexer import DocumentIndexer

__all__ = ["get_llm", "get_embed_model", "configure_settings", "RAGEngine", "DocumentIndexer"]
