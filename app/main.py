import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.bot.gateway import router as bot_router
from app.config import settings
from app.database import init_db
from app.plugins.base import PluginRegistry
from app.plugins.brain.handler import BrainPlugin
from app.plugins.study.handler import StudyPlugin
from app.plugins.system.handler import SystemPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    await init_db()

    # Verify Ollama + required models on startup
    from app.rag.guard import verify_ollama_on_startup
    health = await verify_ollama_on_startup()
    if not health["ok"]:
        logger.error(
            "Ollama startup check FAILED: %s (models: %s)",
            health.get("error"), health.get("models"),
        )
        logger.warning(
            "Bot will start but RAG features will be unavailable. "
            "Run: ollama serve && ollama pull qwen2.5:7b && ollama pull nomic-embed-text"
        )
    else:
        logger.info("Ollama healthy: models=%s", health["models"])

    # Load plugins — BrainPlugin MUST be registered before StudyPlugin
    # because Dispatcher routes free text to BrainPlugin first
    registry = PluginRegistry.get()
    plugins = [
        SystemPlugin(),
        BrainPlugin(),   # Agent brain — intercepts free text, routes to tools
        StudyPlugin(),   # RAG-powered study backend (called by BrainPlugin tools)
    ]
    for plugin in plugins:
        registry.register(plugin)
        await plugin.on_load()
        logger.info("Plugin loaded: %s", plugin.name)

    yield

    # Shutdown
    for p in registry.all():
        await p.on_unload()
    logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan, title="magic-grimoire")
app.include_router(bot_router)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.ollama_llm_model, "agent": settings.llm_provider}
