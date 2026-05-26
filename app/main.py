import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.bot.gateway import router as bot_router
from app.database import init_db
from app.plugins.base import PluginRegistry
from app.plugins.system.handler import SystemPlugin
from app.plugins.study.handler import StudyPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    await init_db()

    # Load plugins
    registry = PluginRegistry.get()
    plugins = [
        SystemPlugin(),
        StudyPlugin(),  # RAG-powered study agent
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
    return {"status": "ok", "docs_indexed": await _index_status()}


async def _index_status() -> int:
    """Return number of indexed documents."""
    try:
        from app.database import get_db
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
    except Exception:
        return 0
