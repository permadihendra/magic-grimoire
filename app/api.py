"""REST API endpoints for OpenClaw integration.

Exposes Magic Grimoire's study tools as HTTP endpoints.
Telegram bot continues to work as before — this adds a second interface.

Base URL: /api
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["openclaw"])


# ── Request/Response models ───────────────────────────────

class AskRequest(BaseModel):
    query: str
    difficulty: str = "normal"
    document: str | None = None

class QuizRequest(BaseModel):
    topic: str
    count: int = 5
    difficulty: str = "normal"

class SummarizeRequest(BaseModel):
    topic: str

class QnaRequest(BaseModel):
    topic: str
    count: int = 10

class RetrieveRequest(BaseModel):
    query: str
    document: str | None = None

class FeedbackRequest(BaseModel):
    type: str = Field(description="'good', 'wrong_source', or 'reset'")
    detail: str = ""

class ToolResponse(BaseModel):
    result: str
    status: str = "ok"


# ── Health ─────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check for OpenClaw integration."""
    from app.config import settings
    return {
        "status": "ok",
        "service": "magic-grimoire",
        "model": settings.ollama_llm_model,
        "agent": settings.llm_provider,
    }


# ── Tool endpoints ─────────────────────────────────────────

@router.post("/tools/ask", response_model=ToolResponse)
async def api_ask(req: AskRequest):
    """Query study documents with RAG.

    Returns answer with source citations.
    """
    from app.plugins.brain.handler import _tool_ask
    try:
        result = await _tool_ask(
            query=req.query,
            difficulty=req.difficulty,
            document=req.document,
            chat_id=0,  # API calls are not chat-scoped
        )
        # _tool_ask returns (answer, follow_up) tuple
        if isinstance(result, tuple):
            answer = result[0] if result[0] else ""
        else:
            answer = str(result)

        if not answer:
            raise HTTPException(status_code=404, detail="No answer found")

        return ToolResponse(result=answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("API ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/quiz", response_model=ToolResponse)
async def api_quiz(req: QuizRequest):
    """Generate practice questions on a topic."""
    from app.plugins.brain.handler import _tool_quiz
    try:
        result = await _tool_quiz(
            topic=req.topic,
            count=str(req.count),
            difficulty=req.difficulty,
            chat_id=0,
        )
        return ToolResponse(result=str(result))
    except Exception as e:
        logger.error("API quiz failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/summarize", response_model=ToolResponse)
async def api_summarize(req: SummarizeRequest):
    """Create a topic summary from documents."""
    from app.plugins.brain.handler import _tool_summarize
    try:
        result = await _tool_summarize(
            topic=req.topic,
            chat_id=0,
        )
        return ToolResponse(result=str(result))
    except Exception as e:
        logger.error("API summarize failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/qna", response_model=ToolResponse)
async def api_qna(req: QnaRequest):
    """Generate Q&A pairs for study."""
    from app.plugins.brain.handler import _tool_qna
    try:
        result = await _tool_qna(
            topic=req.topic,
            count=str(req.count),
            chat_id=0,
        )
        return ToolResponse(result=str(result))
    except Exception as e:
        logger.error("API qna failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/list_docs", response_model=ToolResponse)
async def api_list_docs():
    """List all indexed documents."""
    from app.plugins.brain.handler import _tool_list_docs_v2
    try:
        result = await _tool_list_docs_v2()
        return ToolResponse(result=str(result))
    except Exception as e:
        logger.error("API list_docs failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/retrieve", response_model=ToolResponse)
async def api_retrieve(req: RetrieveRequest):
    """Retrieve raw passages without LLM generation."""
    from app.plugins.brain.handler import _tool_retrieve
    try:
        passages = await _tool_retrieve(
            query=req.query,
            document=req.document,
            chat_id=0,
        )
        if not passages:
            return ToolResponse(result="No passages found.")

        # Format passages for readability
        lines = []
        for i, p in enumerate(passages[:10], 1):
            lines.append(f"**{i}.** {p['filename']} (score: {p['score']:.3f})")
            lines.append(f"   {p['text'][:200]}...")
            lines.append("")

        return ToolResponse(result="\n".join(lines))
    except Exception as e:
        logger.error("API retrieve failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/feedback", response_model=ToolResponse)
async def api_feedback(req: FeedbackRequest):
    """Submit feedback about retrieval quality."""
    from app.plugins.brain.handler import _tool_feedback
    try:
        result = await _tool_feedback(
            type=req.type,
            detail=req.detail,
            chat_id=0,
        )
        return ToolResponse(result=str(result))
    except Exception as e:
        logger.error("API feedback failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
