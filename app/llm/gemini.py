"""LLM client — supports Gemini API direct and llm-gateway (OpenAI-compatible).

Routes based on settings.llm_provider:
  "gemini"  → Gemini REST API (generativelanguage.googleapis.com)
  "gateway" → llm-gateway proxy (localhost:4000, OpenAI-compatible)

All callers use gemini_chat() — routing is transparent.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


async def gemini_chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> str:
    """Send a chat request to the LLM and return the response text.

    Routes to Gemini API or llm-gateway based on settings.llm_provider.

    Args:
        system_prompt: System-level instruction for the model.
        user_message: The user's input message.
        max_tokens: Maximum output tokens.
        timeout: HTTP request timeout in seconds.

    Returns:
        The generated response text.

    Raises:
        RuntimeError: If the API call fails.
    """
    provider = settings.llm_provider

    if provider == "gateway":
        # Gemma v4 uses reasoning tokens — need higher max_tokens to get content
        # Reasoning eats ~450 tokens, so callers' 50-150 values aren't enough
        effective_tokens = max(max_tokens, 1000)
        return await _gateway_chat(system_prompt, user_message, effective_tokens, timeout)
    else:
        return await _gemini_chat(system_prompt, user_message, max_tokens, timeout)


async def _gateway_chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call llm-gateway (OpenAI-compatible API)."""
    url = f"{settings.llm_gateway_url}/v1/chat/completions"
    model = settings.llm_model or "smart-router"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    logger.debug("Gateway request: %s model=%s", url, model)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                url,
                headers={"Authorization": "Bearer not-needed", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Gateway error: %s %s", e.response.status_code, e.response.text[:200])
            raise RuntimeError(
                f"Gateway returned {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Gateway timed out after {timeout}s") from e

    try:
        choice = data["choices"][0]
        message = choice["message"]

        # OpenAI format: content is the main response
        content = message.get("content")
        if content:
            return content.strip()

        # Gemma v4 may put response in reasoning_content if content is null
        reasoning = message.get("reasoning_content")
        if reasoning:
            logger.warning("Gateway: content=null, using reasoning_content")
            return reasoning.strip()

        raise RuntimeError("Gateway returned empty response (content=null)")

    except (KeyError, IndexError) as e:
        logger.error("Unexpected gateway response: %s", data)
        raise RuntimeError(f"Gateway response parse error: {e}") from e


async def _gemini_chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call Gemini REST API directly."""
    api_key = settings.gemini_api_key
    if not api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env"
        )

    model = settings.llm_model
    if not model:
        model = "gemini-2.5-flash-lite"

    url = _GEMINI_API_URL.format(model=model)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.0,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                url,
                params={"key": api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Gemini API error: %s %s", e.response.status_code, e.response.text)
            raise RuntimeError(
                f"Gemini API returned {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Gemini API timed out after {timeout}s") from e

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Gemini response: %s", data)
        block_reason = (
            data.get("promptFeedback", {})
            .get("blockReason", "unknown")
        )
        raise RuntimeError(
            f"Gemini response blocked or empty (reason: {block_reason})"
        ) from e
