"""Minimal Gemini API client — raw httpx, no SDK.

Uses the generateContent REST endpoint.
Free tier: gemini-2.5-flash-lite (1,500 req/day, no credit card).
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


async def gemini_chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    timeout: float = 30.0,
) -> str:
    """Send a chat request to Gemini and return the response text.

    Args:
        system_prompt: System-level instruction for the model.
        user_message: The user's input message.
        max_tokens: Maximum output tokens.
        timeout: HTTP request timeout in seconds.

    Returns:
        The generated response text.

    Raises:
        RuntimeError: If the API key is not configured or the API call fails.
    """
    api_key = settings.gemini_api_key
    if not api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env"
        )

    model = settings.llm_model
    if not model:
        model = "gemini-2.5-flash-lite"

    url = _API_URL.format(model=model)

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
            "temperature": 0.4,
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
        # Check if blocked by safety
        block_reason = (
            data.get("promptFeedback", {})
            .get("blockReason", "unknown")
        )
        raise RuntimeError(
            f"Gemini response blocked or empty (reason: {block_reason})"
        ) from e
