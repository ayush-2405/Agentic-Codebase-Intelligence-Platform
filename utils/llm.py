"""
utils/llm.py — Centralized OpenAI LLM interface.

All agent calls route through here so we get a single place for:
  - retry logic
  - token counting / cost tracking
  - model switching
  - structured output helpers
"""

from __future__ import annotations

import time
import logging
from typing import Any

import openai

import config

logger = logging.getLogger(__name__)

# Initialise client once
_client: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    """Return (or lazily create) the shared OpenAI client."""
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to a .env file or export it."
            )
        _client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str:
    """
    Send a chat completion request and return the assistant message text.

    Args:
        messages:        List of {"role": ..., "content": ...} dicts.
        model:           Override the default model.
        temperature:     Sampling temperature (0 = deterministic).
        max_tokens:      Maximum tokens in the response.
        response_format: Optional dict e.g. {"type": "json_object"}.

    Returns:
        The assistant's reply as a plain string.
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
    }
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        logger.debug(
            "LLM call | model=%s | tokens_in=%d | tokens_out=%d | latency=%.2fs",
            kwargs["model"],
            usage.prompt_tokens if usage else -1,
            usage.completion_tokens if usage else -1,
            elapsed,
        )
        return content
    except openai.OpenAIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise


def chat_completion_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Stream a chat completion and yield content deltas as they arrive.
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        "stream": True,
    }

    t0 = time.perf_counter()
    try:
        with client.chat.completions.create(**kwargs) as stream:
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content or ""
                if delta:
                    yield delta
        logger.debug("LLM streaming call | model=%s | latency=%.2fs", kwargs["model"], time.perf_counter() - t0)
    except openai.OpenAIError as exc:
        logger.error("OpenAI API streaming error: %s", exc)
        raise


def system_user(system: str, user: str, **kwargs) -> str:
    """Convenience wrapper for the common system+user message pattern."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return chat_completion(messages, **kwargs)


def stream_system_user(system: str, user: str, **kwargs):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return chat_completion_stream(messages, **kwargs)
