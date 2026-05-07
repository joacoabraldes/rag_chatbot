# -*- coding: utf-8 -*-
"""Intent classifier — decides whether a query needs RAG retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import CLASSIFIER_PROMPT

if TYPE_CHECKING:
    from core.observability import RequestTrace

_client = AsyncOpenAI(timeout=10.0)


async def classify_intent(
    query: str,
    model: str | None = None,
    trace: Optional["RequestTrace"] = None,
) -> bool:
    """Return True if the query requires RAG retrieval, False otherwise."""
    model_name = model or OPENAI_MODEL_FAST
    resp = await _client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": query},
        ],
        max_completion_tokens=20,
    )
    if trace is not None and resp.usage is not None:
        trace.add_usage(model_name, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    raw = resp.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        return bool(result.get("retrieve", True))
    except (json.JSONDecodeError, AttributeError):
        return True
