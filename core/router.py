# -*- coding: utf-8 -*-
"""Intent classifier — decides whether a query needs RAG retrieval."""

import json

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import CLASSIFIER_PROMPT

_client = AsyncOpenAI(timeout=10.0)


async def classify_intent(query: str, model: str | None = None) -> bool:
    """Return True if the query requires RAG retrieval, False otherwise."""
    resp = await _client.chat.completions.create(
        model=model or OPENAI_MODEL_FAST,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": query},
        ],
        max_completion_tokens=20,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        return bool(result.get("retrieve", True))
    except (json.JSONDecodeError, AttributeError):
        return True
