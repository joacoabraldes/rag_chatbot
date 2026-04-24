# -*- coding: utf-8 -*-
"""Suggested follow-up questions — generated after each response.

Once the main answer has streamed, a fast LLM proposes 3 short follow-ups
grounded in the conversation so far. These become clickable chips in the
UI that re-send as new queries.
"""

from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import FOLLOWUP_PROMPT

_client = AsyncOpenAI(timeout=10.0)


async def generate_followups(
    query: str,
    answer: str,
    model: str | None = None,
) -> List[str]:
    """Return up to 3 short follow-up questions in Spanish.

    Fails silently — an empty list is a valid response and the UI hides
    the section when nothing comes back.
    """
    if not answer.strip():
        return []

    # Trim to keep the prompt small — tail of the answer is usually the richest.
    trimmed_answer = answer.strip()
    if len(trimmed_answer) > 1200:
        trimmed_answer = trimmed_answer[-1200:]

    user_content = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Respuesta dada:\n{trimmed_answer}\n\n"
        "Generá 3 preguntas de seguimiento relevantes."
    )

    try:
        resp = await _client.chat.completions.create(
            model=model or OPENAI_MODEL_FAST,
            messages=[
                {"role": "system", "content": FOLLOWUP_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip common markdown code fences if present.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out: List[str] = []
        for item in data[:3]:
            if isinstance(item, str):
                cleaned = item.strip().strip('"').strip()
                if cleaned and len(cleaned) <= 140:
                    out.append(cleaned)
        return out
    except Exception:
        return []
