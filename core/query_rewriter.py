# -*- coding: utf-8 -*-
"""Query rewriting — rewrite follow-up questions as standalone queries.

Embedding models cannot resolve anaphora ("¿y el mes anterior?",
"¿cómo cambió?"). We use a fast LLM to rewrite the current query using
the conversation history so the retrieval step sees a self-contained
question. This dramatically improves recall on multi-turn chats.
"""

from __future__ import annotations

from typing import List, Optional

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import QUERY_REWRITE_PROMPT

_client = AsyncOpenAI(timeout=10.0)


def _format_history(history: List[dict], max_turns: int = 4) -> str:
    """Render the last N turns as a compact transcript for the prompt."""
    if not history:
        return ""
    # Keep only user/assistant pairs, trim to max_turns * 2 messages.
    filtered = [
        m for m in history if m.get("role") in ("user", "assistant")
    ][-max_turns * 2 :]
    lines = []
    for m in filtered:
        role = "Usuario" if m["role"] == "user" else "Asistente"
        content = (m.get("content") or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def rewrite_standalone(
    query: str,
    history: Optional[List[dict]],
    model: str | None = None,
) -> str:
    """Rewrite ``query`` as a standalone question using the conversation history.

    Returns the original query when there is no history or when the model
    fails — the caller can safely use the returned string for retrieval.
    """
    if not history:
        return query

    history_block = _format_history(history)
    if not history_block:
        return query

    user_content = (
        f"Historial de la conversación:\n{history_block}\n\n"
        f"Pregunta actual:\n{query}\n\n"
        "Devolvé SOLO la pregunta reformulada, sin prefijos ni comillas."
    )

    try:
        resp = await _client.chat.completions.create(
            model=model or OPENAI_MODEL_FAST,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=120,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Strip surrounding quotes if the model ignored instructions.
        if rewritten.startswith(('"', "'")) and rewritten.endswith(('"', "'")):
            rewritten = rewritten[1:-1].strip()
        return rewritten or query
    except Exception:
        return query
