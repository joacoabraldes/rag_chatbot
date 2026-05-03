# -*- coding: utf-8 -*-
"""Suggested follow-up questions — generated after each response.

Once the main answer has streamed, a fast LLM proposes 3 short follow-ups
grounded in the conversation so far. These become clickable chips in the
UI that re-send as new queries — so they MUST be phrased as the user
asking the assistant, never the assistant offering to do something.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import List

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import FOLLOWUP_PROMPT

_client = AsyncOpenAI(timeout=10.0)


# Patterns the assistant uses when offering help or asking for clarification.
# When a "follow-up" matches any of these it was generated from the wrong
# perspective (assistant → user) instead of user → assistant, and we drop it.
#
# Three families:
# 1. "¿Querés/Quieres ...?" — assistant offering (anything after the verb).
#    Legitimate user-to-assistant questions don't start this way; the user
#    wouldn't ask the bot what *the bot* wants.
# 2. "¿Te X?" / "¿Necesitás X?" / "¿Preferís X?" — assistant offering or
#    asking after the user's preference.
# 3. Clarification requests: "¿Cuál es tu duda/pregunta/interés...?" —
#    assistant prompting the user for input.
_ASSISTANT_OFFER_PATTERNS = [
    # Family 1: opening verb means assistant is the speaker.
    r"\bquer[eé]s\b",
    r"\bquieres\b",
    r"\bdese[aá]s\b",
    r"\bdeseas\b",
    r"\bpref[eé]r[ií]s\b",
    r"\bprefieres\b",
    r"\bnecesit[aá]s\b",
    r"\bnecesitas\b",
    # Family 2: "Te + verb/noun" addressed to the user.
    r"\bte\s+(interesa|interesar[ií]a|gustar[ií]a|sirve|conviene|cuento|cuent|"
    r"muestro|paso|explico|armo|preparo|enseño|envio|env[ií]o|aviso|resumo|"
    r"ayudo|comparto|listo|busco|traigo|describo)",
    r"\ble\s+(interesa|gustar[ií]a|sirve|conviene)",
    # Family 3: assistant asking the user about themselves.
    r"\b(cu[aá]l|qu[eé]|sobre\s+qu[eé])\s+es\s+tu\s+(duda|pregunta|inter[eé]s|"
    r"objetivo|caso|consulta|necesidad|preferencia|inquietud|preocupaci[oó]n|tema|punto)",
    r"\btu\s+(duda|pregunta|inter[eé]s|objetivo|consulta|preferencia|inquietud)\b",
    # Family 4: cohortative ("¿podemos profundizar...?") = assistant offering.
    r"\bpod[eé]mos\s+(profundizar|ver|revisar|explorar|seguir|continuar)",
    r"\bpodemos\s+(profundizar|ver|revisar|explorar|seguir|continuar)",
    r"\bvamos\s+a\s+(profundizar|ver|revisar|explorar)",
    # Family 5: "¿Querés/Te gustaría...?" without the verb prefix.
    r"\bgustar[ií]a\s+(que|saber|conocer|profundizar)",
    r"\bservir[ií]a\s+que",
]

_ASSISTANT_OFFER_RE = re.compile(
    "|".join(_ASSISTANT_OFFER_PATTERNS),
    flags=re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _is_assistant_perspective(text: str) -> bool:
    """True when the question is phrased as the assistant offering help.

    We strip accents AND check the original — the regex above is
    accent-tolerant via character classes, but stripping covers the
    long tail of variants we didn't enumerate.
    """
    if not text:
        return False
    if _ASSISTANT_OFFER_RE.search(text):
        return True
    if _ASSISTANT_OFFER_RE.search(_strip_accents(text)):
        return True
    return False


async def _ask_model(model: str, query: str, answer_tail: str, retry_note: str = "") -> List[str]:
    user_content = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Respuesta dada:\n{answer_tail}\n\n"
        "Generá 3 preguntas de seguimiento relevantes."
    )
    if retry_note:
        user_content += f"\n\nIMPORTANTE: {retry_note}"

    resp = await _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FOLLOWUP_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=200,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        return []

    out: List[str] = []
    for item in data[:5]:  # accept extras so we can drop bad ones
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip('"').strip()
        if cleaned and len(cleaned) <= 140:
            out.append(cleaned)
    return out


async def generate_followups(
    query: str,
    answer: str,
    model: str | None = None,
) -> List[str]:
    """Return up to 3 short follow-up questions in Spanish.

    Drops any item phrased from the assistant's perspective ("¿Querés
    que te explique...?"). If the first batch leaves us short, we ask
    the model once more with an explicit corrective note. Fails silently
    — an empty list hides the chips in the UI.
    """
    if not answer.strip():
        return []

    trimmed_answer = answer.strip()
    if len(trimmed_answer) > 1200:
        trimmed_answer = trimmed_answer[-1200:]

    chosen_model = model or OPENAI_MODEL_FAST

    try:
        candidates = await _ask_model(chosen_model, query, trimmed_answer)
    except Exception:
        return []

    valid = [c for c in candidates if not _is_assistant_perspective(c)]

    # If the filter killed too many, retry once with an explicit nudge.
    if len(valid) < 3:
        try:
            extra = await _ask_model(
                chosen_model,
                query,
                trimmed_answer,
                retry_note=(
                    "Las preguntas se envían tal cual al asistente como nuevo "
                    "turno del usuario. NO uses '¿Querés...?', '¿Te interesa...?', "
                    "'¿Necesitás...?', '¿Preferís...?', '¿Te muestro...?'. "
                    "Reformulá como el usuario pidiendo: '¿Cuál es...?', "
                    "'¿Cómo...?', 'Mostrame...', '¿Qué pasó con...?'."
                ),
            )
            for c in extra:
                if c not in valid and not _is_assistant_perspective(c):
                    valid.append(c)
        except Exception:
            pass

    return valid[:3]
