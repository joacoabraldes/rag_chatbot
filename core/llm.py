# -*- coding: utf-8 -*-
"""LLM-powered utilities: query contextualisation, expansion, suggested/followup questions."""

import json
from typing import List, Dict, Any, Optional

from openai import OpenAI, APITimeoutError, APIConnectionError

from .prompt import best_source_name
from .text import short_preview

# ─── Query contextualisation (multi-turn -> self-contained query) ────────────

_CONTEXTUALISE_SYSTEM = (
    "You receive a conversation history and the user's latest message.\n"
    "Your ONLY job is to rewrite the latest message into a SELF-CONTAINED search "
    "query that includes all necessary context (topic, country, indicator, time "
    "period, etc.) drawn from the conversation history.\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else.\n"
    "- Keep the same language as the user.\n"
    "- If the latest message is already self-contained, return it unchanged.\n"
    "- Do NOT answer the question — just reformulate it."
)


def contextualize_query(
    user_input: str,
    history: list,
    openai_model: str = "gpt-5-mini",
    max_pairs: int = 5,
) -> str:
    """Rewrite a vague follow-up into a self-contained search query."""
    if not history:
        return user_input

    recent = history[-(max_pairs * 2):]
    turns = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            turns.append(f"{role}: {content[:500]}")

    if not turns:
        return user_input

    conversation_block = "\n".join(turns)
    user_prompt = (
        f"=== CONVERSATION HISTORY ===\n{conversation_block}\n\n"
        f"=== LATEST USER MESSAGE ===\n{user_input}\n\n"
        "Rewritten self-contained query:"
    )

    try:
        client = OpenAI(timeout=10.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _CONTEXTUALISE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else user_input
    except (APITimeoutError, APIConnectionError):
        return user_input
    except Exception:
        return user_input


# ─── Query expansion (LLM-powered) ──────────────────────────────────────────

_EXPANSION_SYSTEM = (
    "You are a search-query expansion assistant for a Spanish-language RAG system "
    "about economics and finance.\n"
    "Given the user's query, generate 3-5 alternative phrasings/search queries that "
    "capture the same intent but use different vocabulary, synonyms, or related terms.\n"
    "Return ONLY a JSON array of strings. Example: [\"query 1\", \"query 2\", \"query 3\"]\n"
    "Rules:\n"
    "- Keep queries in the same language as the original.\n"
    "- Include relevant acronyms, technical synonyms, and colloquial variants.\n"
    "- Do NOT add explanations, just the JSON array."
)


def expand_queries_llm(
    query: str,
    openai_model: str = "gpt-5-mini",
) -> List[str]:
    """Use an LLM to generate alternative search queries."""
    try:
        client = OpenAI(timeout=12.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _EXPANSION_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        expansions = json.loads(raw)
        if isinstance(expansions, list):
            return [query] + [e for e in expansions if isinstance(e, str) and e.strip()]
    except (APITimeoutError, APIConnectionError):
        pass
    except Exception:
        pass
    return [query]


def expand_queries_simple(q: str) -> List[str]:
    """Legacy hardcoded expansion (kept as fallback)."""
    base = q.strip()
    expansions = {base}
    lower = base.lower()
    if "régimen" in lower and ("monetario" in lower or "cambiario" in lower):
        expansions |= {
            lower.replace("monetario", "cambiario"),
            lower.replace("régimen", "esquema"),
            "crawling peg", "crawling-peg", "deslizamiento cambiario",
            "bandas cambiarias", "ancla nominal", "programa monetario",
            "liberalización cambiaria", "CEPO", "crawling",
        }
    if "abril" in lower:
        expansions |= {"mediados de abril", "segunda quincena de abril"}
    dollar_types = {"dólar", "dolar", "dollar"}
    if dollar_types & set(lower.split()):
        expansions |= {
            "dólar oficial", "dólar MEP", "dólar CCL", "dólar blue",
            "contado con liquidación", "cotización del dólar",
            "tipo de cambio", "MULC",
        }
    return [base] + [e for e in expansions if e != base]


# ─── Suggested / follow-up questions ────────────────────────────────────────

def generate_suggested_questions(
    vectordbs: Dict[str, Any],
    n: int = 5,
    openai_model: str = "gpt-5-mini",
) -> List[str]:
    """Sample corpus metadata and use LLM to generate contextually relevant questions."""
    _FALLBACK = [
        "Cuales son las ultimas decisiones de tasas del banco central?",
        "Resumime las condiciones macroeconomicas actuales.",
        "Cual es la perspectiva de inflacion?",
        "Factores de riesgo clave en los informes mas recientes?",
        "Compara la politica monetaria de las principales economias.",
    ]
    try:
        snippets = []
        for col, vdb in list(vectordbs.items())[:3]:
            try:
                result = vdb.get(limit=8, include=["metadatas", "documents"])
                for doc_text, meta in zip(
                    result.get("documents") or [],
                    result.get("metadatas") or [],
                ):
                    name = best_source_name(meta or {})
                    date = (meta or {}).get("date_iso", "")
                    preview = short_preview(doc_text or "", 120)
                    snippets.append(f"[{col}] {name} {date}: {preview}")
                    if len(snippets) >= 12:
                        break
            except Exception:
                continue
            if len(snippets) >= 12:
                break

        if not snippets:
            return _FALLBACK

        corpus_summary = "\n".join(snippets[:12])
        prompt = (
            f"Based on these document excerpts from an economics/finance knowledge base:\n\n"
            f"{corpus_summary}\n\n"
            f"Generate exactly {n} concise, specific questions a finance or economics analyst "
            f"would want to ask. Return ONLY a JSON array of {n} strings."
        )
        client = OpenAI(timeout=20.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        questions = json.loads(raw)
        if isinstance(questions, list) and len(questions) >= n:
            return [str(q) for q in questions[:n]]
    except Exception:
        pass
    return _FALLBACK


_FOLLOWUP_SYSTEM = (
    "Eres un asistente que genera preguntas de seguimiento. "
    "Dado una pregunta del usuario y la respuesta del asistente, "
    "genera exactamente 3 preguntas cortas y relevantes que el usuario "
    "podria querer hacer a continuacion. Las preguntas deben ser especificas "
    "y basadas en la respuesta dada. Devuelve SOLO un array JSON de 3 strings. "
    "Las preguntas deben estar en espanol."
)


def generate_followup_questions(
    user_query: str,
    assistant_answer: str,
    openai_model: str = "gpt-5-mini",
) -> List[str]:
    """Generate 3 follow-up questions based on the user's query and the assistant's answer."""
    try:
        client = OpenAI(timeout=15.0)
        prompt = (
            f"Pregunta del usuario: {user_query}\n\n"
            f"Respuesta del asistente: {assistant_answer[:1500]}\n\n"
            "Genera 3 preguntas de seguimiento:"
        )
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q) for q in questions[:3]]
    except Exception:
        pass
    return []
