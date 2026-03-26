# -*- coding: utf-8 -*-
"""Prompt loading, confidence assessment, source formatting."""

import os
from typing import Dict, List, Any, Optional

from .scoring import sim_from_distance

# ─── Source / page helpers ───────────────────────────────────────────────────

SOURCE_KEYS = ["rel_path", "source", "file", "filename", "doc", "title", "name", "path", "uri", "url", "id"]
PAGE_KEYS = ["page", "page_number", "pageIndex", "page_num", "pageno", "pageNo"]


def best_source_name(md: Dict[str, Any]) -> str:
    for k in SOURCE_KEYS:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            try:
                return os.path.basename(v) or v
            except Exception:
                return v
    return "desconocido"


def best_page(md: Dict[str, Any]) -> Optional[str]:
    for k in PAGE_KEYS:
        v = md.get(k)
        if v not in (None, ""):
            return str(v)
    return None


def format_source_tag(md: Dict[str, Any]) -> str:
    name = best_source_name(md)
    parts = [name]
    page = best_page(md)
    if page:
        parts.append(f"Pag. {page}")
    d = md.get("date_iso")
    if d:
        parts.append(f"Fecha: {d}")
    return " | ".join(parts)


# ─── Prompt loading ─────────────────────────────────────────────────────────

_FALLBACK_PROMPT = (
    "Eres un asistente que responde SOLO con el CONTEXTO provisto.\n"
    "Puedes sintetizar y combinar fragmentos del contexto.\n"
    "Si falta información crítica para responder con precisión, di exactamente:\n"
    "\"No se puede responder con el contexto disponible.\"\n\n"
    "=== CONTEXTO ===\n{context}\n\n=== PREGUNTA ===\n{question}\n"
)


def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return _FALLBACK_PROMPT


# ─── Confidence assessment ──────────────────────────────────────────────────

def assess_confidence(
    top_results: list,
    sim_threshold: float = 0.35,
    min_good_results: int = 2,
) -> dict:
    if not top_results:
        return {"confident": False, "reason": "no_results", "avg_sim": 0.0, "good_count": 0}
    sims = [sim_from_distance(dist) for (_, dist, _) in top_results]
    avg_sim = sum(sims) / len(sims)
    good_count = sum(1 for s in sims if s >= sim_threshold)
    if good_count < min_good_results:
        return {"confident": False, "reason": "low_relevance", "avg_sim": avg_sim, "good_count": good_count}
    return {"confident": True, "reason": "ok", "avg_sim": avg_sim, "good_count": good_count}


def build_confidence_note(conf: dict) -> str:
    if conf["confident"]:
        return ""
    if conf["reason"] == "no_results":
        return (
            "\n\n\u26a0\ufe0f NOTA INTERNA: No se encontraron resultados relevantes en los documentos. "
            "Indic\u00e1 que no hay informaci\u00f3n disponible y pregunt\u00e1 "
            "al usuario si puede reformular la pregunta o dar m\u00e1s detalles "
            "(per\u00edodo, regi\u00f3n, indicador espec\u00edfico, etc.)."
        )
    if conf["reason"] == "low_relevance":
        return (
            f"\n\n\u26a0\ufe0f NOTA INTERNA: Solo {conf['good_count']} fragmentos tienen "
            f"relevancia aceptable (similitud promedio: {conf['avg_sim']:.2f}). "
            "La informaci\u00f3n podr\u00eda ser parcial. Respond\u00e9 con lo que tengas "
            "pero advert\u00ed al usuario y pedile que precise su consulta "
            "(ej: pa\u00eds, per\u00edodo, indicador espec\u00edfico)."
        )
    return ""
