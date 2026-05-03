# -*- coding: utf-8 -*-
"""Section taxonomy for daily economic reports.

A *section* is the structural unit of the source PDF — the FX block,
the Tasas block, the Bonos block, etc. Sections are stored as a
single closed-set string per chunk so retrieval can pre-filter the
universe before the vector search ("traeme solo lo que sea de FX").

Sections are orthogonal to ``topic_keys``: a chunk in section ``fx``
can have ``topic_keys=["tipo_cambio", "politica_monetaria"]``. The
section reflects *where in the report* the chunk lives; topic_keys
reflect *what economic concepts it touches*.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Dict, List, Optional

from openai import OpenAI

from core.config import OPENAI_MODEL_FAST


SECTION_TAXONOMY: List[Dict[str, str]] = [
    {
        "key": "resumen",
        "label": "Resumen del día",
        "hint": "Primer bloque del informe — panorama general que mezcla "
        "internacional, commodities y Argentina.",
    },
    {
        "key": "internacional",
        "label": "Mercados internacionales",
        "hint": "Wall Street (SPX, NDX), tasas EE.UU., Europa, Asia, "
        "geopolítica, dólar global (DXY).",
    },
    {
        "key": "commodities",
        "label": "Commodities",
        "hint": "Petróleo (WTI, Brent), agro (soja, maíz, trigo, WASDE), "
        "oro, plata, metales.",
    },
    {
        "key": "fx",
        "label": "Tipo de cambio",
        "hint": "Dólar oficial, MEP, CCL, MLC, brecha cambiaria, "
        "intervención del BCRA en el spot.",
    },
    {
        "key": "tasas",
        "label": "Tasas y política monetaria",
        "hint": "Tasas en pesos, futuros de dólar, repos, badlar, "
        "licitaciones del Tesoro (LECAP, BONCAP), política monetaria local.",
    },
    {
        "key": "bonos",
        "label": "Bonos y deuda",
        "hint": "Soberanos hard dollar (GD, AL), bonos CER/Boncer (TX, TZX), "
        "riesgo país, curva de pesos a largo plazo.",
    },
    {
        "key": "reservas",
        "label": "Reservas BCRA",
        "hint": "Compras/ventas netas del BCRA, posición de reservas brutas y netas.",
    },
    {
        "key": "equity_arg",
        "label": "Equity argentino",
        "hint": "ADRs argentinos (YPF, GGAL, PAM, etc.), Merval al CCL.",
    },
]


_BY_KEY: Dict[str, Dict[str, str]] = {item["key"]: item for item in SECTION_TAXONOMY}


def section_keys() -> List[str]:
    return [item["key"] for item in SECTION_TAXONOMY]


def section_labels() -> Dict[str, str]:
    return {item["key"]: item["label"] for item in SECTION_TAXONOMY}


def section_taxonomy_for_prompt() -> str:
    return "\n".join(
        f"- {item['key']}: {item['label']} — {item['hint']}"
        for item in SECTION_TAXONOMY
    )


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def canonicalize_section(value: str | None) -> Optional[str]:
    """Map free-form text to a section key, or None if no match."""
    if not value:
        return None
    n = _normalize(value)
    if n in _BY_KEY:
        return n
    for item in SECTION_TAXONOMY:
        if n == _normalize(item["label"]):
            return item["key"]
    return None


_client = OpenAI(timeout=30.0)


def classify_section(
    header: str,
    body_preview: str,
    model: str | None = None,
) -> str:
    """Classify a (header, opening text) pair into a section key.

    ``header`` is the ALL-CAPS title we detected; ``body_preview`` is the
    first ~400 chars of the block. Falls back to ``resumen`` on any
    failure — that's the safest default for the opening block of a PDF.
    """
    llm_model = model or OPENAI_MODEL_FAST
    prompt = (
        "Clasificá el siguiente bloque de un informe económico diario en "
        "UNA sola sección, usando SOLO las keys de esta taxonomía:\n"
        f"{section_taxonomy_for_prompt()}\n\n"
        f"HEADER (puede estar vacío para el bloque inicial):\n{header.strip()[:200]}\n\n"
        f"PRIMERAS LÍNEAS DEL BLOQUE:\n{body_preview.strip()[:400]}\n\n"
        'Respondé SOLO JSON: {"section": "<key>"}'
    )
    try:
        resp = _client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=30,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        candidate = canonicalize_section(data.get("section"))
        if candidate:
            return candidate
    except Exception as e:
        print(f"  Warning: section classification failed: {e}")
    return "resumen"
