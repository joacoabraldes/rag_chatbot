# -*- coding: utf-8 -*-
"""Controlled topic taxonomy utilities.

This module centralizes the allowed economic topics used for:
- chunk topic tagging during ingestion/retagging,
- metadata filtering in retrieval,
- prompt constraints for LLM-based extraction tasks.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Sequence


TOPIC_TAXONOMY: List[Dict[str, object]] = [
    {
        "key": "inflacion",
        "label": "inflacion",
        "aliases": ["ipc", "precios", "indice de precios", "inflacion mensual"],
    },
    {
        "key": "tipo_cambio",
        "label": "tipo de cambio",
        "aliases": ["dolar", "fx", "devaluacion", "crawling peg"],
    },
    {
        "key": "actividad",
        "label": "actividad economica",
        "aliases": [
            "actividad",
            "nivel de actividad",
            "crecimiento",
            "recesion",
            "pbi",
            "producto bruto",
            "producto interno bruto",
            "gdp",
        ],
    },
    {
        "key": "tasas",
        "label": "tasas de interes",
        "aliases": ["tasa", "interes", "rendimientos"],
    },
    {
        "key": "politica_monetaria",
        "label": "politica monetaria",
        "aliases": ["bcra", "banco central", "liquidez", "agregados monetarios"],
    },
    {
        "key": "fiscal",
        "label": "politica fiscal",
        "aliases": ["deficit", "superavit", "gasto publico", "ingresos fiscales"],
    },
    {
        "key": "empleo",
        "label": "empleo",
        "aliases": [
            "mercado laboral",
            "desempleo",
            "ocupacion",
            "salarios",
            "ingresos",
            "paritarias",
            "salario real",
        ],
    },
    {
        "key": "sector_externo",
        "label": "sector externo",
        "aliases": ["balanza comercial", "exportaciones", "importaciones", "cuenta corriente"],
    },
    {
        "key": "reservas",
        "label": "reservas internacionales",
        "aliases": ["reservas", "reservas del bcra"],
    },
    {
        "key": "deuda",
        "label": "deuda",
        "aliases": ["bonos", "riesgo pais", "financiamiento", "credito soberano"],
    },
]


_BY_KEY: Dict[str, Dict[str, object]] = {
    str(item["key"]): item for item in TOPIC_TAXONOMY
}


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("-", " ").replace("_", " ")


def taxonomy_keys() -> List[str]:
    return [str(item["key"]) for item in TOPIC_TAXONOMY]


def taxonomy_labels() -> Dict[str, str]:
    return {str(item["key"]): str(item["label"]) for item in TOPIC_TAXONOMY}


def taxonomy_for_prompt() -> str:
    lines = []
    for item in TOPIC_TAXONOMY:
        key = str(item["key"])
        label = str(item["label"])
        aliases = ", ".join(str(a) for a in item.get("aliases", []))
        lines.append(f"- {key}: {label} (alias: {aliases})")
    return "\n".join(lines)


def canonicalize_topic(topic: str) -> str | None:
    """Map a free-form topic string to a controlled taxonomy key."""
    if not topic:
        return None
    n = _normalize(topic)

    # Direct key/label match first.
    for item in TOPIC_TAXONOMY:
        key = str(item["key"])
        if n == _normalize(key) or n == _normalize(str(item["label"])):
            return key

    # Then alias exact/contains match.
    for item in TOPIC_TAXONOMY:
        key = str(item["key"])
        aliases = [str(a) for a in item.get("aliases", [])]
        for alias in aliases:
            alias_n = _normalize(alias)
            if n == alias_n or alias_n in n or n in alias_n:
                return key

    return None


def canonicalize_topics(raw_topics: Sequence[str]) -> List[str]:
    """Normalize and deduplicate topic keys while preserving order."""
    seen = set()
    out: List[str] = []
    for raw in raw_topics:
        key = canonicalize_topic(raw)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def topic_labels_from_keys(keys: Iterable[str]) -> List[str]:
    labels = taxonomy_labels()
    out = []
    for k in keys:
        out.append(labels.get(k, k))
    return out


def build_topic_metadata(topic_keys: Sequence[str], max_slots: int = 4) -> Dict[str, str]:
    """Build normalized metadata fields used by retrieval filters.

    Returns fields:
    - topic_keys: comma-separated canonical keys
    - topic_tags: comma-separated human labels
    - topic_tag_1..topic_tag_N: fixed slots for exact-match filtering
    """
    keys = canonicalize_topics(topic_keys)[:max_slots]
    labels = topic_labels_from_keys(keys)
    md: Dict[str, str] = {
        "topic_keys": ",".join(keys),
        "topic_tags": ",".join(labels),
    }
    for i in range(max_slots):
        md[f"topic_tag_{i + 1}"] = keys[i] if i < len(keys) else ""
    return md
