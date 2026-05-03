# -*- coding: utf-8 -*-
"""Extract structured metadata filters from user queries for Chroma where clauses."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from core.config import OPENAI_MODEL_FAST
from core.prompts import FILTER_EXTRACTOR_PROMPT
from core.sections import canonicalize_section, section_keys, section_taxonomy_for_prompt
from core.taxonomy import canonicalize_topics, taxonomy_for_prompt

_client = AsyncOpenAI(timeout=10.0)


def _safe_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        return None


def _extract_compact_date_tokens(text: str) -> List[str]:
    return re.findall(r"\b(20\d{6})\b", text or "")


def _files_from_compact_date_tokens(
    query: str,
    file_dates: Dict[str, str],
) -> List[str]:
    files: List[str] = []
    for token in _extract_compact_date_tokens(query):
        date_iso = f"{token[0:4]}-{token[4:6]}-{token[6:8]}"
        for fname, fdate in file_dates.items():
            if fdate == date_iso and fname not in files:
                files.append(fname)
    return files


def _sanitize_source_files(raw_files: List[str], file_dates: Dict[str, str]) -> List[str]:
    valid_names = set(file_dates.keys())
    out: List[str] = []
    for name in raw_files:
        if not isinstance(name, str):
            continue
        n = name.strip()
        if n and n in valid_names and n not in out:
            out.append(n)
    return out


def _files_in_date_range(
    file_dates: Dict[str, str],
    date_from: str | None,
    date_to: str | None,
) -> List[str]:
    out: List[str] = []
    for fname, fdate in file_dates.items():
        if not fdate:
            continue
        if date_from and fdate < date_from:
            continue
        if date_to and fdate > date_to:
            continue
        out.append(fname)
    return sorted(out)


def _build_where_clause(extracted: Dict, file_dates: Dict[str, str]) -> Optional[Dict]:
    """Convert extracted filters into a ChromaDB where dict."""
    if not extracted:
        return None

    source_file_in = _sanitize_source_files(
        extracted.get("source_file_in") or [], file_dates
    )
    source_file_in.extend(
        f for f in _files_from_compact_date_tokens(extracted.get("query_text", ""), file_dates)
        if f not in source_file_in
    )

    # Resolve direct date hints to files if the model only emitted date range.
    date_from = _safe_iso_date(extracted.get("date_from"))
    date_to = _safe_iso_date(extracted.get("date_to"))
    if date_from and date_to and date_from == date_to:
        for fname, fdate in file_dates.items():
            if fdate == date_from and fname not in source_file_in:
                source_file_in.append(fname)

    topic_keys = canonicalize_topics(extracted.get("topic_keys_any") or [])

    sections_in: List[str] = []
    for s in extracted.get("section_in") or []:
        canon = canonicalize_section(s)
        if canon and canon not in sections_in:
            sections_in.append(canon)

    clauses: List[Dict] = []

    if source_file_in:
        clauses.append({"source_file": {"$in": source_file_in}})

    # Chroma's range operators are numeric-only; transform date constraints
    # into source_file sets using the summary's file_dates map.
    if date_from or date_to:
        range_files = _files_in_date_range(file_dates, date_from, date_to)
        if source_file_in:
            source_file_in = [f for f in source_file_in if f in set(range_files)]
        else:
            source_file_in = range_files

    # Re-append source_file clause after applying date-derived file filtering.
    clauses = [c for c in clauses if "source_file" not in c]
    if source_file_in:
        clauses.insert(0, {"source_file": {"$in": source_file_in}})

    if sections_in:
        clauses.append({"section": {"$in": sections_in}})

    if topic_keys:
        topic_slot_filters = [
            {"topic_tag_1": {"$in": topic_keys}},
            {"topic_tag_2": {"$in": topic_keys}},
            {"topic_tag_3": {"$in": topic_keys}},
            {"topic_tag_4": {"$in": topic_keys}},
        ]
        clauses.append({"$or": topic_slot_filters})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


async def extract_retrieval_filter(
    query: str,
    collection_summary: Optional[Dict],
    model: str | None = None,
) -> Dict:
    """Return structured filter info: {where, extracted, applied}.

    ``applied`` only indicates that at least one metadata constraint was parsed.
    It does not guarantee that retrieval will return any chunk.
    """
    file_dates = (collection_summary or {}).get("file_dates") or {}

    # Fast local fallback when no summary metadata exists.
    if not file_dates:
        return {"where": None, "extracted": {}, "applied": False}

    files_block = "\n".join(
        f"- {fname}: {date}" for fname, date in sorted(file_dates.items())
    )
    user_prompt = (
        f"Query: {query}\n\n"
        "Archivos disponibles (nombre -> fecha):\n"
        f"{files_block}\n\n"
        "Taxonomía de temas permitidos:\n"
        f"{taxonomy_for_prompt()}\n\n"
        "Taxonomía de secciones del informe:\n"
        f"{section_taxonomy_for_prompt()}\n\n"
        "Devolvé SOLO JSON."
    )

    extracted: Dict = {}
    try:
        resp = await _client.chat.completions.create(
            model=model or OPENAI_MODEL_FAST,
            messages=[
                {"role": "system", "content": FILTER_EXTRACTOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=220,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        maybe = json.loads(raw)
        if isinstance(maybe, dict):
            extracted = maybe
    except Exception:
        extracted = {}

    # Force text for compact-date fallback detection.
    extracted["query_text"] = query
    where = _build_where_clause(extracted, file_dates=file_dates)

    section_in: List[str] = []
    for s in extracted.get("section_in") or []:
        canon = canonicalize_section(s)
        if canon and canon not in section_in:
            section_in.append(canon)

    return {
        "where": where,
        "extracted": {
            "source_file_in": _sanitize_source_files(
                extracted.get("source_file_in") or [], file_dates
            ),
            "date_from": _safe_iso_date(extracted.get("date_from")),
            "date_to": _safe_iso_date(extracted.get("date_to")),
            "section_in": section_in,
            "topic_keys_any": canonicalize_topics(extracted.get("topic_keys_any") or []),
        },
        "applied": where is not None,
    }
