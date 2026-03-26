# -*- coding: utf-8 -*-
"""Date parsing and inference — merged from rag_utils + ingest."""

import re, datetime
from typing import Dict, Any, Optional

# ─── Patterns for free-text date extraction ──────────────────────────────────

DATE_TEXT_PATTERNS = [
    re.compile(r"(?P<d>[0-3]?\d)[-/.](?P<m>[01]?\d)[-/.](?P<y>\d{4})"),
    re.compile(r"(?P<y>\d{4})[-/.](?P<m>[01]?\d)[-/.](?P<d>[0-3]?\d)"),
]

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

# ─── Patterns for filename/metadata date extraction ──────────────────────────

DATE_FILENAME_PATTERNS = [
    re.compile(r"(?P<y>20\d{2}|19\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])"),
    re.compile(r"(?P<y>20\d{2}|19\d{2})[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<d>0[1-9]|[12]\d|3[01])"),
    re.compile(r"(?P<d>0[1-9]|[12]\d|3[01])[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<y>20\d{2}|19\d{2})"),
]


def normalize_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_date_from_text(text: str) -> Optional[str]:
    """Find a date inside free text (numeric patterns + Spanish month names)."""
    t = text.lower()
    for pat in DATE_TEXT_PATTERNS:
        hit = pat.search(t)
        if hit:
            y, mth, d = int(hit.group("y")), int(hit.group("m")), int(hit.group("d"))
            try:
                datetime.date(y, mth, d)
                return normalize_iso(y, mth, d)
            except ValueError:
                continue
    hit = re.search(r"(?P<d>[0-3]?\d)\s+de\s+(?P<mes>\w+)\s+de\s+(?P<y>\d{4})", t)
    if hit:
        d = int(hit.group("d"))
        y = int(hit.group("y"))
        mes_name = hit.group("mes")
        if mes_name in MESES:
            return normalize_iso(y, MESES[mes_name], d)
    return None


def parse_date_iso(md: Dict[str, Any]) -> Optional[datetime.date]:
    """Parse a date_iso metadata field into a datetime.date."""
    s = md.get("date_iso")
    if not isinstance(s, str):
        return None
    try:
        y, m, d = map(int, s[:10].split("-"))
        return datetime.date(y, m, d)
    except Exception:
        return None


def try_parse_date_from_string(s: str) -> Optional[str]:
    """Try to extract a date from a filename or metadata string."""
    if not s:
        return None
    for pat in DATE_FILENAME_PATTERNS:
        m = pat.search(s)
        if m:
            y = int(m.group("y"))
            mth = int(m.group("m"))
            d = int(m.group("d"))
            try:
                datetime.date(y, mth, d)
                return normalize_iso(y, mth, d)
            except ValueError:
                continue
    return None


def extract_date_range_from_query(query: str) -> Optional[dict]:
    """Extract date range filters from natural language query.

    Returns a dict like {"$gte": "2026-02-01", "$lte": "2026-02-28"} or None.
    """
    import calendar
    today = datetime.date.today()
    lower = query.lower()

    # "last month" / "el mes pasado" / "mes anterior"
    if any(p in lower for p in ["mes pasado", "mes anterior", "last month", "último mes", "ultimo mes"]):
        first = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        last_day = calendar.monthrange(first.year, first.month)[1]
        last = first.replace(day=last_day)
        return {"$gte": first.isoformat(), "$lte": last.isoformat()}

    # "last week" / "semana pasada" / "última semana"
    if any(p in lower for p in ["semana pasada", "última semana", "ultima semana", "last week"]):
        end = today - datetime.timedelta(days=today.weekday() + 1)
        start = end - datetime.timedelta(days=6)
        return {"$gte": start.isoformat(), "$lte": end.isoformat()}

    # "past N days" / "últimos N días"
    m = re.search(r'(?:últimos?|ultimos?|past|last)\s+(\d+)\s+(?:días|dias|days)', lower)
    if m:
        n = int(m.group(1))
        start = today - datetime.timedelta(days=n)
        return {"$gte": start.isoformat(), "$lte": today.isoformat()}

    # Specific month + year: "marzo 2026", "march 2026", "febrero 2026"
    for mes_name, mes_num in MESES.items():
        pattern = rf'{mes_name}\s+(?:de\s+)?(\d{{4}})'
        m_hit = re.search(pattern, lower)
        if m_hit:
            year = int(m_hit.group(1))
            last_day = calendar.monthrange(year, mes_num)[1]
            return {"$gte": f"{year:04d}-{mes_num:02d}-01", "$lte": f"{year:04d}-{mes_num:02d}-{last_day:02d}"}

    # English month names
    en_months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                 "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
    for name, num in en_months.items():
        pattern = rf'{name}\s+(\d{{4}})'
        m_hit = re.search(pattern, lower)
        if m_hit:
            year = int(m_hit.group(1))
            last_day = calendar.monthrange(year, num)[1]
            return {"$gte": f"{year:04d}-{num:02d}-01", "$lte": f"{year:04d}-{num:02d}-{last_day:02d}"}

    return None


def infer_date_iso(meta: Dict[str, Any], date_field: Optional[str]) -> Optional[str]:
    """Infer an ISO date from metadata fields or filename patterns."""
    if date_field and isinstance(meta.get(date_field), str):
        s = meta.get(date_field)
        iso = try_parse_date_from_string(s) or (s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else None)
        if iso:
            return iso

    for k in ["rel_path", "source", "id", "title", "name", "filename"]:
        v = meta.get(k)
        if isinstance(v, str):
            iso = try_parse_date_from_string(v)
            if iso:
                return iso
    return None
