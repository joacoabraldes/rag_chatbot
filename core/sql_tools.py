# -*- coding: utf-8 -*-
"""SQL tools exposed to the LLM via function calling.

Each tool is a small Python function that runs ONE bounded query against
Postgres and returns a list of dicts. They're the "hard data" half of
the hybrid RAG: the LLM gets narrative chunks from pgvector AND can
call these tools when it needs exact numbers.

Conventions (see .claude/skills/sql-builder/SKILL.md):
- Always parameterized SQL — never f-string user values.
- ISO date strings in/out (YYYY-MM-DD).
- Bounded result sets (LIMIT) to avoid blowing up the prompt.
- Validate inputs at the top — the LLM may pass garbage.
- Return typed dicts; never raw tuples.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import psycopg

from core.config import DATABASE_URL

log = logging.getLogger("rag.sql_tools")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no configurada en .env")
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def _validate_iso_date(value: str, field: str = "date") -> str:
    """Coerce 'YYYY-MM-DD' (or longer ISO) to plain YYYY-MM-DD."""
    if not isinstance(value, str):
        raise ValueError(f"{field} debe ser string ISO YYYY-MM-DD, recibí {type(value).__name__}")
    cleaned = value.strip()[:10]
    try:
        datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"{field} no parsea como YYYY-MM-DD: {value!r}") from e
    return cleaned


def _serialize(value: Any) -> Any:
    """Make a row value JSON-friendly for the tool result message."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        # Floats are good enough for prompt context; preserve precision in JSON.
        return float(value)
    return value


def _rows_to_dicts(cur) -> list[dict]:
    cols = [c.name for c in cur.description]
    return [
        {col: _serialize(val) for col, val in zip(cols, row)}
        for row in cur.fetchall()
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_fx(date_from: str, date_to: str | None = None) -> list[dict]:
    """Daily aggregated FX quotes from public.fx.

    Returns one row per date with columns: date, ccl, mepal (MEP via AL30),
    mepgd (MEP via GD30), cclgd (CCL via GD30), a3500 (BCRA reference),
    canje (CCL/MEP ratio), brecha (MEP/oficial - 1), last_mlc.

    All numeric values may be NULL for older dates with partial coverage.
    """
    df = _validate_iso_date(date_from, "date_from")
    dt = _validate_iso_date(date_to, "date_to") if date_to else df

    sql = """
        SELECT
            "date",
            ccl,
            mepal,
            mepgd,
            cclgd,
            a3500,
            canje,
            brecha,
            last_mlc
        FROM public.fx
        WHERE "date" BETWEEN %s AND %s
        ORDER BY "date"
        LIMIT 90
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (df, dt))
        return _rows_to_dicts(cur)


def get_forex_operations(
    date_from: str,
    date_to: str | None = None,
    instrumento: str | None = None,
    currency_out: str | None = None,
    currency_in: str | None = None,
    settle: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Intraday operations from the MAE (public.forex).

    Useful for: detalle de operaciones, volúmenes intradiarios por
    instrumento o por par de monedas, cotización de operaciones puntuales.

    Filters are AND-combined. Default LIMIT 100 — pedile fechas acotadas
    o instrumentos específicos para no traer ruido.
    """
    df = _validate_iso_date(date_from, "date_from")
    dt = _validate_iso_date(date_to, "date_to") if date_to else df

    if not isinstance(limit, int) or limit < 1:
        limit = 100
    limit = min(limit, 500)

    sql = """
        SELECT
            "date",
            rueda,
            instrumento,
            currency_out,
            currency_in,
            settle,
            settle_date,
            monto,
            cotizacion,
            hora
        FROM public.forex
        WHERE "date" BETWEEN %s AND %s
    """
    params: list[Any] = [df, dt]

    if instrumento:
        sql += " AND instrumento ILIKE %s"
        params.append(f"%{instrumento}%")
    if currency_out:
        sql += " AND TRIM(currency_out) = %s"
        params.append(currency_out.strip().upper())
    if currency_in:
        sql += " AND TRIM(currency_in) = %s"
        params.append(currency_in.strip().upper())
    if settle is not None:
        sql += " AND settle = %s"
        params.append(int(settle))

    sql += ' ORDER BY "date", hora NULLS LAST LIMIT %s'
    params.append(limit)

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur)


def get_forex_volume_summary(
    date_from: str,
    date_to: str | None = None,
    currency: str | None = None,
) -> list[dict]:
    """Daily volume summary aggregated across operations in public.forex.

    Returns per (date, currency_out): total volume in the source currency
    and operation count. Use for queries about MAE volume or BCRA
    intervention proxies.
    """
    df = _validate_iso_date(date_from, "date_from")
    dt = _validate_iso_date(date_to, "date_to") if date_to else df

    sql = """
        SELECT
            "date",
            COALESCE(TRIM(currency_out), '?') AS currency_out,
            SUM(monto) AS volumen_total,
            COUNT(*) AS n_operaciones,
            AVG(cotizacion) AS cotizacion_promedio
        FROM public.forex
        WHERE "date" BETWEEN %s AND %s
          AND monto IS NOT NULL
    """
    params: list[Any] = [df, dt]
    if currency:
        sql += " AND TRIM(currency_out) = %s"
        params.append(currency.strip().upper())
    sql += ' GROUP BY "date", TRIM(currency_out) ORDER BY "date" LIMIT 90'

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur)


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_fx",
            "description": (
                "Devuelve cotizaciones diarias agregadas del mercado argentino "
                "para un rango de fechas. Cada fila incluye: ccl (contado con "
                "liqui), mepal (MEP vía AL30), mepgd (MEP vía GD30), cclgd "
                "(CCL vía GD30), a3500 (referencia BCRA), canje (ratio CCL/MEP), "
                "brecha (MEP/oficial - 1), last_mlc (último precio MLC). "
                "USAR SIEMPRE que la pregunta requiera un valor exacto del dólar, "
                "MEP, CCL, brecha cambiaria o A3500 para una fecha."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Fecha desde, formato YYYY-MM-DD. Si la pregunta es sobre 'hoy' usá la fecha actual del prompt.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Fecha hasta inclusive, YYYY-MM-DD. Omitir para una sola fecha.",
                    },
                },
                "required": ["date_from"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forex_operations",
            "description": (
                "Devuelve operaciones intradiarias del MAE desde public.forex. "
                "Una fila por operación (date, rueda, instrumento, currency_out, "
                "currency_in, settle, monto, cotizacion, hora). Usar para detalle "
                "de operaciones puntuales, volúmenes intradiarios por instrumento "
                "o cotización de un instrumento específico. Para totales agregados, "
                "preferir get_forex_volume_summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "description": "Fecha desde, YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "Fecha hasta, YYYY-MM-DD. Default = date_from."},
                    "instrumento": {
                        "type": "string",
                        "description": "Filtro por nombre de instrumento (substring, ILIKE). Ej: 'USB', 'USMEP', 'USD/ARS'.",
                    },
                    "currency_out": {"type": "string", "description": "Moneda saliente (USB, UST, USMEP, etc.)"},
                    "currency_in": {"type": "string", "description": "Moneda entrante (ART, etc.)"},
                    "settle": {"type": "integer", "description": "Días de settlement (0=hoy, 1=T+1, etc.)"},
                    "limit": {"type": "integer", "description": "Tope de filas (default 100, máx 500)"},
                },
                "required": ["date_from"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forex_volume_summary",
            "description": (
                "Volúmenes diarios agregados de operaciones de divisas, sumados por "
                "fecha y moneda. Devuelve volumen_total, n_operaciones y "
                "cotizacion_promedio. Usar para preguntas sobre 'cuánto se operó', "
                "volumen MLC/MAE, o promedio de cotización por día."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                    "currency": {"type": "string", "description": "Filtro opcional por moneda saliente"},
                },
                "required": ["date_from"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_TOOL_MAP = {
    "get_fx": get_fx,
    "get_forex_operations": get_forex_operations,
    "get_forex_volume_summary": get_forex_volume_summary,
}


def execute_tool_sync(name: str, arguments: dict) -> dict:
    """Run a tool and wrap the result/error so it can be JSON-serialized
    and sent back to the model as a 'tool' role message.
    """
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"tool desconocida: {name!r}"}
    if not isinstance(arguments, dict):
        return {"error": "arguments debe ser un objeto JSON"}
    try:
        rows = fn(**arguments)
        return {"rows": rows, "n_rows": len(rows)}
    except (ValueError, TypeError) as e:
        return {"error": f"input inválido: {e}"}
    except psycopg.Error as e:
        log.exception("SQL error in tool %s", name)
        return {"error": f"error de base de datos: {e.__class__.__name__}"}
    except Exception as e:
        log.exception("Unexpected error in tool %s", name)
        return {"error": f"error inesperado: {e.__class__.__name__}: {e}"}


async def execute_tool(name: str, arguments: dict) -> dict:
    """Async wrapper — runs the sync tool in a thread pool."""
    return await asyncio.to_thread(execute_tool_sync, name, arguments)


def tool_result_message(tool_call_id: str, payload: dict) -> dict:
    """Build the 'tool' role message that goes back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False, default=str),
    }
