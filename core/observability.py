# -*- coding: utf-8 -*-
"""Per-request tracing for the RAG pipeline.

Each /chat request creates a ``RequestTrace`` that captures:
- a stable ``request_id`` to correlate every log line of the request
- one ``Span`` per pipeline stage (rewrite, router, retrieval, rerank, llm, ...)
- LLM token usage per model (real numbers from OpenAI's stream_options)
- tool calls executed against Postgres
- the chunks that ended up in the prompt
- estimated USD cost based on a hardcoded price table

Traces are appended as one JSON line per request to ``logs/traces.jsonl``.
Read them with ``scripts/trace_stats.py``.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("rag.observability")

# Single trace file. Append-only, one record per line. Rotation can be added
# later with logging.handlers.RotatingFileHandler if it ever grows.
_TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"


# ---------------------------------------------------------------------------
# Pricing — USD per 1M tokens (input, output).
# Update this table when OpenAI rates change. Models not listed are billed at 0
# and reported as ``unpriced`` so they're easy to spot in stats.
# ---------------------------------------------------------------------------

_MODEL_PRICES: Dict[str, tuple[float, float]] = {
    "gpt-5-mini":   (0.25, 2.00),
    "gpt-4.1-nano": (0.10, 0.40),
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return cost in USD for a single LLM call. Returns 0 for unpriced models."""
    price = _MODEL_PRICES.get(model)
    if not price:
        return 0.0
    input_per_million, output_per_million = price
    return (
        prompt_tokens * input_per_million / 1_000_000
        + completion_tokens * output_per_million / 1_000_000
    )


# ---------------------------------------------------------------------------
# Span — one stage of the pipeline
# ---------------------------------------------------------------------------


class Span:
    """Context manager that records duration and attributes for a stage.

    Use ``span.set(key=value)`` from inside the block to add attributes
    discovered while the stage is running. On exception the span captures
    the error kind and re-raises (we never swallow).
    """

    def __init__(self, trace: "RequestTrace", name: str, **attrs: Any) -> None:
        self.trace = trace
        self.name = name
        self.attrs: Dict[str, Any] = dict(attrs)
        self._t0: float = 0.0

    def __enter__(self) -> "Span":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = round((time.perf_counter() - self._t0) * 1000)
        record = {"duration_ms": duration_ms, **self.attrs}
        if exc_type is not None:
            record["error"] = f"{exc_type.__name__}: {exc}"
            self.trace.errors.append(
                {"stage": self.name, "kind": exc_type.__name__, "msg": str(exc)}
            )
        self.trace.spans[self.name] = record
        return False  # never swallow exceptions

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


# ---------------------------------------------------------------------------
# RequestTrace — the unit of observability
# ---------------------------------------------------------------------------


class RequestTrace:
    """One trace per /chat request.

    The trace lives for the duration of the request, accumulates spans/tools/
    chunks/usage as it goes, and is flushed to ``traces.jsonl`` on
    ``finalize()``. The same instance is also passed down into the LLM helpers
    so they can record token usage.
    """

    def __init__(
        self,
        query: str,
        show_sources: bool,
        history_len: int,
    ) -> None:
        self.request_id = "req_" + secrets.token_hex(6)
        self.ts = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        # Truncate query so the trace file doesn't balloon on long pastes.
        self.query = (query or "")[:500]
        self.show_sources = show_sources
        self.history_len = history_len

        self.spans: Dict[str, Dict[str, Any]] = {}
        self.tools: List[Dict[str, Any]] = []
        self.chunks_used: List[Dict[str, Any]] = []
        self.usage: Dict[str, Dict[str, int]] = {}  # by model
        self.errors: List[Dict[str, Any]] = []

    # ----- span / event helpers -----

    def span(self, name: str, **attrs: Any) -> Span:
        return Span(self, name, **attrs)

    def add_tool(
        self,
        name: str,
        duration_ms: float,
        n_rows: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "name": name,
            "duration_ms": round(duration_ms),
            "n_rows": n_rows if n_rows is not None else 0,
        }
        if error:
            entry["error"] = error
        self.tools.append(entry)

    def add_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        slot = self.usage.setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        slot["prompt_tokens"] += int(prompt_tokens or 0)
        slot["completion_tokens"] += int(completion_tokens or 0)
        slot["calls"] += 1

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        for c in chunks:
            md = c.get("metadata") or {}
            self.chunks_used.append(
                {
                    "id": c.get("id"),
                    "source_file": md.get("source_file"),
                    "pub_date": md.get("pub_date"),
                    "section": md.get("section"),
                    "similarity": round(float(c.get("similarity", 0.0)), 3),
                    "rerank_score": round(float(c.get("rerank_score", 0.0)), 3),
                }
            )

    # ----- finalization -----

    def total_cost_usd(self) -> float:
        total = 0.0
        for model, u in self.usage.items():
            total += compute_cost(model, u["prompt_tokens"], u["completion_tokens"])
        return round(total, 6)

    def finalize(self, status: str = "ok") -> Dict[str, Any]:
        duration_ms = round((time.perf_counter() - self._t0) * 1000)
        record = {
            "request_id": self.request_id,
            "ts": self.ts,
            "query": self.query,
            "show_sources": self.show_sources,
            "history_len": self.history_len,
            "duration_ms": duration_ms,
            "status": status if not self.errors else "error",
            "spans": self.spans,
            "tools": self.tools,
            "chunks": self.chunks_used,
            "usage": self.usage,
            "cost_usd": self.total_cost_usd(),
            "errors": self.errors,
        }
        _write_record(record)
        return record


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def _write_record(record: Dict[str, Any]) -> None:
    """Append one record as a single JSON line. Best-effort — never raises."""
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.exception("Failed to write trace record (request_id=%s)", record.get("request_id"))
