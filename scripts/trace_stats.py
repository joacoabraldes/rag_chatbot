#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate stats from logs/traces.jsonl.

This is the DIY equivalent of a dashboard: read the trace file, compute
per-stage latency percentiles, total cost, tokens by model, tool counts,
and recent errors. Useful to answer questions like:
    - which stage is slowest at p95?
    - how much have we spent today?
    - which tools is the model actually calling?
    - what's failing and where?

Usage:
    python scripts/trace_stats.py
    python scripts/trace_stats.py --last 50
    python scripts/trace_stats.py --since 2026-05-07
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Add project root so we can reuse the price table.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.observability import compute_cost  # noqa: E402

_TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _filter_records(
    records: List[Dict[str, Any]],
    last: Optional[int],
    since: Optional[str],
) -> List[Dict[str, Any]]:
    out = records
    if since:
        out = [r for r in out if str(r.get("ts", ""))[:10] >= since]
    if last and last > 0:
        out = out[-last:]
    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _fmt_ms(v: float) -> str:
    return f"{v:.0f}"


def _fmt_money(v: float) -> str:
    return f"${v:.4f}"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _summarize_stages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_stage: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        for name, attrs in (r.get("spans") or {}).items():
            d = attrs.get("duration_ms")
            if isinstance(d, (int, float)):
                by_stage[name].append(float(d))

    out = []
    for stage, values in by_stage.items():
        out.append(
            {
                "stage": stage,
                "count": len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
        )
    out.sort(key=lambda x: x["p95"], reverse=True)
    return out


def _summarize_tools(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_tool: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "duration_ms": [], "errors": 0}
    )
    for r in records:
        for t in r.get("tools") or []:
            slot = by_tool[t.get("name", "?")]
            slot["calls"] += 1
            d = t.get("duration_ms")
            if isinstance(d, (int, float)):
                slot["duration_ms"].append(float(d))
            if t.get("error"):
                slot["errors"] += 1

    out = []
    for name, data in by_tool.items():
        durations = data["duration_ms"]
        avg = sum(durations) / len(durations) if durations else 0.0
        out.append(
            {
                "name": name,
                "calls": data["calls"],
                "avg_ms": avg,
                "errors": data["errors"],
            }
        )
    out.sort(key=lambda x: x["calls"], reverse=True)
    return out


def _summarize_usage(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    )
    for r in records:
        for model, u in (r.get("usage") or {}).items():
            slot = by_model[model]
            slot["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            slot["completion_tokens"] += int(u.get("completion_tokens") or 0)
            slot["calls"] += int(u.get("calls") or 0)

    out = []
    for model, slot in by_model.items():
        cost = compute_cost(model, slot["prompt_tokens"], slot["completion_tokens"])
        out.append(
            {
                "model": model,
                "prompt_tokens": slot["prompt_tokens"],
                "completion_tokens": slot["completion_tokens"],
                "calls": slot["calls"],
                "cost_usd": cost,
            }
        )
    out.sort(key=lambda x: x["cost_usd"], reverse=True)
    return out


def _print_table(rows: List[Dict[str, Any]], cols: List[tuple[str, str, int]]) -> None:
    """Render a list of dicts as a fixed-width table.

    cols: list of (header, key, width) tuples.
    """
    header = "  ".join(h.ljust(w) for h, _, w in cols)
    print(f"  {header}")
    print(f"  {'-' * (len(header))}")
    for row in rows:
        line = "  ".join(str(row.get(k, "")).ljust(w) for _, k, w in cols)
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumir traces JSONL")
    parser.add_argument("--last", type=int, default=None, help="Tomar las últimas N entradas")
    parser.add_argument("--since", type=str, default=None, help="ISO date YYYY-MM-DD")
    parser.add_argument("--path", type=Path, default=_TRACE_PATH, help="Ruta al traces.jsonl")
    parser.add_argument("--errors", action="store_true", help="Solo mostrar errores")
    args = parser.parse_args()

    all_records = _read_records(args.path)
    records = _filter_records(all_records, args.last, args.since)

    if not records:
        print(f"Sin traces en {args.path}")
        return

    ts_min = min(str(r.get("ts", "")) for r in records)[:19]
    ts_max = max(str(r.get("ts", "")) for r in records)[:19]
    n_total = len(records)
    n_errors = sum(1 for r in records if r.get("status") == "error")
    total_cost = sum(float(r.get("cost_usd") or 0.0) for r in records)
    avg_cost = total_cost / n_total if n_total else 0.0
    max_cost = max((float(r.get("cost_usd") or 0.0) for r in records), default=0.0)
    durations = [float(r.get("duration_ms") or 0.0) for r in records]

    if args.errors:
        errors = [r for r in records if r.get("status") == "error"]
        print(f"\nERRORES ({len(errors)} de {n_total})")
        print("=" * 60)
        for r in errors[-50:]:
            for e in r.get("errors") or []:
                print(
                    f"  {r.get('request_id'):<14} | {str(r.get('ts'))[:19]} | "
                    f"{e.get('stage'):<14} | {e.get('kind'):<24} | {e.get('msg', '')[:80]}"
                )
        return

    print()
    print(f"RAG trace stats - {n_total} requests")
    print("=" * 60)
    print(f"  ventana: {ts_min}  ->  {ts_max}")
    print(f"  errores: {n_errors} ({n_errors / n_total * 100:.1f}%)")
    print(f"  request total: p50 {_fmt_ms(_percentile(durations, 0.5))}ms | "
          f"p95 {_fmt_ms(_percentile(durations, 0.95))}ms | "
          f"max {_fmt_ms(max(durations) if durations else 0)}ms")

    # Stages
    print("\nLatencia por etapa (ms):")
    stage_rows = _summarize_stages(records)
    rows = [
        {
            "stage": s["stage"],
            "count": s["count"],
            "p50": _fmt_ms(s["p50"]),
            "p95": _fmt_ms(s["p95"]),
            "max": _fmt_ms(s["max"]),
        }
        for s in stage_rows
    ]
    _print_table(
        rows,
        cols=[("stage", "stage", 18), ("count", "count", 6), ("p50", "p50", 8), ("p95", "p95", 8), ("max", "max", 8)],
    )

    # Cost & tokens
    print("\nCosto y tokens:")
    print(f"  total: {_fmt_money(total_cost)}")
    print(f"  por request: avg {_fmt_money(avg_cost)} | max {_fmt_money(max_cost)}")
    print()
    usage_rows = _summarize_usage(records)
    rows = [
        {
            "model": u["model"],
            "calls": u["calls"],
            "in": _fmt_tokens(u["prompt_tokens"]),
            "out": _fmt_tokens(u["completion_tokens"]),
            "cost": _fmt_money(u["cost_usd"]),
        }
        for u in usage_rows
    ]
    _print_table(
        rows,
        cols=[("model", "model", 16), ("calls", "calls", 6), ("in", "in", 8), ("out", "out", 8), ("cost", "cost", 10)],
    )

    # Tools
    tool_rows = _summarize_tools(records)
    if tool_rows:
        print("\nTool calls:")
        rows = [
            {
                "tool": t["name"],
                "calls": t["calls"],
                "avg_ms": _fmt_ms(t["avg_ms"]),
                "errors": t["errors"],
            }
            for t in tool_rows
        ]
        _print_table(
            rows,
            cols=[("tool", "tool", 28), ("calls", "calls", 6), ("avg_ms", "avg_ms", 8), ("errors", "errors", 6)],
        )

    # Errors snippet
    if n_errors:
        print(f"\nÚltimos errores (mostrando 10 de {n_errors}):")
        errors = [r for r in records if r.get("status") == "error"]
        for r in errors[-10:]:
            for e in r.get("errors") or []:
                print(
                    f"  {r.get('request_id'):<14} | {str(r.get('ts'))[:19]} | "
                    f"{e.get('stage'):<14} | {e.get('kind'):<24} | {e.get('msg', '')[:60]}"
                )


if __name__ == "__main__":
    main()
