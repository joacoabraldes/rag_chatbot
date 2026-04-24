# -*- coding: utf-8 -*-
"""Lightweight evaluation harness for the RAG pipeline.

Runs a golden set of queries through the same code path the /chat
endpoint uses (router + query rewriting + retrieval + rerank), and
reports simple metrics:

- router_correct : did the intent classifier match expects_retrieve?
- retrieve_ok    : for retrieval queries, did we get any chunks back?
- keyword_hit    : did the retrieved chunks contain the expected keywords?
- file_hit       : when expect_source_file_contains is set, does the
                   expected substring appear in any retrieved source_file?

Usage:
    python scripts/eval.py                          # default golden set
    python scripts/eval.py --golden path/to.jsonl   # custom set
    python scripts/eval.py --verbose                # show each result

Extend ``tests/eval/golden.jsonl`` with real expectations as you iterate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 stdout on Windows so Spanish characters and box-drawing glyphs render.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Load env vars so the OpenAI key works when run from CLI.
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on the path regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RERANK_TOP_N, TOP_K  # noqa: E402
from core.query_rewriter import rewrite_standalone  # noqa: E402
from core.reranker import rerank  # noqa: E402
from core.router import classify_intent  # noqa: E402
from core.vectorstore import query_chunks_async  # noqa: E402

DEFAULT_GOLDEN = ROOT / "tests" / "eval" / "golden.jsonl"


def load_golden(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(json.loads(line))
    return entries


async def run_case(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single golden entry and return a result dict."""
    query = entry["query"]
    history = entry.get("history") or []
    expects_retrieve = entry.get("expects_retrieve")
    expect_keywords: List[str] = [
        k.lower() for k in (entry.get("expect_keywords") or [])
    ]
    expect_file = entry.get("expect_source_file_contains")

    t0 = time.perf_counter()

    # 1) Query rewrite when history is present.
    retrieval_query = await rewrite_standalone(query, history) if history else query

    # 2) Classifier + retrieval in parallel (mirrors /chat pipeline).
    async def _classify():
        try:
            return await classify_intent(query)
        except Exception:
            return True

    needs_retrieval, chunks = await asyncio.gather(
        _classify(),
        query_chunks_async(retrieval_query, k=TOP_K),
    )

    # 3) Rerank if retrieval is needed and we got something.
    if needs_retrieval and chunks:
        chunks = rerank(retrieval_query, chunks, top_n=RERANK_TOP_N)

    dt = (time.perf_counter() - t0) * 1000

    # Metrics
    router_correct = (
        expects_retrieve is None or bool(needs_retrieval) == bool(expects_retrieve)
    )
    retrieve_ok = (not needs_retrieval) or (len(chunks) > 0)

    joined_text = " ".join(c.get("text", "") for c in chunks).lower()
    keyword_hit = all(k in joined_text for k in expect_keywords) if expect_keywords else True

    files_retrieved = [c["metadata"].get("source_file", "") for c in chunks]
    file_hit = (
        expect_file is None
        or any(expect_file in f for f in files_retrieved)
    )

    return {
        "id": entry.get("id", query[:30]),
        "query": query,
        "rewritten": retrieval_query if retrieval_query != query else None,
        "needs_retrieval": needs_retrieval,
        "n_chunks": len(chunks),
        "router_correct": router_correct,
        "retrieve_ok": retrieve_ok,
        "keyword_hit": keyword_hit,
        "file_hit": file_hit,
        "latency_ms": round(dt),
        "files": files_retrieved,
    }


def fmt_bool(ok: bool) -> str:
    return "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"


def print_report(results: List[Dict[str, Any]], verbose: bool) -> None:
    total = len(results)
    agg = {
        "router_correct": sum(1 for r in results if r["router_correct"]),
        "retrieve_ok": sum(1 for r in results if r["retrieve_ok"]),
        "keyword_hit": sum(1 for r in results if r["keyword_hit"]),
        "file_hit": sum(1 for r in results if r["file_hit"]),
    }
    avg_latency = sum(r["latency_ms"] for r in results) / max(total, 1)

    if verbose:
        print("\n" + "─" * 78)
        print(f"{'ID':<22} {'Router':>7} {'Retr':>5} {'Kw':>4} {'File':>5} {'ms':>6}  Query")
        print("─" * 78)
        for r in results:
            print(
                f"{r['id'][:20]:<22} "
                f"{fmt_bool(r['router_correct']):>7} "
                f"{fmt_bool(r['retrieve_ok']):>5} "
                f"{fmt_bool(r['keyword_hit']):>4} "
                f"{fmt_bool(r['file_hit']):>5} "
                f"{r['latency_ms']:>6}  "
                f"{r['query'][:40]}"
            )
            if r["rewritten"]:
                print(f"   ↳ rewrite: {r['rewritten'][:70]}")
        print()

    print("═" * 60)
    print(f"  EVAL SUMMARY — {total} casos — avg latency {avg_latency:.0f}ms")
    print("─" * 60)
    for metric, passed in agg.items():
        pct = 100.0 * passed / total if total else 0.0
        bar = "█" * int(pct / 5)
        print(f"  {metric:<18} {passed:>2}/{total}  {pct:5.1f}%  {bar}")
    print("═" * 60)


async def main_async(path: Path, verbose: bool) -> int:
    entries = load_golden(path)
    if not entries:
        print(f"No golden entries in {path}", file=sys.stderr)
        return 2
    print(f"Running {len(entries)} golden cases from {path}...\n")
    results = []
    for entry in entries:
        r = await run_case(entry)
        results.append(r)
        if not verbose:
            status = all([r["router_correct"], r["retrieve_ok"], r["keyword_hit"], r["file_hit"]])
            mark = fmt_bool(status)
            print(f"  {mark}  {r['id']:<24} {r['latency_ms']:>5}ms  {r['query'][:50]}")
    print_report(results, verbose=verbose)
    all_pass = all(
        r["router_correct"] and r["retrieve_ok"] and r["keyword_hit"] and r["file_hit"]
        for r in results
    )
    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG pipeline eval harness")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if not args.golden.exists():
        print(f"Golden set not found: {args.golden}", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args.golden, args.verbose))


if __name__ == "__main__":
    sys.exit(main())
