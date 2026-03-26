# -*- coding: utf-8 -*-
"""Parallel retrieval, deduplication, caching, and context building."""

import sys, datetime, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

from .prompt import format_source_tag

# ─── Parallel retrieval ──────────────────────────────────────────────────────


def retrieve_from_collections_parallel(
    vectordbs: Dict[str, Any],
    queries: List[str],
    fetch_k_each: List[int],
    max_workers: int = 4,
    date_filter: Optional[dict] = None,
) -> list:
    """Query all (collection, query) pairs concurrently and merge results."""
    chroma_filter = {"date_iso": date_filter} if date_filter else None
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
            if k_each <= 0:
                continue
            for qexp in queries:
                if chroma_filter:
                    future = executor.submit(
                        vdb.similarity_search_with_score, qexp, k_each,
                        filter=chroma_filter,
                    )
                else:
                    future = executor.submit(
                        vdb.similarity_search_with_score, qexp, k_each,
                    )
                futures_map[future] = col

        candidates: list = []
        try:
            for future in as_completed(futures_map, timeout=30):
                col = futures_map[future]
                try:
                    results = future.result(timeout=30)
                    for doc, dist in results:
                        md = dict(doc.metadata or {})
                        md["_collection"] = col
                        doc.metadata = md
                        candidates.append((doc, dist))
                except TimeoutError:
                    print(f"[RAG] Collection '{col}' timed out", file=sys.stderr)
                except Exception as e:
                    print(f"[RAG] Collection '{col}' retrieval error: {e}", file=sys.stderr)
        except TimeoutError:
            print("[RAG] retrieve_from_collections_parallel overall timeout", file=sys.stderr)

    return candidates


# ─── Deduplication ───────────────────────────────────────────────────────────


def deduplicate_candidates(candidates: list, content_window: int = 150) -> list:
    """Deduplicate chunks by content + source to avoid repeated results."""
    seen = set()
    unique = []
    for doc, dist in candidates:
        key = (
            doc.page_content[:content_window],
            doc.metadata.get("rel_path", ""),
            doc.metadata.get("chunk_id", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append((doc, dist))
    return unique


# ─── Context building ───────────────────────────────────────────────────────


def build_context(top_docs, formatter=format_source_tag) -> str:
    """Build the context string from a list of (doc, ...) tuples or documents."""
    parts = []
    for i, item in enumerate(top_docs, 1):
        doc = item[0] if isinstance(item, (tuple, list)) else item
        tag = formatter(doc.metadata)
        parts.append(f"[{i}] ({tag})\n{doc.page_content}")
    return "\n\n".join(parts)


# ─── Misc ────────────────────────────────────────────────────────────────────


def split_k_across(n_items: int, k: int) -> List[int]:
    base = k // n_items
    rem = k % n_items
    sizes = [base] * n_items
    for i in range(rem):
        sizes[i] += 1
    return sizes


# ─── Retrieval result cache ──────────────────────────────────────────────────

_retrieval_cache: Dict[str, Any] = {}
RETRIEVAL_CACHE_TTL = 300
RETRIEVAL_CACHE_MAX_SIZE = 100


def make_cache_key(query: str, cols: List[str], k: int, recency_weight: float) -> str:
    payload = f"{query}|{sorted(cols)}|{k}|{recency_weight:.2f}"
    return hashlib.md5(payload.encode()).hexdigest()


def get_cached_results(key: str) -> Optional[list]:
    entry = _retrieval_cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if (datetime.datetime.now().timestamp() - ts) > RETRIEVAL_CACHE_TTL:
        del _retrieval_cache[key]
        return None
    return results


def set_cached_results(key: str, results: list) -> None:
    if len(_retrieval_cache) >= RETRIEVAL_CACHE_MAX_SIZE:
        oldest_key = min(_retrieval_cache, key=lambda k: _retrieval_cache[k][0])
        del _retrieval_cache[oldest_key]
    _retrieval_cache[key] = (datetime.datetime.now().timestamp(), results)


def clear_retrieval_cache() -> None:
    _retrieval_cache.clear()
