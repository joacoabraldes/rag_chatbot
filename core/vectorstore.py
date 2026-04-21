# -*- coding: utf-8 -*-
"""ChromaDB operations — add, query, list collections."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import chromadb

from core.config import CHROMA_COLLECTION, CHROMA_DB_DIR
from core.embedder import embed_query, embed_texts

_client: chromadb.PersistentClient | None = None

# Cache for collection summary — recomputed every TTL seconds.
# Full metadata scan on 2k chunks is ~100-300ms; cache avoids paying that per request.
_SUMMARY_TTL_SECONDS = 300.0
_summary_cache: Dict[str, Dict] = {}  # keyed by collection name


def get_client() -> chromadb.PersistentClient:
    """Lazy-load and cache the ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def get_collection(name: str | None = None) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=name or CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: List[Dict], collection_name: str | None = None) -> None:
    """Add pre-processed chunks to ChromaDB.

    Each chunk dict must have: chunk_id, text, metadata.
    Embeddings are computed here.
    """
    collection = get_collection(collection_name)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(documents).tolist()

    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )


def query_chunks(
    query: str,
    k: int = 8,
    collection_name: str | None = None,
) -> List[Dict]:
    """Query ChromaDB and return the top-k results as dicts."""
    collection = get_collection(collection_name)
    embedding = embed_query(query).tolist()

    count = collection.count()
    if count == 0:
        return []
    n = min(k, count)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[Dict] = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return chunks


async def query_chunks_async(
    query: str,
    k: int = 8,
    collection_name: str | None = None,
) -> List[Dict]:
    """Async wrapper around query_chunks — runs in a thread pool."""
    return await asyncio.to_thread(query_chunks, query, k, collection_name)


def list_collections() -> List[str]:
    """Return names of all ChromaDB collections."""
    client = get_client()
    return [c.name for c in client.list_collections()]


def get_collection_summary(
    collection_name: str | None = None,
    force_refresh: bool = False,
) -> Dict:
    """Return aggregate stats about the collection (for injecting into prompts).

    Fields: total_chunks, total_docs, date_min, date_max, latest_file.
    Cached for ``_SUMMARY_TTL_SECONDS`` to avoid scanning metadata on every request.
    """
    name = collection_name or CHROMA_COLLECTION
    now = time.time()
    cached = _summary_cache.get(name)
    if not force_refresh and cached and cached["expires_at"] > now:
        return cached["data"]

    collection = get_collection(name)
    count = collection.count()
    if count == 0:
        summary = {
            "total_chunks": 0,
            "total_docs": 0,
            "date_min": None,
            "date_max": None,
            "latest_file": None,
        }
    else:
        res = collection.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        dates = sorted(
            {m.get("pub_date") for m in metas if m.get("pub_date")}
        )
        # Map each source_file to its most recent pub_date
        file_dates: Dict[str, str] = {}
        for m in metas:
            f = m.get("source_file")
            d = m.get("pub_date", "") or ""
            if not f:
                continue
            if f not in file_dates or d > file_dates[f]:
                file_dates[f] = d
        latest_file = (
            max(file_dates.items(), key=lambda kv: kv[1])[0]
            if file_dates
            else None
        )
        summary = {
            "total_chunks": count,
            "total_docs": len(file_dates),
            "date_min": dates[0] if dates else None,
            "date_max": dates[-1] if dates else None,
            "latest_file": latest_file,
        }

    _summary_cache[name] = {
        "data": summary,
        "expires_at": now + _SUMMARY_TTL_SECONDS,
    }
    return summary


async def get_collection_summary_async(
    collection_name: str | None = None,
) -> Dict:
    """Async wrapper — runs the (potentially slow) scan in a thread pool."""
    return await asyncio.to_thread(get_collection_summary, collection_name)
