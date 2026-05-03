# -*- coding: utf-8 -*-
"""Postgres + pgvector backend for the chunk store.

Drop-in replacement for ``core.vectorstore``. Same public interface:
- ``add_chunks``
- ``query_chunks`` / ``query_chunks_async``
- ``list_collections``
- ``get_collection_summary`` / ``get_collection_summary_async``

Switched on via ``VECTOR_BACKEND=postgres`` in the environment.
``core.vectorstore`` re-exports from this module when that flag is set,
so callers don't need to know which backend is active.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import psycopg
from pgvector.psycopg import register_vector

from core.config import DATABASE_URL, SIMILARITY_THRESHOLD_RETRIEVAL
from core.embedder import embed_query, embed_texts


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _connect() -> psycopg.Connection:
    """Open a fresh connection. We open per-call instead of pooling because
    Supabase's session pooler already multiplexes connections server-side
    and the request rate is low.
    """
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurada. Revisá .env."
        )
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Where-dict translation (Chroma syntax -> SQL fragment + params)
# ---------------------------------------------------------------------------


def _is_topic_tag_clause(clause: dict) -> bool:
    """True if the clause targets one of the legacy topic_tag_N columns.

    Those columns don't exist in PG (we use a single ``topic_keys text[]``
    instead) so we collapse the OR-of-slots pattern that filter_extractor
    emits for Chroma into an array-overlap query.
    """
    if not isinstance(clause, dict) or len(clause) != 1:
        return False
    return next(iter(clause)).startswith("topic_tag_")


def _quote_col(name: str) -> str:
    """Quote columns that collide with SQL reserved words."""
    return f'"{name}"' if name in {"date"} else name


def _walk(node: dict) -> Tuple[str, list]:
    """Recursive translation of one Chroma where-node to SQL."""
    if "$and" in node:
        results = [_walk(c) for c in node["$and"]]
        sqls = [r[0] for r in results if r[0]]
        params = [p for r in results for p in r[1]]
        return (f"({' AND '.join(sqls)})", params) if sqls else ("", [])

    if "$or" in node:
        children = node["$or"]
        # The filter_extractor emits OR over topic_tag_1..4 to simulate an
        # ARRAY containment query. In Postgres we have the real thing.
        if children and all(_is_topic_tag_clause(c) for c in children):
            for c in children:
                key = next(iter(c))
                clause = c[key]
                if isinstance(clause, dict) and "$in" in clause:
                    return ("topic_keys && %s::text[]", [list(clause["$in"])])
        results = [_walk(c) for c in children]
        sqls = [r[0] for r in results if r[0]]
        params = [p for r in results for p in r[1]]
        return (f"({' OR '.join(sqls)})", params) if sqls else ("", [])

    # Single column clause: {"col": {"$in": [...]}} or {"col": value}.
    for key, val in node.items():
        col = _quote_col(key)
        if isinstance(val, dict):
            if "$in" in val:
                return (f"{col} = ANY(%s)", [list(val["$in"])])
            if "$eq" in val:
                return (f"{col} = %s", [val["$eq"]])
        else:
            return (f"{col} = %s", [val])
    return ("", [])


def _build_where(where: Optional[Dict]) -> Tuple[str, list]:
    if not where:
        return "", []
    sql, params = _walk(where)
    return (f" AND {sql}", params) if sql else ("", [])


# ---------------------------------------------------------------------------
# Metadata coercion
# ---------------------------------------------------------------------------


def _meta_to_row(chunk: Dict) -> Tuple:
    """Flatten chunk metadata into the column tuple for INSERT."""
    md = chunk.get("metadata") or {}
    topic_keys_raw = md.get("topic_keys") or ""
    if isinstance(topic_keys_raw, list):
        topic_keys = [str(k).strip() for k in topic_keys_raw if str(k).strip()]
    else:
        topic_keys = [k.strip() for k in str(topic_keys_raw).split(",") if k.strip()]
    section = (md.get("section") or "resumen").strip() or "resumen"
    return (
        chunk["chunk_id"],
        md.get("source_file") or "",
        md.get("pub_date") or None,
        md.get("page_number"),
        md.get("block_index"),
        md.get("block_header") or None,
        section,
        md.get("chunk_index"),
        chunk["text"],
        topic_keys,
        md.get("topic_tags") or "",
    )


def _row_to_chunk(row: Tuple) -> Dict:
    """Hydrate a SELECT row into the dict shape callers expect."""
    (
        chunk_id,
        source_file,
        pub_date,
        page_number,
        block_index,
        block_header,
        section,
        chunk_index,
        text_chunk,
        topic_keys,
        topic_tags,
        distance,
    ) = row
    return {
        "id": chunk_id,
        "text": text_chunk,
        "metadata": {
            "source_file": source_file or "",
            # downstream code does string compares on pub_date — keep it ISO.
            "pub_date": pub_date.isoformat() if pub_date else "",
            "page_number": page_number,
            "block_index": block_index,
            "block_header": block_header or "",
            "section": section or "",
            "chunk_index": chunk_index,
            "topic_keys": ",".join(topic_keys or []),
            "topic_tags": topic_tags or "",
        },
        "distance": float(distance),
        "similarity": 1.0 - float(distance),
    }


# ---------------------------------------------------------------------------
# Public API — mirrors core/vectorstore.py
# ---------------------------------------------------------------------------


_INSERT_SQL = """
    INSERT INTO report_chunks (
        chunk_id, source_file, pub_date, page_number,
        block_index, block_header, section, chunk_index,
        text_chunk, topic_keys, topic_tags, embedding
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (chunk_id) DO UPDATE SET
        source_file  = EXCLUDED.source_file,
        pub_date     = EXCLUDED.pub_date,
        page_number  = EXCLUDED.page_number,
        block_index  = EXCLUDED.block_index,
        block_header = EXCLUDED.block_header,
        section      = EXCLUDED.section,
        chunk_index  = EXCLUDED.chunk_index,
        text_chunk   = EXCLUDED.text_chunk,
        topic_keys   = EXCLUDED.topic_keys,
        topic_tags   = EXCLUDED.topic_tags,
        embedding    = EXCLUDED.embedding
"""


def add_chunks(chunks: List[Dict], collection_name: Optional[str] = None) -> None:
    """Compute embeddings locally and upsert into report_chunks."""
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    rows = []
    for chunk, emb in zip(chunks, embeddings):
        base = _meta_to_row(chunk)
        rows.append(base + (emb.tolist() if hasattr(emb, "tolist") else list(emb),))

    with _connect() as conn, conn.cursor() as cur:
        # Insert in batches so a single failure doesn't kill the whole load.
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            cur.executemany(_INSERT_SQL, rows[i : i + batch_size])
            conn.commit()


def query_chunks(
    query: str,
    k: int = 8,
    collection_name: Optional[str] = None,
    where: Optional[Dict] = None,
    similarity_threshold: float | None = None,
) -> List[Dict]:
    """Hybrid retrieval: top-K + similarity floor.

    ``where`` accepts the same Chroma-style dict that filter_extractor
    produces; we translate it to a SQL WHERE on the fly. Pass
    ``similarity_threshold=0.0`` to disable the floor.
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD_RETRIEVAL

    embedding = embed_query(query).tolist()
    where_sql, where_params = _build_where(where)

    sql = f"""
        SELECT
            chunk_id,
            source_file,
            pub_date,
            page_number,
            block_index,
            block_header,
            section,
            chunk_index,
            text_chunk,
            topic_keys,
            topic_tags,
            embedding <=> %s::vector AS distance
        FROM report_chunks
        WHERE 1=1{where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [embedding] + where_params + [embedding, k]

    out: List[Dict] = []
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            chunk = _row_to_chunk(row)
            if similarity_threshold > 0 and chunk["similarity"] < similarity_threshold:
                continue
            out.append(chunk)
    return out


async def query_chunks_async(
    query: str,
    k: int = 8,
    collection_name: Optional[str] = None,
    where: Optional[Dict] = None,
    similarity_threshold: float | None = None,
) -> List[Dict]:
    """Async wrapper — runs query_chunks in a thread pool."""
    return await asyncio.to_thread(
        query_chunks, query, k, collection_name, where, similarity_threshold
    )


def list_collections() -> List[str]:
    """Postgres has a single fixed 'collection' (one table)."""
    return ["informes"]


# ---------------------------------------------------------------------------
# Collection summary — same shape as the Chroma version, cached for 5 min.
# ---------------------------------------------------------------------------


_SUMMARY_TTL_SECONDS = 300.0
_summary_cache: Dict[str, Dict] = {}


def get_collection_summary(
    collection_name: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict:
    """Same return shape as the Chroma version: total_chunks, total_docs,
    date_min, date_max, latest_file, file_dates."""
    name = collection_name or "informes"
    now = time.time()
    cached = _summary_cache.get(name)
    if not force_refresh and cached and cached["expires_at"] > now:
        return cached["data"]

    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM report_chunks")
        total_chunks = cur.fetchone()[0]

        if total_chunks == 0:
            summary = {
                "total_chunks": 0,
                "total_docs": 0,
                "date_min": None,
                "date_max": None,
                "latest_file": None,
                "file_dates": {},
            }
        else:
            cur.execute(
                """
                SELECT source_file, MAX(pub_date) AS latest
                FROM report_chunks
                WHERE source_file <> ''
                GROUP BY source_file
                """
            )
            file_dates: Dict[str, str] = {}
            for source_file, latest in cur.fetchall():
                file_dates[source_file] = latest.isoformat() if latest else ""

            cur.execute(
                "SELECT MIN(pub_date), MAX(pub_date) FROM report_chunks WHERE pub_date IS NOT NULL"
            )
            date_min, date_max = cur.fetchone()

            latest_file = (
                max(file_dates.items(), key=lambda kv: kv[1])[0]
                if file_dates
                else None
            )

            summary = {
                "total_chunks": total_chunks,
                "total_docs": len(file_dates),
                "date_min": date_min.isoformat() if date_min else None,
                "date_max": date_max.isoformat() if date_max else None,
                "latest_file": latest_file,
                "file_dates": file_dates,
            }

    _summary_cache[name] = {
        "data": summary,
        "expires_at": now + _SUMMARY_TTL_SECONDS,
    }
    return summary


async def get_collection_summary_async(
    collection_name: Optional[str] = None,
) -> Dict:
    return await asyncio.to_thread(get_collection_summary, collection_name)
