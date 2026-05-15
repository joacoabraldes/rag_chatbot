# -*- coding: utf-8 -*-
"""Vector store backed by Postgres + pgvector (Supabase).

Public interface used by the rest of the app:
- ``add_chunks``
- ``query_chunks`` / ``query_chunks_async``
- ``list_collections``
- ``get_collection_summary`` / ``get_collection_summary_async``

Embeddings are computed locally with sentence-transformers (384 dims)
and stored alongside chunk metadata in ``report_chunks``.

Retrieval uses **hybrid search via Reciprocal Rank Fusion (RRF)**:
- Vec ranker: pgvector cosine (HNSW index on ``embedding``).
- BM25 ranker: Postgres FTS over ``text_tsv`` (GIN index, Spanish stemmer).
- Fusion: ``score = Σ 1 / (k + rank_i)`` with ``k=60`` (Cormack et al., 2009).
Each ranker independently pulls a fan-out of candidates; their ranks (not
scores) are fused. Chunks that show up in both rankers win; lexical-only
hits still get a fighting chance for entities/numbers the embedder misses.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from core.config import DATABASE_URL, SIMILARITY_THRESHOLD_RETRIEVAL
from core.db import get_pool
from core.embedder import embed_query, embed_texts


# ---------------------------------------------------------------------------
# RRF tuning knobs
# ---------------------------------------------------------------------------

# Standard suavizado constant from the original RRF paper. Aplasta el efecto
# de "estar primero" lo suficiente como para que un falso #1 no domine.
_RRF_K = 60

# Candidatos que pide cada ranker. Más fanout = mejor recall de fusión a
# costa de un poco de latencia. 50 alcanza para un corpus de miles.
_RRF_FANOUT = 50


# ---------------------------------------------------------------------------
# Where-dict translation (Chroma-style syntax -> SQL fragment + params)
#
# The filter_extractor still emits filters in the Chroma-style dict syntax
# ($and, $or, $in) so we keep translating it here. Lets us swap the
# extractor later without touching the storage layer.
# ---------------------------------------------------------------------------


def _is_topic_tag_clause(clause: dict) -> bool:
    """True if the clause targets one of the legacy topic_tag_N columns.

    Those columns don't exist in PG (we use a single ``topic_keys text[]``
    instead) so we collapse the OR-of-slots pattern that filter_extractor
    emits into an array-overlap query.
    """
    if not isinstance(clause, dict) or len(clause) != 1:
        return False
    return next(iter(clause)).startswith("topic_tag_")


def _quote_col(name: str) -> str:
    """Quote columns that collide with SQL reserved words."""
    return f'"{name}"' if name in {"date"} else name


def _walk(node: dict) -> Tuple[str, list]:
    """Recursive translation of one where-node to SQL."""
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
    """Hydrate a base SELECT row into the dict shape callers expect.

    Expects the first 12 columns produced by the RRF query (chunk_id …
    distance). The RRF-specific columns (rrf_score, vec_rank, bm25_rank)
    are attached by the caller.
    """
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
# Public API
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

    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        # Insert in batches so a single failure doesn't kill the whole load.
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            cur.executemany(_INSERT_SQL, rows[i : i + batch_size])
            conn.commit()


# ---------------------------------------------------------------------------
# RRF retrieval
# ---------------------------------------------------------------------------


# We materialize the query embedding + FTS tsquery in a CTE so both rankers
# can reference them without recomputing. Filters (``where_sql``) are
# applied to BOTH rankers — a chunk excluded by a section filter should
# disappear from FX and vec alike.
_RRF_SQL_TEMPLATE = """
    WITH q AS (
        SELECT %s::vector AS qv, plainto_tsquery('spanish', %s) AS qts
    ),
    -- vec_top: pull top-K from the HNSW index BEFORE assigning ranks.
    -- Doing the LIMIT inside a subquery lets the planner push it to the
    -- index. The outer row_number() then ranks only those K rows.
    vec_top AS (
        SELECT chunk_id,
               embedding <=> (SELECT qv FROM q) AS dist
        FROM report_chunks
        WHERE 1=1{where_sql}
        ORDER BY embedding <=> (SELECT qv FROM q)
        LIMIT %s
    ),
    vec AS (
        SELECT chunk_id,
               row_number() OVER (ORDER BY dist) AS rnk
        FROM vec_top
    ),
    bm25_top AS (
        SELECT chunk_id,
               ts_rank_cd(text_tsv, (SELECT qts FROM q)) AS score
        FROM report_chunks
        WHERE text_tsv @@ (SELECT qts FROM q){where_sql}
        ORDER BY ts_rank_cd(text_tsv, (SELECT qts FROM q)) DESC
        LIMIT %s
    ),
    bm25 AS (
        SELECT chunk_id,
               row_number() OVER (ORDER BY score DESC) AS rnk
        FROM bm25_top
    ),
    fused_ids AS (
        SELECT chunk_id FROM vec
        UNION
        SELECT chunk_id FROM bm25
    ),
    scored AS (
        SELECT f.chunk_id,
               v.rnk AS vec_rnk,
               b.rnk AS bm25_rnk,
               COALESCE(1.0 / (%s + v.rnk), 0)
                 + COALESCE(1.0 / (%s + b.rnk), 0) AS rrf_score
        FROM fused_ids f
        LEFT JOIN vec  v USING (chunk_id)
        LEFT JOIN bm25 b USING (chunk_id)
    )
    SELECT
        r.chunk_id,
        r.source_file,
        r.pub_date,
        r.page_number,
        r.block_index,
        r.block_header,
        r.section,
        r.chunk_index,
        r.text_chunk,
        r.topic_keys,
        r.topic_tags,
        (r.embedding <=> (SELECT qv FROM q)) AS distance,
        s.rrf_score,
        s.vec_rnk,
        s.bm25_rnk
    FROM scored s
    JOIN report_chunks r ON r.chunk_id = s.chunk_id
    ORDER BY s.rrf_score DESC
    LIMIT %s
"""


def query_chunks(
    query: str,
    k: int = 8,
    collection_name: Optional[str] = None,
    where: Optional[Dict] = None,
    similarity_threshold: float | None = None,
) -> List[Dict]:
    """Hybrid retrieval via Reciprocal Rank Fusion (vec + BM25).

    ``where`` accepts the same Chroma-style dict that filter_extractor
    produces; we translate it to a SQL WHERE on the fly. Pass
    ``similarity_threshold=0.0`` to disable the floor.

    The threshold is applied **only** to chunks that came from vec alone
    (no BM25 contribution). Chunks that match BM25 lexically pass through
    even if their cosine similarity is below threshold — that's the whole
    point of having two rankers.
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD_RETRIEVAL

    embedding = embed_query(query).tolist()
    where_sql, where_params = _build_where(where)
    sql = _RRF_SQL_TEMPLATE.format(where_sql=where_sql)

    # Order matters — see _RRF_SQL_TEMPLATE for the slot layout.
    params: list = (
        [embedding, query]            # CTE q
        + where_params + [_RRF_FANOUT]  # vec WHERE + LIMIT
        + where_params + [_RRF_FANOUT]  # bm25 WHERE + LIMIT
        + [_RRF_K, _RRF_K]            # rrf suavizado constants
        + [k]                          # final LIMIT
    )

    out: List[Dict] = []
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            base = row[:12]
            rrf_score, vec_rnk, bm25_rnk = row[12], row[13], row[14]
            chunk = _row_to_chunk(base)
            chunk["rrf_score"] = float(rrf_score) if rrf_score is not None else 0.0
            chunk["vec_rank"] = int(vec_rnk) if vec_rnk is not None else None
            chunk["bm25_rank"] = int(bm25_rnk) if bm25_rnk is not None else None

            has_bm25 = bm25_rnk is not None
            if (
                similarity_threshold > 0
                and not has_bm25
                and chunk["similarity"] < similarity_threshold
            ):
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
# Collection summary — cached for 5 min to avoid scanning metadata per request.
# ---------------------------------------------------------------------------


_SUMMARY_TTL_SECONDS = 300.0
_summary_cache: Dict[str, Dict] = {}


def get_collection_summary(
    collection_name: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict:
    """Return aggregate stats about the collection (for injecting into prompts).

    Fields: total_chunks, total_docs, date_min, date_max, latest_file, file_dates.
    """
    name = collection_name or "informes"
    now = time.time()
    cached = _summary_cache.get(name)
    if not force_refresh and cached and cached["expires_at"] > now:
        return cached["data"]

    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
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
                # Full mapping of filename -> latest pub_date. Used by the
                # filter extractor to resolve hints ("20260417") to real
                # filenames for a where-clause. Not included in the
                # system-prompt summary block.
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


def truncate_collection() -> int:
    """Delete all rows from report_chunks and return the count removed.

    Used by ingest --reset and tests that need a clean slate.
    """
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM report_chunks")
        n = cur.fetchone()[0]
        cur.execute("TRUNCATE report_chunks")
        conn.commit()
    _summary_cache.clear()
    return n
