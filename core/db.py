# -*- coding: utf-8 -*-
"""Shared Postgres connection pool.

A single sync ``ConnectionPool`` is created at startup and reused by the
vectorstore (chunk queries, ingest) and the SQL tools (fx/forex). Before
this, every ``/chat`` request opened 3-5 fresh ``psycopg.connect()`` calls;
under modest concurrency Supabase's connection limits started cutting us
off and the TCP+TLS+auth handshake added ~150-300 ms of dead time per
call.

The pool is pgvector-aware: ``configure`` runs ``register_vector`` on each
new connection, so callers can read/write ``vector(384)`` columns without
having to remember.

The CLI scripts (ingest, retag, eval) don't call ``init_pool`` explicitly —
``get_pool`` lazy-initializes on first use, so the same code paths work
both inside FastAPI's lifespan and from short-lived scripts.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from core.config import DATABASE_URL

log = logging.getLogger("rag.db")

_pool: Optional[ConnectionPool] = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def init_pool(min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Initialize the global pool. Idempotent — safe to call twice."""
    global _pool
    if _pool is not None:
        return _pool
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL no está configurada. Revisá .env."
        )
    _pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        timeout=30.0,
        kwargs={"connect_timeout": 10},
    )
    # Open eagerly so the first request doesn't pay the handshake.
    _pool.open(wait=True)
    log.info("Postgres pool inicializado (min=%d, max=%d)", min_size, max_size)
    return _pool


def get_pool() -> ConnectionPool:
    """Return the initialized pool, lazy-initializing if needed."""
    if _pool is None:
        return init_pool()
    return _pool


def close_pool() -> None:
    """Tear down the pool on shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        log.info("Postgres pool cerrado")
