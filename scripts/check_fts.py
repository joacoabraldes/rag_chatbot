#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifica que la migración FTS quedó aplicada en report_chunks.

Chequea: la columna ``text_tsv`` existe, el índice GIN existe, y cuántas
filas tienen tsvector poblado. Si todo está OK, sale con código 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import psycopg

from core.config import DATABASE_URL


def main() -> int:
    if not DATABASE_URL:
        print("❌ DATABASE_URL no está definida en .env")
        return 1

    with psycopg.connect(DATABASE_URL, connect_timeout=15) as conn, conn.cursor() as cur:
        # 1. Columna text_tsv presente.
        cur.execute(
            """
            SELECT column_name, data_type, is_generated, generation_expression
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = 'report_chunks'
              AND column_name  = 'text_tsv'
            """
        )
        col = cur.fetchone()
        if not col:
            print("❌ Columna text_tsv no existe. Aplicá schema/02_fts.sql primero.")
            return 1
        print(f"✅ Columna text_tsv: tipo={col[1]}, generated={col[2]}")

        # 2. Índice GIN.
        cur.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename  = 'report_chunks'
              AND indexname  = 'rc_text_tsv_gin_idx'
            """
        )
        idx = cur.fetchone()
        if not idx:
            print("❌ Índice rc_text_tsv_gin_idx no existe.")
            return 1
        print(f"✅ Índice: {idx[0]}")

        # 3. Cobertura.
        cur.execute("SELECT count(*) FROM report_chunks")
        total = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM report_chunks WHERE text_tsv IS NOT NULL")
        populated = cur.fetchone()[0]
        print(f"✅ Chunks: {total} totales, {populated} con text_tsv poblado")

        # 4. Smoke test de FTS — debería tocar el índice GIN.
        cur.execute(
            """
            SELECT count(*) FROM report_chunks
            WHERE text_tsv @@ plainto_tsquery('spanish', 'dolar')
            """
        )
        n = cur.fetchone()[0]
        print(f"✅ Smoke test FTS: 'dolar' matchea {n} chunks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
