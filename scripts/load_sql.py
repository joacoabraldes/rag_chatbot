#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ejecuta archivos .sql contra Postgres / Supabase usando DATABASE_URL del .env.

Útil cuando PowerShell no expande variables o cuando el SQL Editor de
Supabase se cuelga con archivos grandes.

Uso:
    python scripts/load_sql.py schema/fx.sql
    python scripts/load_sql.py schema/forex.sql schema/fx.sql
    python scripts/load_sql.py --commit-every 200 schema/forex.sql
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from core.config import DATABASE_URL  # noqa: E402


def split_statements(sql: str) -> list[str]:
    """Divide el contenido del archivo en statements terminados en ';'.

    Maneja comillas simples, comillas dobles y bloques DO $$ ... $$ del
    plpgsql. No es un parser SQL completo — alcanza para los dumps tipo
    INSERT de Supabase / pg_dump.
    """
    statements: list[str] = []
    buf: list[str] = []
    in_single = False  # dentro de '...'
    in_double = False  # dentro de "..."
    in_dollar: str | None = None  # tag actual del bloque dollar-quoted ('$$', '$tag$')
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # Bloques dollar-quoted (DO $$ ... $$ y similares).
        if not in_single and not in_double:
            if ch == "$":
                # Captura el tag entre dos $: $$ o $tag$
                j = i + 1
                while j < n and (sql[j].isalnum() or sql[j] == "_"):
                    j += 1
                if j < n and sql[j] == "$":
                    tag = sql[i : j + 1]
                    if in_dollar is None:
                        in_dollar = tag
                        buf.append(tag)
                        i = j + 1
                        continue
                    elif in_dollar == tag:
                        in_dollar = None
                        buf.append(tag)
                        i = j + 1
                        continue

        if in_dollar:
            buf.append(ch)
            i += 1
            continue

        # String literals — escape de comillas duplicadas (SQL standard).
        if ch == "'" and not in_double:
            if in_single and nxt == "'":
                buf.append("''")
                i += 2
                continue
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        # Comentarios de línea -- ... \n
        if not in_single and not in_double and ch == "-" and nxt == "-":
            while i < n and sql[i] != "\n":
                buf.append(sql[i])
                i += 1
            continue

        # Fin de statement.
        if ch == ";" and not in_single and not in_double:
            buf.append(";")
            stmt = "".join(buf).strip()
            if stmt and stmt != ";":
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    return statements


def run_file(path: Path, commit_every: int) -> tuple[int, int]:
    """Ejecuta un .sql en chunks. Devuelve (statements, errores)."""
    print(f"\n📄 {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    sql = path.read_text(encoding="utf-8")
    statements = split_statements(sql)
    print(f"   {len(statements)} statements detectados")

    import psycopg

    n_ok = 0
    n_err = 0
    t0 = time.perf_counter()
    last_log = t0

    with psycopg.connect(DATABASE_URL, autocommit=False) as conn:
        with conn.cursor() as cur:
            for i, stmt in enumerate(statements, 1):
                try:
                    cur.execute(stmt)
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    snippet = stmt[:120].replace("\n", " ")
                    print(f"   ❌ stmt {i}: {e.__class__.__name__}: {e}")
                    print(f"      → {snippet}...")
                    conn.rollback()
                    continue

                if i % commit_every == 0:
                    conn.commit()
                    now = time.perf_counter()
                    if now - last_log > 1.0:
                        rate = i / (now - t0)
                        print(f"   ... {i}/{len(statements)} ({rate:.0f} stmt/s)")
                        last_log = now

            conn.commit()

    dt = time.perf_counter() - t0
    print(f"   ✅ {n_ok} ejecutados, {n_err} errores en {dt:.1f}s")
    return n_ok, n_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Cargar archivos .sql contra DATABASE_URL")
    parser.add_argument("files", nargs="+", help="Archivos .sql a ejecutar (en orden)")
    parser.add_argument(
        "--commit-every",
        type=int,
        default=200,
        help="Hacer commit cada N statements (default: 200)",
    )
    args = parser.parse_args()

    if not DATABASE_URL:
        print("❌ DATABASE_URL no está en .env")
        return 1

    total_ok = 0
    total_err = 0
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"❌ No existe: {f}")
            return 1
        ok, err = run_file(path, args.commit_every)
        total_ok += ok
        total_err += err

    print(f"\n═══════════════════════════════════════════")
    print(f"  TOTAL: {total_ok} OK, {total_err} errores")
    print(f"═══════════════════════════════════════════")
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
