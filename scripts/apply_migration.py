#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aplica un archivo .sql contra DATABASE_URL (leída desde .env).

Útil cuando no tenés ``psql`` instalado o cuando el DATABASE_URL no está
expuesta en el shell (.env la mantiene local al proyecto).

Uso:
    python scripts/apply_migration.py schema/02_fts.sql
    python scripts/apply_migration.py schema/02_fts.sql --dry-run

Sale con código 0 si todo OK, 1 si falló.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Aplicar un .sql contra DATABASE_URL")
    parser.add_argument("sql_file", help="Ruta al archivo .sql a ejecutar")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Imprime el SQL sin ejecutarlo",
    )
    args = parser.parse_args()

    sql_path = Path(args.sql_file)
    if not sql_path.is_absolute():
        sql_path = ROOT / sql_path
    if not sql_path.exists():
        print(f"❌ Archivo no encontrado: {sql_path}")
        return 1

    sql = sql_path.read_text(encoding="utf-8")

    if args.dry_run:
        print(f"--- {sql_path.name} ({len(sql)} chars) ---")
        print(sql)
        return 0

    if not DATABASE_URL:
        print("❌ DATABASE_URL no está definida en .env")
        return 1

    masked = DATABASE_URL
    if "@" in masked and ":" in masked:
        prefix, host = masked.split("@", 1)
        if ":" in prefix:
            user, _ = prefix.rsplit(":", 1)
            masked = f"{user}:***@{host}"
    print(f"Aplicando {sql_path.name} contra {masked}")

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, autocommit=True) as conn:
            # autocommit=True permite que cada ALTER/CREATE corra en su
            # propia transacción implícita, igual que psql.
            #
            # psycopg3 emite NOTICE/WARNING via callback en vez de la lista
            # ``conn.notices`` que existía en psycopg2 — registramos un
            # handler que los imprime apenas llegan.
            conn.add_notice_handler(
                lambda diag: print(f"  [{diag.severity}] {diag.message_primary}")
            )
            with conn.cursor() as cur:
                cur.execute(sql)
        print("✅ Migración aplicada.")
        return 0
    except psycopg.errors.DuplicateObject as e:
        # Las migraciones son idempotentes (IF NOT EXISTS) pero por las dudas.
        print(f"⚠️  Objeto ya existía: {e}")
        return 0
    except psycopg.Error as e:
        print(f"❌ Error de Postgres: {e.__class__.__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
