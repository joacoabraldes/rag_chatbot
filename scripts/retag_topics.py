#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-tag existing chunks in report_chunks using the controlled topic taxonomy.

Reads chunks straight from Postgres, re-runs the LLM topic extractor in
batches and writes ``topic_keys`` (text[]) and ``topic_tags`` (CSV) back.
Use ``--backfill-sections`` to also re-classify the section of any chunk
whose ``section`` is missing or 'resumen'.

Usage:
    python scripts/retag_topics.py
    python scripts/retag_topics.py --batch-size 200
    python scripts/retag_topics.py --backfill-sections
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import psycopg

from core.config import DATABASE_URL, OPENAI_MODEL_FAST
from core.llm import extract_topic_tags_batch
from core.sections import classify_section
from core.taxonomy import canonicalize_topics


def _connect() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no está configurada. Revisá .env.")
    return psycopg.connect(DATABASE_URL)


def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-etiquetar chunks existentes con taxonomía controlada"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Tamaño de lote para actualización (default: 200)",
    )
    parser.add_argument(
        "--model",
        default=OPENAI_MODEL_FAST,
        help="Modelo OpenAI para extracción de topics y clasificación de secciones",
    )
    parser.add_argument(
        "--backfill-sections",
        action="store_true",
        help="Re-clasificar también la sección de chunks sin sección (1 LLM call extra por chunk)",
    )
    args = parser.parse_args()

    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM report_chunks")
        count = cur.fetchone()[0]
        if count == 0:
            print("Tabla vacía. Nada para retaggear.")
            return

        print(f"report_chunks: {count} chunks")

        cur.execute(
            """
            SELECT chunk_id, text_chunk, block_header, section
            FROM report_chunks
            ORDER BY chunk_id
            """
        )
        rows = cur.fetchall()

    updated = 0
    with _connect() as conn, conn.cursor() as cur:
        for i, batch in _batched(rows, args.batch_size):
            chunk_ids = [r[0] for r in batch]
            chunk_docs = [r[1] for r in batch]
            chunk_headers = [r[2] or "" for r in batch]
            chunk_sections = [r[3] or "" for r in batch]

            tags = extract_topic_tags_batch(chunk_docs, model=args.model)

            for j, chunk_id in enumerate(chunk_ids):
                raw = tags[j] if j < len(tags) else ""
                topic_keys = canonicalize_topics(
                    [t.strip() for t in raw.split(",") if t.strip()]
                )
                topic_tags = ",".join(topic_keys)

                if args.backfill_sections and not chunk_sections[j]:
                    new_section = classify_section(
                        chunk_headers[j], chunk_docs[j], model=args.model
                    )
                    cur.execute(
                        """
                        UPDATE report_chunks
                        SET topic_keys = %s, topic_tags = %s, section = %s
                        WHERE chunk_id = %s
                        """,
                        (topic_keys, topic_tags, new_section, chunk_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE report_chunks
                        SET topic_keys = %s, topic_tags = %s
                        WHERE chunk_id = %s
                        """,
                        (topic_keys, topic_tags, chunk_id),
                    )

            conn.commit()
            updated += len(batch)
            print(f"  actualizado: {updated}/{count}")

    print("Retag finalizado.")


if __name__ == "__main__":
    main()
