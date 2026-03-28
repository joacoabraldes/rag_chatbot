#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF ingestion with semantic chunking, metadata extraction, and topic tagging.

Usage:
    python scripts/ingest.py --pdf-dir ./docs
    python scripts/ingest.py --pdf-dir ./docs --date 2026-02-13 --collection informes
    python scripts/ingest.py --pdf-dir ./docs --threshold 0.8
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Add project root to path so core modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pdfplumber

from core.chunking import semantic_chunk, split_into_sentences
from core.config import OPENAI_MODEL_FAST, SIMILARITY_THRESHOLD
from core.llm import extract_topic_tags_batch
from core.vectorstore import add_chunks, get_client


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text per page from a PDF file.

    Returns a list of dicts: [{"page_number": int, "text": str}, ...]
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i, "text": text.strip()})
    return pages


# ---------------------------------------------------------------------------
# Date inference from filename
# ---------------------------------------------------------------------------


def infer_date_from_filename(filename: str) -> str:
    """Try to extract an ISO date from the filename. Returns '' on failure."""
    # YYYYMMDD
    m = re.search(
        r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", filename
    )
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # YYYY-MM-DD or YYYY_MM_DD
    m = re.search(
        r"(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])", filename
    )
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


# ---------------------------------------------------------------------------
# Single-PDF processing
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: str,
    pub_date: str,
    threshold: float,
    model: str,
    skip_tags: bool = False,
) -> list[dict]:
    """Process one PDF into chunks with full metadata.

    Returns a list of chunk dicts ready for ``add_chunks()``.
    """
    filename = os.path.basename(pdf_path)
    pages = extract_pages(pdf_path)

    if not pages:
        print(f"  {filename}: no se pudo extraer texto")
        return []

    # Build sentences with page tracking
    all_sentences: list[str] = []
    sentence_pages: list[int] = []
    for page in pages:
        sentences = split_into_sentences(page["text"])
        for s in sentences:
            all_sentences.append(s)
            sentence_pages.append(page["page_number"])

    if not all_sentences:
        print(f"  {filename}: no se encontraron oraciones")
        return []

    # Semantic chunking
    chunk_groups = semantic_chunk(all_sentences, threshold)

    date_compact = pub_date.replace("-", "") if pub_date else "00000000"
    file_slug = re.sub(r"[^a-zA-Z0-9]", "_", os.path.splitext(filename)[0])[:40]
    chunk_objects: list[dict] = []
    chunks_for_tags: list[str] = []

    cursor = 0
    for ci, group in enumerate(chunk_groups):
        text = " ".join(group)
        page_num = sentence_pages[cursor] if cursor < len(sentence_pages) else 1
        cursor += len(group)

        chunk_id = f"{file_slug}_{date_compact}_p{page_num}_c{ci}"
        chunk_objects.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "source_file": filename,
                    "pub_date": pub_date,
                    "page_number": page_num,
                    "chunk_index": ci,
                    "topic_tags": "",
                },
            }
        )
        chunks_for_tags.append(text)

    # Extract topic tags via LLM
    if not skip_tags:
        print(f"  Extrayendo topic tags para {len(chunks_for_tags)} chunks...")
        tags = extract_topic_tags_batch(chunks_for_tags, model)
        for i, chunk in enumerate(chunk_objects):
            if i < len(tags):
                chunk["metadata"]["topic_tags"] = tags[i]

    print(
        f"  {filename}: {len(pages)} páginas -> {len(chunk_objects)} chunks"
    )
    return chunk_objects


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Indexar PDFs en ChromaDB con chunking semántico"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./docs",
        help="Directorio con archivos PDF (default: ./docs)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Nombre de la colección en ChromaDB (default: env CHROMA_COLLECTION)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral de similitud para chunking semántico (default: env SIMILARITY_THRESHOLD)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Fecha manual ISO (YYYY-MM-DD) para todos los PDFs del lote",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo OpenAI para extracción de topic tags",
    )
    parser.add_argument(
        "--skip-tags",
        action="store_true",
        help="Omitir extracción de topic tags (más rápido, sin llamadas a OpenAI)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borrar la colección existente antes de indexar",
    )
    args = parser.parse_args()

    threshold = args.threshold or SIMILARITY_THRESHOLD
    model = args.model or OPENAI_MODEL_FAST
    pdf_dir = Path(args.pdf_dir)

    if not pdf_dir.exists():
        print(f"Directorio no encontrado: {pdf_dir}")
        return

    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No se encontraron archivos PDF en {pdf_dir}")
        return

    from core.config import CHROMA_COLLECTION
    collection_name = args.collection or CHROMA_COLLECTION

    if args.reset:
        client = get_client()
        try:
            client.delete_collection(collection_name)
            print(f"Colección '{collection_name}' borrada.\n")
        except Exception:
            print(f"Colección '{collection_name}' no existía, continuando.\n")

    print(f"Encontrados {len(pdf_files)} PDF(s) en {pdf_dir}\n")
    if args.skip_tags:
        print("  (topic tags desactivados)\n")

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        pub_date = args.date or infer_date_from_filename(pdf_path.name)
        if not pub_date:
            print(
                f"  Advertencia: no se pudo inferir fecha de {pdf_path.name}. "
                "Usá --date para especificarla."
            )
        chunks = process_pdf(str(pdf_path), pub_date, threshold, model, skip_tags=args.skip_tags)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No se generaron chunks.")
        return

    print(f"\nIndexando {len(all_chunks)} chunks en ChromaDB...")
    add_chunks(all_chunks, collection_name)
    print("Listo.")


if __name__ == "__main__":
    main()
