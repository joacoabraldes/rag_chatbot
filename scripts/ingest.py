#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF ingestion with section-aware semantic chunking.

Pipeline per PDF:
1. Extract text per page with pdfplumber.
2. Detect ALL-CAPS section headers (lines like "MAS BAJAS DE TASAS:").
3. Split the document into header-blocks. The first block (before any
   header) is implicitly the "resumen" of the day.
4. Classify each block into a canonical section via LLM.
5. Within each block, run token-aware semantic chunking; every chunk
   inherits the block's ``section``.
6. Tag chunks with topic_keys (10-key controlled taxonomy).
7. Upsert into Postgres (report_chunks).

Usage:
    python scripts/ingest.py --pdf-dir ./docs
    python scripts/ingest.py --pdf-dir ./docs --date 2026-02-13
    python scripts/ingest.py --pdf-dir ./docs --skip-tags --skip-sections
    python scripts/ingest.py --pdf-dir ./docs --reset
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
from core.sections import classify_section
from core.taxonomy import build_topic_metadata
from core.vectorstore import add_chunks, truncate_collection


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
# Section header detection
# ---------------------------------------------------------------------------

# Headers in "Las Claves del día" look like:
#   "MAS VOLUMEN EN DIVISA Y BAJAS DEL MLC:"
#   "LLAMADO A LICITACIÓN DE PESOS Y DÓLARES:"
# i.e. an ALL-CAPS line ending with ':' that's >= 15 chars long.
# Allowed chars cover Spanish accents, digits and basic punctuation.
_HEADER_RE = re.compile(
    r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 ,\.\-/áéíóúñ\(\)]{14,150}):\s*",
    re.MULTILINE,
)


def split_into_blocks(full_text: str) -> list[dict]:
    """Split document text into header-delimited blocks.

    Returns a list of dicts: [{"header": str, "body": str}, ...]
    The first block (before any header) carries header="" and represents
    the opening narrative of the report.
    """
    matches = list(_HEADER_RE.finditer(full_text))
    blocks: list[dict] = []

    if not matches:
        return [{"header": "", "body": full_text.strip()}]

    first_start = matches[0].start()
    if first_start > 0:
        head_body = full_text[:first_start].strip()
        if head_body:
            blocks.append({"header": "", "body": head_body})

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[body_start:body_end].strip()
        if body:
            blocks.append({"header": header, "body": body})

    return blocks


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
    skip_sections: bool = False,
) -> list[dict]:
    """Process one PDF into chunks with full metadata.

    Returns a list of chunk dicts ready for ``add_chunks()``.
    """
    filename = os.path.basename(pdf_path)
    pages = extract_pages(pdf_path)

    if not pages:
        print(f"  {filename}: no se pudo extraer texto")
        return []

    # Build a single text stream + per-line page mapping. We need page
    # numbers to survive section splitting and chunking.
    line_pages: list[tuple[str, int]] = []
    for page in pages:
        for line in page["text"].splitlines():
            line_pages.append((line, page["page_number"]))

    full_text = "\n".join(line for line, _ in line_pages)
    if not full_text.strip():
        return []

    # Build a char-offset → page mapping so we can recover the page of any
    # substring after section splitting.
    char_to_page: list[int] = []
    for line, page_num in line_pages:
        char_to_page.extend([page_num] * len(line))
        char_to_page.append(page_num)  # for the join newline
    if char_to_page:
        char_to_page = char_to_page[:-1]  # drop trailing join entry

    def page_at(offset: int) -> int:
        if not char_to_page:
            return 1
        return char_to_page[min(offset, len(char_to_page) - 1)]

    blocks = split_into_blocks(full_text)
    print(f"  {filename}: {len(pages)} páginas, {len(blocks)} bloques detectados")

    date_compact = pub_date.replace("-", "") if pub_date else "00000000"
    file_slug = re.sub(r"[^a-zA-Z0-9]", "_", os.path.splitext(filename)[0])[:40]

    chunk_objects: list[dict] = []
    chunks_for_tags: list[str] = []
    block_running_offset = 0

    for bi, block in enumerate(blocks):
        header = block["header"]
        body = block["body"]

        # Re-find this block's offset in full_text to get page numbers.
        # Search forward from the previous block's offset so duplicate
        # headers don't get mismatched.
        block_offset = full_text.find(body, block_running_offset)
        if block_offset == -1:
            block_offset = block_running_offset
        block_running_offset = block_offset + len(body)

        sentences = split_into_sentences(body)
        if not sentences:
            continue

        # Block-level section fallback. The opening block (bi==0) has no
        # header and tends to cover many topics (internacional + commodities
        # + Argentina) — block-level classification collapses all of that
        # into "resumen". For that block we override per chunk below.
        if skip_sections:
            block_section = "resumen" if bi == 0 else ""
        else:
            block_section = classify_section(header, body, model=model)

        chunk_groups = semantic_chunk(sentences, threshold=threshold)

        # Map each chunk back to a page using the offset of its first sentence.
        sentence_offset = block_offset
        for ci, group in enumerate(chunk_groups):
            chunk_text = " ".join(group)
            first_sentence = group[0]
            local = full_text.find(first_sentence, sentence_offset)
            if local == -1:
                local = sentence_offset
            page_num = page_at(local)
            sentence_offset = local + len(chunk_text)

            # Per-chunk classification only for the opening block — surfaces
            # commodities / equity_arg / bonos that get absorbed by 'resumen'
            # when classifying the whole intro at once.
            if bi == 0 and not skip_sections:
                chunk_section = classify_section(
                    header="", body_preview=chunk_text, model=model
                )
            else:
                chunk_section = block_section

            chunk_id = f"{file_slug}_{date_compact}_b{bi}_p{page_num}_c{ci}"
            chunk_objects.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source_file": filename,
                        "pub_date": pub_date,
                        "page_number": page_num,
                        "block_index": bi,
                        "block_header": header[:200],
                        "section": chunk_section,
                        "chunk_index": ci,
                        "topic_tags": "",
                        "topic_keys": "",
                        "topic_tag_1": "",
                        "topic_tag_2": "",
                        "topic_tag_3": "",
                        "topic_tag_4": "",
                    },
                }
            )
            chunks_for_tags.append(chunk_text)

    # Extract topic tags via LLM
    if not skip_tags:
        print(f"  Extrayendo topic tags para {len(chunks_for_tags)} chunks...")
        tags = extract_topic_tags_batch(chunks_for_tags, model)
        for i, chunk in enumerate(chunk_objects):
            if i < len(tags):
                topic_keys = [t.strip() for t in tags[i].split(",") if t.strip()]
                chunk["metadata"].update(build_topic_metadata(topic_keys))

    print(
        f"  {filename}: {len(pages)} páginas -> {len(chunk_objects)} chunks "
        f"(secciones: {sorted({c['metadata']['section'] for c in chunk_objects})})"
    )
    return chunk_objects


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Indexar PDFs en Postgres + pgvector con chunking semántico"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./docs",
        help="Directorio con archivos PDF (default: ./docs)",
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
        help="Modelo OpenAI para extracción de topic tags y clasificación de secciones",
    )
    parser.add_argument(
        "--skip-tags",
        action="store_true",
        help="Omitir extracción de topic tags (más rápido, sin llamadas a OpenAI)",
    )
    parser.add_argument(
        "--skip-sections",
        action="store_true",
        help="Omitir clasificación de secciones (todos los chunks quedan como 'resumen')",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="TRUNCATE report_chunks antes de indexar",
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

    if args.reset:
        n = truncate_collection()
        print(f"report_chunks vaciada ({n} filas eliminadas).\n")

    print(f"Encontrados {len(pdf_files)} PDF(s) en {pdf_dir}\n")
    if args.skip_tags:
        print("  (topic tags desactivados)")
    if args.skip_sections:
        print("  (clasificación de secciones desactivada)")
    if args.skip_tags or args.skip_sections:
        print()

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        pub_date = args.date or infer_date_from_filename(pdf_path.name)
        if not pub_date:
            print(
                f"  Advertencia: no se pudo inferir fecha de {pdf_path.name}. "
                "Usá --date para especificarla."
            )
        chunks = process_pdf(
            str(pdf_path),
            pub_date,
            threshold,
            model,
            skip_tags=args.skip_tags,
            skip_sections=args.skip_sections,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No se generaron chunks.")
        return

    print(f"\nIndexando {len(all_chunks)} chunks en Postgres...")
    add_chunks(all_chunks)
    print("Listo.")


if __name__ == "__main__":
    main()
