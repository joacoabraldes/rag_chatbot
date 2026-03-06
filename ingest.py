#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, pathlib, uuid, math, re, datetime
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

try:
    from langchain_experimental.text_splitter import SemanticChunker
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

DB_DIR_DEFAULT = "./chroma_db"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

DATE_PATTERNS = [
    # YYYYMMDD
    re.compile(r"(?P<y>20\d{2}|19\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])"),
    # YYYY-MM-DD or YYYY_MM_DD or YYYY.MM.DD
    re.compile(r"(?P<y>20\d{2}|19\d{2})[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<d>0[1-9]|[12]\d|3[01])"),
    # DD-MM-YYYY etc.
    re.compile(r"(?P<d>0[1-9]|[12]\d|3[01])[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<y>20\d{2}|19\d{2})"),
]

def to_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"

def try_parse_date_from_string(s: str) -> Optional[str]:
    if not s: return None
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if m:
            y = int(m.group("y"))
            mth = int(m.group("m"))
            d = int(m.group("d"))
            try:
                datetime.date(y, mth, d)
                return to_iso(y, mth, d)
            except ValueError:
                continue
    return None

def infer_date_iso(meta: Dict[str, Any], date_field: Optional[str]) -> Optional[str]:
    # 1) si viene ya una fecha usable
    if date_field and isinstance(meta.get(date_field), str):
        # aceptamos YYYY-MM-DD (flexible)
        s = meta.get(date_field)
        iso = try_parse_date_from_string(s) or s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else None
        if iso: return iso

    # 2) intentar desde varias claves candidatas
    for k in ["rel_path", "source", "id", "title", "name", "filename"]:
        v = meta.get(k)
        if isinstance(v, str):
            iso = try_parse_date_from_string(v)
            if iso: return iso
    return None

def read_txt_dir(path: str) -> List[Document]:
    docs = []
    p = pathlib.Path(path)
    for f in sorted(p.glob("**/*.txt")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta = {"source": str(f), "rel_path": f.name, "chunk_id": None}
        di = infer_date_iso(meta, None)
        if di: meta["date_iso"] = di
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def read_pdf_dir(path: str) -> List[Document]:
    """Lee todos los .pdf de un directorio, extrayendo texto por página."""
    if not HAS_PYPDF:
        raise ImportError("Se requiere 'pypdf' para leer PDFs. Instalá con: pip install pypdf")
    docs = []
    p = pathlib.Path(path)
    for f in sorted(p.glob("**/*.pdf")):
        try:
            reader = pypdf.PdfReader(str(f))
        except Exception as e:
            print(f"  ⚠️ No se pudo leer {f.name}: {e}")
            continue
        full_text_parts = []
        for pi, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                full_text_parts.append(page_text)
        full_text = "\n\n".join(full_text_parts)
        if not full_text.strip():
            print(f"  ⚠️ {f.name}: no se pudo extraer texto (puede ser imagen/scan)")
            continue
        meta = {
            "source": str(f),
            "rel_path": f.name,
            "chunk_id": None,
            "total_pages": len(reader.pages),
        }
        di = infer_date_iso(meta, None)
        if di:
            meta["date_iso"] = di
        docs.append(Document(page_content=full_text, metadata=meta))
        print(f"  ✅ {f.name}: {len(reader.pages)} págs, {len(full_text)} chars")
    return docs

def read_jsonl(path: str, date_field: Optional[str]) -> List[Document]:
    """
    Soporta:
    A) {"text": "...", "metadata": {...}}
    B) {"text": "...", <metadatos top-level: id, source, rel_path, ...>}
    """
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            md = obj.get("metadata")
            if isinstance(md, dict):              # Formato A
                meta = md
            else:                                  # Formato B
                meta = {k: v for k, v in obj.items() if k != "text"}

            # fallbacks útiles
            if "rel_path" not in meta and "source" in meta:
                try:
                    meta["rel_path"] = os.path.basename(str(meta["source"]))
                except Exception:
                    pass
            if "source" not in meta and "rel_path" in meta:
                meta["source"] = meta["rel_path"]

            # fecha inferida
            di = infer_date_iso(meta, date_field)
            if di:
                meta["date_iso"] = di

            docs.append(Document(page_content=text, metadata=meta))
    return docs

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into smaller chunks, preserving and enriching metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunked: List[Document] = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        full_text = doc.page_content
        cursor = 0
        for ci, piece in enumerate(splits):
            start = full_text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            cursor = max(cursor, end - chunk_overlap)

            md = dict(doc.metadata)
            md["chunk_id"] = ci
            md["chunk_start"] = start
            md["chunk_end"] = end
            md["chunk_total"] = len(splits)
            chunked.append(Document(page_content=piece, metadata=md))
    return chunked

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Directorio con .txt (cada archivo = un chunk)")
    g.add_argument("--pdf", help="Directorio con .pdf")
    g.add_argument("--jsonl", help="Archivo .jsonl (texto + metadatos)")
    ap.add_argument("--persist", default=os.environ.get("CHROMA_DB_DIR", DB_DIR_DEFAULT),
                    help="Directorio de persistencia de Chroma")
    ap.add_argument("--collection", default="docs", help="Nombre de la colección")
    ap.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL",
                    "BAAI/bge-m3"))
    ap.add_argument("--batch-size", type=int, default=1000, help="Lote (< 5461)")
    ap.add_argument("--date-field", default=None,
                    help="Nombre de campo en metadata que contiene la fecha (si existe). Si no, se infiere del filename.")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help=f"Tamaño de cada chunk en caracteres (default: {DEFAULT_CHUNK_SIZE})")
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help=f"Solapamiento entre chunks consecutivos (default: {DEFAULT_CHUNK_OVERLAP})")
    ap.add_argument("--no-chunk", action="store_true",
                    help="Desactivar chunking (cada archivo/JSONL entry = un documento)")
    ap.add_argument("--semantic-chunk", action="store_true",
                    help="Usar chunking semántico en lugar de tamaño fijo")
    ap.add_argument("--semantic-threshold", type=int, default=75,
                    help="Percentil de umbral para chunking semántico (default: 75)")
    args = ap.parse_args()

    # 1) Carga
    if args.input:
        docs = read_txt_dir(args.input)
    elif args.pdf:
        docs = read_pdf_dir(args.pdf)
    else:
        docs = read_jsonl(args.jsonl, args.date_field)
    if not docs:
        print("No se encontraron documentos para indexar.")
        return

    # 1b) Chunking
    if not args.no_chunk:
        pre_count = len(docs)
        if args.semantic_chunk:
            if not HAS_SEMANTIC:
                print("⚠️ langchain-experimental no instalado. Instalá con: pip install langchain-experimental")
                print("   Usando chunking por tamaño fijo como fallback.")
                docs = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
            else:
                hf_chunker = HuggingFaceEmbeddings(
                    model_name=args.model,
                    encode_kwargs={"normalize_embeddings": True},
                )
                semantic_splitter = SemanticChunker(
                    hf_chunker,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=args.semantic_threshold,
                )
                chunked = []
                for doc in docs:
                    splits = semantic_splitter.split_text(doc.page_content)
                    for ci, piece in enumerate(splits):
                        md = dict(doc.metadata)
                        md["chunk_id"] = ci
                        md["chunk_total"] = len(splits)
                        chunked.append(Document(page_content=piece, metadata=md))
                docs = chunked
                print(f"Semantic chunking: {pre_count} documentos → {len(docs)} chunks "
                      f"(threshold={args.semantic_threshold}%)")
        else:
            docs = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
            print(f"Chunking: {pre_count} documentos → {len(docs)} chunks "
                  f"(size={args.chunk_size}, overlap={args.chunk_overlap})")
    else:
        print(f"Chunking desactivado. {len(docs)} documentos sin dividir.")

    # 2) Embeddings locales
    hf = HuggingFaceEmbeddings(model_name=args.model, encode_kwargs={"normalize_embeddings": True})

    # 3) Cliente persistente
    client = chromadb.PersistentClient(path=args.persist)

    # 4) VectorStore
    vectordb = Chroma(client=client, collection_name=args.collection, embedding_function=hf)

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    ids = [str(uuid.uuid4()) for _ in docs]

    total = len(texts)
    bs = min(args.batch_size, 5400)
    nbatches = math.ceil(total / bs)
    print(f"Iniciando ingesta: {total} documentos en {nbatches} lotes de hasta {bs}…")

    added = 0
    for bi, (t_batch, m_batch, i_batch) in enumerate(
        zip(batched(texts, bs), batched(metas, bs), batched(ids, bs)), start=1
    ):
        vectordb.add_texts(texts=t_batch, metadatas=m_batch, ids=i_batch)
        added += len(t_batch)
        print(f"  Lote {bi}/{nbatches} OK — acumulado: {added}")

    print(f"Ingesta completa. {added} documentos agregados en '{args.persist}' (colección '{args.collection}').")

if __name__ == "__main__":
    main()
