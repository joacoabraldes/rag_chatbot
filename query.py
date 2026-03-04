# -*- coding: utf-8 -*-
import os, datetime, argparse
from typing import List, Dict, Tuple, Any, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from rag_utils import (
    load_prompt, recency_score, sim_from_distance, combined_score,
    parse_date_iso, expand_queries_simple, expand_queries_llm,
    short_preview,
)

# ========= Main =========

def main():
    p = argparse.ArgumentParser(description="Consulta RAG (CLI).")
    p.add_argument("query", help="Pregunta del usuario")
    p.add_argument("--persist", default="./chroma_db", help="Ruta de la base de Chroma")
    p.add_argument("--collections", nargs="+", help="Colecciones a usar")
    p.add_argument("--k", type=int, default=10, help="Cantidad total de documentos")
    p.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--recency-weight", type=float, default=0.35)
    p.add_argument("--half-life", type=int, default=30)
    p.add_argument("--min-date", default="")
    p.add_argument("--fetch-factor", type=float, default=2.0)
    p.add_argument("--use-query-expansion", action="store_true")
    p.add_argument("--llm-expansion", action="store_true",
                   help="Use LLM to generate query expansions (requires OpenAI key)")
    p.add_argument("--prompt", default="./prompt_template.txt")
    p.add_argument("--list", action="store_true", help="Listar colecciones disponibles")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=args.persist)
    all_cols = [c.name for c in client.list_collections()]

    if args.list:
        print("Colecciones disponibles:", ", ".join(all_cols) or "(ninguna)")
        return

    cols = args.collections or all_cols
    if not cols:
        print("No hay colecciones.")
        return

    embedder = HuggingFaceEmbeddings(model_name=args.embed_model, encode_kwargs={"normalize_embeddings": True})
    vectordbs = {col: Chroma(client=client, collection_name=col, embedding_function=embedder) for col in cols}

    # Recuperación
    queries = [args.query]
    if args.llm_expansion:
        queries = expand_queries_llm(args.query, openai_model=args.openai_model)
    elif args.use_query_expansion:
        queries = expand_queries_simple(args.query)

    ks = [max(1, args.k // len(cols)) for _ in cols]
    fetch_k_each = [max(5, int(k * args.fetch_factor)) for k in ks]
    results = []
    today = datetime.date.today()
    min_date = None
    if args.min_date:
        try:
            y, m, d = map(int, args.min_date[:10].split("-"))
            min_date = datetime.date(y, m, d)
        except Exception:
            pass

    for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
        candidates = []
        for qexp in queries:
            candidates += vdb.similarity_search_with_score(qexp, k=k_each)
        seen = set()
        uniq = []
        for doc, dist in candidates:
            key = (doc.page_content[:120], doc.metadata.get("rel_path"), doc.metadata.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                uniq.append((doc, dist))
        for (doc, dist) in uniq:
            md = dict(doc.metadata or {})
            md["_collection"] = col
            doc.metadata = md
            d_iso = parse_date_iso(md)
            if min_date and d_iso and d_iso < min_date:
                continue
            cscore = combined_score(dist, d_iso, today, args.recency_weight, args.half_life)
            results.append((doc, dist, cscore))

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:args.k]

    if not top:
        print("⚠️ No se encontraron resultados.")
        return

    # Contexto
    parts = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        name = os.path.basename(md.get("rel_path") or md.get("source") or "desconocido")
        d = md.get("date_iso", "?")
        parts.append(f"[{i}] ({md.get('_collection')}) {name} (fecha: {d})\n{doc.page_content}")
    context = "\n\n".join(parts)

    system_prompt = load_prompt(args.prompt).format(context=context, question=args.query)

    # OpenAI
    client_oa = OpenAI()
    resp = client_oa.chat.completions.create(
        model=args.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.query},
        ],
        temperature=args.temperature
    )
    answer = resp.choices[0].message.content
    print("\n🧠 RESPUESTA:\n", answer, "\n")
    print("=== Fuentes ===")
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        name = os.path.basename(md.get("rel_path") or md.get("source") or "desconocido")
        d = md.get("date_iso", "?")
        print(f"[{i}] ({md.get('_collection')}) {name}  dist={dist:.3f}  score={cscore:.3f}  fecha={d}")
        print(" ", short_preview(doc.page_content))
        print()

if __name__ == "__main__":
    main()
