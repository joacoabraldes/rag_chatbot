# -*- coding: utf-8 -*-
"""CLI query tool for RAG Analyst."""

import os, datetime, argparse

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from core.prompt import load_prompt, format_source_tag, best_source_name
from core.dates import parse_date_iso
from core.scoring import combined_score, boost_keywords
from core.text import short_preview
from core.retrieval import (
    retrieve_from_collections_parallel, deduplicate_candidates, split_k_across,
)
from core.llm import expand_queries_simple, expand_queries_llm


def main():
    p = argparse.ArgumentParser(description="Consulta RAG (CLI).")
    p.add_argument("query", help="Pregunta del usuario")
    p.add_argument("--persist", default="./chroma_db", help="Ruta de la base de Chroma")
    p.add_argument("--collections", nargs="+", help="Colecciones a usar")
    p.add_argument("--k", type=int, default=10, help="Cantidad total de documentos")
    p.add_argument("--embed-model", default="BAAI/bge-m3")
    p.add_argument("--openai-model", default="gpt-5-mini")
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

    # Query expansion
    if args.llm_expansion:
        queries = expand_queries_llm(args.query, openai_model=args.openai_model)
    elif args.use_query_expansion:
        queries = expand_queries_simple(args.query)
    else:
        queries = [args.query]

    # Retrieval using shared parallel retrieval
    ks = split_k_across(len(cols), args.k)
    fetch_k_each = [max(5, int(k * args.fetch_factor)) for k in ks]

    candidates_all = retrieve_from_collections_parallel(vectordbs, queries, fetch_k_each)
    candidates_all = deduplicate_candidates(candidates_all)

    # Date filter
    min_date = None
    if args.min_date:
        try:
            y, m, d = map(int, args.min_date[:10].split("-"))
            min_date = datetime.date(y, m, d)
        except Exception:
            pass

    # Scoring
    today = datetime.date.today()
    results = []
    for doc, dist in candidates_all:
        d_iso = parse_date_iso(doc.metadata)
        if min_date and d_iso and d_iso < min_date:
            continue
        cscore = combined_score(dist, d_iso, today, args.recency_weight, args.half_life)
        cscore += boost_keywords(doc.page_content)
        results.append((doc, dist, cscore))

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:args.k]

    if not top:
        print("No se encontraron resultados.")
        return

    # Context
    parts = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        tag = format_source_tag(md)
        parts.append(f"[{i}] ({tag})\n{doc.page_content}")
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
    )
    answer = resp.choices[0].message.content
    print("\nRESPUESTA:\n", answer, "\n")
    print("=== Fuentes ===")
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        name = best_source_name(md)
        d = md.get("date_iso", "?")
        print(f"[{i}] ({md.get('_collection')}) {name}  dist={dist:.3f}  score={cscore:.3f}  fecha={d}")
        print(" ", short_preview(doc.page_content))
        print()


if __name__ == "__main__":
    main()
