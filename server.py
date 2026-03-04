import os, datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from rag_utils import (
    load_prompt, build_context, format_source_tag, parse_date_iso,
    combined_score, deduplicate_candidates, assess_confidence,
    build_confidence_note, expand_queries_llm, expand_queries_simple,
    split_k_across, boost_keywords, contextualize_query,
)

DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
PROMPT_PATH = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_EMBED = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


# ─── Request / Response models ──────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    k: int = 10
    temperature: float = 0.0
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    embed_model: Optional[str] = None
    collections: Optional[List[str]] = None
    recency_weight: float = 0.35
    half_life_days: int = 30
    use_llm_expansion: bool = False
    use_simple_expansion: bool = False
    history: Optional[List[dict]] = None  # previous messages for multi-turn


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: dict


app = FastAPI(title="RAG Chatbot API")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        model_name = req.embed_model or DEFAULT_EMBED
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        client = chromadb.PersistentClient(path=DB_DIR)
        all_cols = [c.name for c in client.list_collections()]
        cols = req.collections or all_cols
        if not cols:
            raise HTTPException(status_code=400, detail="No collections available.")

        # Contextualise query using conversation history
        search_query = req.query
        if req.history:
            search_query = contextualize_query(
                req.query, req.history, openai_model=req.openai_model
            )

        # Query expansion
        if req.use_llm_expansion:
            queries = expand_queries_llm(search_query, openai_model=req.openai_model)
        elif req.use_simple_expansion:
            queries = expand_queries_simple(search_query)
        else:
            queries = [search_query]

        vectordbs = {
            col: Chroma(client=client, collection_name=col, embedding_function=hf)
            for col in cols
        }

        ks = split_k_across(len(cols), req.k)
        fetch_k_each = [max(5, int(k * 2.0)) for k in ks]
        candidates_all = []

        for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
            if k_each <= 0:
                continue
            for qexp in queries:
                for doc, dist in vdb.similarity_search_with_score(qexp, k=k_each):
                    md = dict(doc.metadata or {})
                    md["_collection"] = col
                    doc.metadata = md
                    candidates_all.append((doc, dist))

        candidates_all = deduplicate_candidates(candidates_all)

        # Score & rank
        today = datetime.date.today()
        scored = []
        for doc, dist in candidates_all:
            d_iso = parse_date_iso(doc.metadata)
            cscore = combined_score(dist, d_iso, today, req.recency_weight, req.half_life_days)
            cscore += boost_keywords(doc.page_content)
            scored.append((doc, dist, cscore))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: req.k]

        # Build context
        context = build_context(top)
        confidence = assess_confidence(top)
        confidence_note = build_confidence_note(confidence)

        system_prompt = load_prompt(PROMPT_PATH).format(
            context=context, question=req.query
        ) + confidence_note

        # Build messages (multi-turn support)
        messages = [{"role": "system", "content": system_prompt}]
        if req.history:
            for msg in req.history[-(10):]:  # last 5 pairs max
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": req.query})

        # Call OpenAI
        client_oa = OpenAI()
        resp = client_oa.chat.completions.create(
            model=req.openai_model,
            messages=messages,
            temperature=req.temperature,
        )
        answer = resp.choices[0].message.content

        # Sources
        sources = []
        for i, (doc, dist, cscore) in enumerate(top, 1):
            tag = format_source_tag(doc.metadata)
            sources.append(f"[{i}] {tag}")

        return AskResponse(answer=answer, sources=sources, confidence=confidence)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
