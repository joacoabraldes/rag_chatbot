import os, datetime, logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI, OpenAI, APITimeoutError, APIConnectionError

from rag_utils import (
    load_prompt, build_context, format_source_tag, parse_date_iso,
    combined_score, deduplicate_candidates, assess_confidence,
    build_confidence_note, expand_queries_llm, expand_queries_simple,
    split_k_across, boost_keywords, contextualize_query,
    retrieve_from_collections_parallel,
    make_cache_key, get_cached_results, set_cached_results,
)

DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
PROMPT_PATH = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_EMBED = os.environ.get(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
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


app = FastAPI(title="RAG Analyst API")

# Module-level embedder cache — avoids reloading BAAI/bge-m3 on every request
_embedder_cache: dict = {}


def _get_embedder(model_name: str) -> HuggingFaceEmbeddings:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder_cache[model_name]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        model_name = req.embed_model or DEFAULT_EMBED
        hf = _get_embedder(model_name)

        client = chromadb.PersistentClient(path=DB_DIR)
        all_cols = [c.name for c in client.list_collections()]
        cols = req.collections or all_cols
        if not cols:
            raise HTTPException(status_code=400, detail="No hay colecciones disponibles.")

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

        # Check retrieval cache
        cache_key = make_cache_key(search_query, cols, req.k, req.recency_weight)
        cached = get_cached_results(cache_key)
        if cached is not None:
            candidates_all = cached
        else:
            candidates_all = retrieve_from_collections_parallel(
                vectordbs, queries, fetch_k_each
            )
            set_cached_results(cache_key, candidates_all)

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
        client_oa = OpenAI(timeout=60.0)
        resp = client_oa.chat.completions.create(
            model=req.openai_model,
            messages=messages,
            temperature=req.temperature,
        )
        if not resp.choices or resp.choices[0].message.content is None:
            raise HTTPException(status_code=502, detail="El servicio de IA no genero una respuesta valida.")
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
        logging.exception("Error in /ask endpoint")
        raise HTTPException(status_code=500, detail="Ocurrio un error inesperado al procesar la consulta. Por favor, intenta de nuevo.")


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming variant of /ask — returns Server-Sent Events.

    Each SSE event is: data: <chunk_text>\\n\\n
    A final event data: [DONE]\\n\\n signals end of stream.
    """
    try:
        model_name = req.embed_model or DEFAULT_EMBED
        hf = _get_embedder(model_name)

        client = chromadb.PersistentClient(path=DB_DIR)
        all_cols = [c.name for c in client.list_collections()]
        cols = req.collections or all_cols
        if not cols:
            raise HTTPException(status_code=400, detail="No hay colecciones disponibles.")

        search_query = req.query
        if req.history:
            search_query = contextualize_query(
                req.query, req.history, openai_model=req.openai_model
            )

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

        cache_key = make_cache_key(search_query, cols, req.k, req.recency_weight)
        cached = get_cached_results(cache_key)
        if cached is not None:
            candidates_all = cached
        else:
            candidates_all = retrieve_from_collections_parallel(vectordbs, queries, fetch_k_each)
            set_cached_results(cache_key, candidates_all)

        candidates_all = deduplicate_candidates(candidates_all)

        today = datetime.date.today()
        scored = []
        for doc, dist in candidates_all:
            d_iso = parse_date_iso(doc.metadata)
            cscore = combined_score(dist, d_iso, today, req.recency_weight, req.half_life_days)
            cscore += boost_keywords(doc.page_content)
            scored.append((doc, dist, cscore))
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: req.k]

        context = build_context(top)
        confidence = assess_confidence(top)
        confidence_note = build_confidence_note(confidence)
        try:
            system_prompt = load_prompt(PROMPT_PATH).format(
                context=context, question=req.query
            ) + confidence_note
        except Exception:
            system_prompt = f"Context:\n{context}\n\nQuestion: {req.query}" + confidence_note

        messages = [{"role": "system", "content": system_prompt}]
        if req.history:
            for msg in req.history[-10:]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": req.query})

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in /ask/stream endpoint")
        raise HTTPException(status_code=500, detail="Ocurrio un error inesperado al procesar la consulta. Por favor, intenta de nuevo.")

    async def event_generator():
        try:
            async_client = AsyncOpenAI(timeout=60.0)
            stream = await async_client.chat.completions.create(
                model=req.openai_model,
                messages=messages,
                temperature=req.temperature,
                stream=True,
            )
            async for chunk in stream:
                try:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield f"data: {chunk.choices[0].delta.content}\n\n"
                except Exception:
                    continue
        except (APITimeoutError, APIConnectionError) as e:
            logging.exception("AI service timeout/connection error in stream")
            yield "data: [ERROR] El servicio de IA tardo demasiado o no se pudo conectar. Intenta de nuevo.\n\n"
        except Exception as e:
            logging.exception("Unexpected error in stream")
            yield "data: [ERROR] Ocurrio un error inesperado al generar la respuesta.\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
