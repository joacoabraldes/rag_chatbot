# -*- coding: utf-8 -*-
"""FastAPI route handlers."""

import os, json, logging, datetime, pathlib, re
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError

from .models import AskRequest
from core.prompt import (
    load_prompt, assess_confidence, build_confidence_note, best_source_name, best_page,
)
from core.dates import parse_date_iso, extract_date_range_from_query
from core.scoring import combined_score, boost_keywords
from core.retrieval import (
    retrieve_from_collections_parallel, deduplicate_candidates,
    split_k_across, build_context,
    make_cache_key, get_cached_results, set_cached_results,
)
from core.llm import (
    contextualize_query, expand_queries_llm, expand_queries_simple,
    generate_followup_questions, generate_suggested_questions,
)
from core.text import postprocess_response

DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
PROMPT_PATH = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_EMBED = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

router = APIRouter()

# Module-level embedder cache
_embedder_cache: dict = {}


def _get_embedder(model_name: str) -> HuggingFaceEmbeddings:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder_cache[model_name]


def _retrieve_and_prepare(req: AskRequest):
    """Shared retrieval, scoring, context building for the streaming endpoint."""
    model_name = req.embed_model or DEFAULT_EMBED
    hf = _get_embedder(model_name)

    client = chromadb.PersistentClient(path=DB_DIR)
    all_cols = [c.name for c in client.list_collections()]
    cols = req.collections or all_cols
    if not cols:
        raise HTTPException(status_code=400, detail="No hay colecciones disponibles.")

    search_query = req.query
    if req.history and req.use_llm_expansion:
        with ThreadPoolExecutor(max_workers=2) as pool:
            ctx_future = pool.submit(
                contextualize_query, req.query, req.history,
                openai_model=req.openai_model,
            )
            exp_future = pool.submit(
                expand_queries_llm, req.query,
                openai_model=req.openai_model,
            )
            try:
                search_query = ctx_future.result(timeout=12)
            except Exception:
                search_query = req.query
            try:
                raw_expansions = exp_future.result(timeout=14)
            except Exception:
                raw_expansions = [req.query]
        queries = [search_query] + [q for q in raw_expansions if q != req.query]
    elif req.history:
        search_query = contextualize_query(
            req.query, req.history, openai_model=req.openai_model
        )
        if req.use_simple_expansion:
            queries = expand_queries_simple(search_query)
        else:
            queries = [search_query]
    elif req.use_llm_expansion:
        queries = expand_queries_llm(search_query, openai_model=req.openai_model)
    elif req.use_simple_expansion:
        queries = expand_queries_simple(search_query)
    else:
        queries = [search_query]

    vectordbs = {
        col: Chroma(client=client, collection_name=col, embedding_function=hf)
        for col in cols
    }

    date_filter = extract_date_range_from_query(req.query)

    ks = split_k_across(len(cols), req.k)
    fetch_k_each = [max(5, int(k * 2.0)) for k in ks]

    cache_key = make_cache_key(search_query, cols, req.k, req.recency_weight)
    cached = get_cached_results(cache_key)
    if cached is not None:
        candidates_all = cached
    else:
        candidates_all = retrieve_from_collections_parallel(
            vectordbs, queries, fetch_k_each, date_filter=date_filter,
        )
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

    return top, confidence, messages


@router.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming endpoint — returns Server-Sent Events."""
    try:
        top, confidence, messages = _retrieve_and_prepare(req)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in /ask/stream endpoint")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error inesperado al procesar la consulta. Por favor, intentá de nuevo.",
        )

    async def event_generator():
        full_answer = []
        try:
            async_client = AsyncOpenAI(timeout=60.0)
            stream = await async_client.chat.completions.create(
                model=req.openai_model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                try:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_answer.append(text)
                        # SSE spec: multi-line data must have each line
                        # prefixed with "data: ", otherwise the client
                        # parser drops lines that lack the prefix.
                        for line in text.split("\n"):
                            yield f"data: {line}\n"
                        yield "\n"
                except Exception:
                    continue
        except (APITimeoutError, APIConnectionError):
            logging.exception("AI service timeout/connection error in stream")
            yield "data: [ERROR] El servicio de IA tardó demasiado o no se pudo conectar. Intentá de nuevo.\n\n"
        except Exception:
            logging.exception("Unexpected error in stream")
            yield "data: [ERROR] Ocurrió un error inesperado al generar la respuesta.\n\n"

        yield "data: [DONE]\n\n"

        answer_text = postprocess_response("".join(full_answer))

        # Parse cited source indices from the answer
        cited_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text))

        # Build sources list — only include cited sources
        sources_list = []
        for i, (doc, dist, cscore) in enumerate(top, 1):
            if cited_indices and i not in cited_indices:
                continue
            md = doc.metadata
            sources_list.append({
                "index": i,
                "filename": best_source_name(md),
                "page": best_page(md),
                "collection": md.get("_collection", ""),
                "date_iso": md.get("date_iso", ""),
                "rel_path": best_source_name(md),
                "text": doc.page_content[:500],
            })
        yield f"data: [SOURCES] {json.dumps(sources_list, ensure_ascii=False)}\n\n"
        try:
            followups = generate_followup_questions(req.query, answer_text, openai_model=req.openai_model)
        except Exception:
            followups = []
        yield f"data: [FOLLOWUPS] {json.dumps(followups, ensure_ascii=False)}\n\n"

        yield "data: [END]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/collections")
def list_collections():
    client = chromadb.PersistentClient(path=DB_DIR)
    return {"collections": [c.name for c in client.list_collections()]}


@router.get("/suggested-questions")
def get_suggested_questions():
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        all_cols = [c.name for c in client.list_collections()]
        if not all_cols:
            return {"questions": []}
        hf = _get_embedder(DEFAULT_EMBED)
        vectordbs = {
            col: Chroma(client=client, collection_name=col, embedding_function=hf)
            for col in all_cols[:3]
        }
        questions = generate_suggested_questions(
            vectordbs, openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        )
        return {"questions": questions}
    except Exception:
        return {"questions": []}


@router.get("/docs/{filename}")
async def serve_document(filename: str):
    safe = pathlib.Path(filename).name
    path = pathlib.Path(DOCS_DIR) / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return FileResponse(str(path), filename=safe)
