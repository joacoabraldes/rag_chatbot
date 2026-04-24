# -*- coding: utf-8 -*-
"""FastAPI route handlers — chat endpoint with intent routing."""

import asyncio
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import ChatRequest
from core.config import RERANK_TOP_N, TOP_K
from core.followups import generate_followups
from core.llm import stream_chat
from core.prompts import build_direct_system_prompt, build_rag_system_prompt
from core.query_rewriter import rewrite_standalone
from core.reranker import rerank
from core.router import classify_intent
from core.vectorstore import (
    get_collection_summary_async,
    list_collections,
    query_chunks_async,
)

log = logging.getLogger("rag")
logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s\033[0m %(message)s",
    datefmt="%H:%M:%S",
)

router = APIRouter()


def _build_context(chunks: list) -> tuple[str, list]:
    """Build context string from retrieved chunks, sorted chronologically."""
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c["metadata"].get("pub_date", ""),
    )
    parts = []
    for i, chunk in enumerate(sorted_chunks, 1):
        md = chunk["metadata"]
        header_fields = [
            f"[{i}] Fuente: {md.get('source_file', 'desconocido')}",
            f"Fecha: {md.get('pub_date', 'N/A')}",
            f"Página: {md.get('page_number', 'N/A')}",
        ]
        topic_tags = (md.get("topic_tags") or "").strip()
        if topic_tags:
            header_fields.append(f"Temas: {topic_tags}")
        header = " | ".join(header_fields)
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts), sorted_chunks


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint with intent-based routing."""
    log.info("─" * 60)
    log.info(
        "\033[1m>>> QUERY:\033[0m %s | show_sources=%s",
        req.query[:80],
        req.show_sources,
    )

    # Step 1a: rewrite the query for retrieval if history exists.
    # Without history the rewriter is a no-op (returns the original query),
    # so we skip the LLM call entirely — only multi-turn queries pay for it.
    history_for_rewrite = req.history or []
    has_history = any(
        m.get("role") in ("user", "assistant") for m in history_for_rewrite
    )
    t_rewrite = time.perf_counter()
    if has_history:
        retrieval_query = await rewrite_standalone(req.query, history_for_rewrite)
        dt_rewrite = (time.perf_counter() - t_rewrite) * 1000
        if retrieval_query.strip() != req.query.strip():
            log.info(
                "\033[96m[REWRITE]\033[0m %.0fms | '%s' → '%s'",
                dt_rewrite,
                req.query[:60],
                retrieval_query[:60],
            )
        else:
            log.info(
                "\033[96m[REWRITE]\033[0m %.0fms | sin cambios",
                dt_rewrite,
            )
    else:
        retrieval_query = req.query

    # Step 1b: classify intent + vector search + collection summary in parallel
    t0 = time.perf_counter()

    async def _classify():
        try:
            return await classify_intent(req.query)
        except Exception:
            logging.exception("Intent classification failed, defaulting to RAG")
            return True

    async def _summary():
        try:
            return await get_collection_summary_async()
        except Exception:
            logging.exception("Collection summary failed; continuing without it")
            return None

    needs_retrieval, chunks, collection_summary = await asyncio.gather(
        _classify(),
        query_chunks_async(retrieval_query, k=TOP_K),
        _summary(),
    )

    dt_parallel = (time.perf_counter() - t0) * 1000

    if needs_retrieval:
        log.info(
            "\033[33m[ROUTER]\033[0m retrieve=\033[1;32mTRUE\033[0m  (%.0fms, parallel)",
            dt_parallel,
        )
    else:
        log.info(
            "\033[33m[ROUTER]\033[0m retrieve=\033[1;31mFALSE\033[0m (%.0fms) → respuesta directa",
            dt_parallel,
        )

    # Step 2: build messages
    sorted_chunks: list = []

    if needs_retrieval and chunks:
        log.info(
            "\033[36m[CHROMA]\033[0m %d chunks recuperados (%.0fms, parallel)",
            len(chunks),
            dt_parallel,
        )

        t_rerank = time.perf_counter()
        chunks = rerank(retrieval_query, chunks, top_n=RERANK_TOP_N)
        dt_rerank = (time.perf_counter() - t_rerank) * 1000
        log.info(
            "\033[34m[RERANK]\033[0m %d → %d chunks (%.0fms)",
            len(chunks) + (TOP_K - len(chunks)),
            len(chunks),
            dt_rerank,
        )

        context, sorted_chunks = _build_context(chunks)
        system_prompt = build_rag_system_prompt(
            context, req.show_sources, collection_summary
        )
        for i, ch in enumerate(sorted_chunks, 1):
            md = ch["metadata"]
            log.info(
                "   [%d] %s | %s | p.%s | dist=%.3f | rerank=%.3f | tags=%s",
                i,
                md.get("source_file", "?"),
                md.get("pub_date", "?"),
                md.get("page_number", "?"),
                ch.get("distance", -1),
                ch.get("rerank_score", -1),
                md.get("topic_tags", "")[:50],
            )
    elif needs_retrieval:
        log.info(
            "\033[36m[CHROMA]\033[0m \033[31m0 chunks\033[0m (%.0fms) → sin contexto",
            dt_parallel,
        )
        system_prompt = build_direct_system_prompt(collection_summary)
    else:
        system_prompt = build_direct_system_prompt(collection_summary)

    messages = [{"role": "system", "content": system_prompt}]

    if req.history:
        for msg in req.history[-10:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

    messages.append({"role": "user", "content": req.query})

    log.info(
        "\033[35m[LLM]\033[0m Enviando %d mensajes (system prompt: %d chars)",
        len(messages),
        len(system_prompt),
    )

    # Step 3: stream response
    async def event_generator():
        full_answer = []
        t2 = time.perf_counter()
        t_first_token: float | None = None
        n_chunks = 0
        try:
            async for text in stream_chat(messages):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                full_answer.append(text)
                n_chunks += 1
                yield f"data: {json.dumps(text)}\n\n"
        except Exception:
            logging.exception("Error in LLM stream")
            yield (
                "data: [ERROR] Ocurrió un error al generar la respuesta.\n\n"
            )

        t_end = time.perf_counter()
        dt_llm = (t_end - t2) * 1000
        dt_ttft = ((t_first_token - t2) * 1000) if t_first_token else None
        dt_stream = ((t_end - t_first_token) * 1000) if t_first_token else None
        answer_text = "".join(full_answer)
        log.info(
            "\033[35m[LLM]\033[0m Respuesta: %d chars | total=%.0fms | ttft=%s | stream=%s | chunks=%d",
            len(answer_text),
            dt_llm,
            f"{dt_ttft:.0f}ms" if dt_ttft is not None else "n/a",
            f"{dt_stream:.0f}ms" if dt_stream is not None else "n/a",
            n_chunks,
        )

        yield "data: [DONE]\n\n"

        # Send routing metadata for debug badge
        meta = {
            "retrieve": needs_retrieval,
            "chunks": len(sorted_chunks),
            "sources_sent": bool(req.show_sources and needs_retrieval and sorted_chunks),
            "ttft_ms": round(dt_ttft) if dt_ttft is not None else None,
            "total_ms": round(dt_llm),
            "stream_chunks": n_chunks,
            "rewritten": (
                retrieval_query if retrieval_query.strip() != req.query.strip() else None
            ),
        }
        yield f"data: [META] {json.dumps(meta, ensure_ascii=False)}\n\n"

        # Send sources metadata if applicable — includes chunk text so the
        # frontend can show a preview drawer without a second round-trip.
        if req.show_sources and needs_retrieval and sorted_chunks:
            sources = []
            for i, chunk in enumerate(sorted_chunks, 1):
                md = chunk["metadata"]
                sources.append(
                    {
                        "index": i,
                        "source_file": md.get("source_file", ""),
                        "pub_date": md.get("pub_date", ""),
                        "page_number": md.get("page_number", ""),
                        "topic_tags": md.get("topic_tags", ""),
                        "text": chunk.get("text", ""),
                    }
                )
            log.info(
                "\033[32m[SOURCES]\033[0m Enviando %d fuentes al frontend",
                len(sources),
            )
            yield (
                f"data: [SOURCES] "
                f"{json.dumps(sources, ensure_ascii=False)}\n\n"
            )
        else:
            reasons = []
            if not req.show_sources:
                reasons.append("show_sources=false")
            if not needs_retrieval:
                reasons.append("retrieve=false")
            if needs_retrieval and not sorted_chunks:
                reasons.append("0 chunks")
            log.info(
                "\033[32m[SOURCES]\033[0m No se envían fuentes (%s)",
                ", ".join(reasons) if reasons else "?",
            )

        # Generate 3 follow-up suggestions after the answer is complete.
        # Skip if the answer is empty or was an error.
        if answer_text.strip() and not answer_text.startswith("[ERROR]"):
            t_fu = time.perf_counter()
            try:
                followups = await generate_followups(req.query, answer_text)
            except Exception:
                logging.exception("Follow-ups generation failed")
                followups = []
            dt_fu = (time.perf_counter() - t_fu) * 1000
            if followups:
                log.info(
                    "\033[95m[FOLLOWUPS]\033[0m %d sugerencias (%.0fms)",
                    len(followups),
                    dt_fu,
                )
                yield (
                    f"data: [FOLLOWUPS] "
                    f"{json.dumps(followups, ensure_ascii=False)}\n\n"
                )
            else:
                log.info(
                    "\033[95m[FOLLOWUPS]\033[0m sin sugerencias (%.0fms)",
                    dt_fu,
                )

        yield "data: [END]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/collections")
def get_collections():
    """List available ChromaDB collections."""
    return {"collections": list_collections()}
