# -*- coding: utf-8 -*-
"""FastAPI route handlers — chat endpoint with intent routing."""

import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import ChatRequest
from core.config import TOP_K
from core.llm import stream_chat
from core.prompts import build_direct_system_prompt, build_rag_system_prompt
from core.router import classify_intent
from core.vectorstore import list_collections, query_chunks

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
        header = (
            f"[{i}] Fuente: {md.get('source_file', 'desconocido')} | "
            f"Fecha: {md.get('pub_date', 'N/A')} | "
            f"Página: {md.get('page_number', 'N/A')}"
        )
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

    # Step 1: classify intent
    t0 = time.perf_counter()
    try:
        needs_retrieval = classify_intent(req.query)
    except Exception:
        logging.exception("Intent classification failed, defaulting to RAG")
        needs_retrieval = True
    dt_classify = (time.perf_counter() - t0) * 1000

    if needs_retrieval:
        log.info(
            "\033[33m[ROUTER]\033[0m retrieve=\033[1;32mTRUE\033[0m  (%.0fms)",
            dt_classify,
        )
    else:
        log.info(
            "\033[33m[ROUTER]\033[0m retrieve=\033[1;31mFALSE\033[0m (%.0fms) → respuesta directa",
            dt_classify,
        )

    # Step 2: build messages
    sorted_chunks: list = []

    if needs_retrieval:
        t1 = time.perf_counter()
        chunks = query_chunks(req.query, k=TOP_K)
        dt_search = (time.perf_counter() - t1) * 1000

        if chunks:
            context, sorted_chunks = _build_context(chunks)
            system_prompt = build_rag_system_prompt(context, req.show_sources)
            log.info(
                "\033[36m[CHROMA]\033[0m %d chunks recuperados (%.0fms)",
                len(sorted_chunks),
                dt_search,
            )
            for i, ch in enumerate(sorted_chunks, 1):
                md = ch["metadata"]
                log.info(
                    "   [%d] %s | %s | p.%s | dist=%.3f | tags=%s",
                    i,
                    md.get("source_file", "?"),
                    md.get("pub_date", "?"),
                    md.get("page_number", "?"),
                    ch.get("distance", -1),
                    md.get("topic_tags", "")[:50],
                )
        else:
            system_prompt = build_direct_system_prompt()
            log.info(
                "\033[36m[CHROMA]\033[0m \033[31m0 chunks\033[0m (%.0fms) → sin contexto",
                dt_search,
            )
    else:
        system_prompt = build_direct_system_prompt()

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
        try:
            async for text in stream_chat(messages):
                full_answer.append(text)
                for line in text.split("\n"):
                    yield f"data: {line}\n"
                yield "\n"
        except Exception:
            logging.exception("Error in LLM stream")
            yield (
                "data: [ERROR] Ocurrió un error al generar la respuesta.\n\n"
            )

        dt_llm = (time.perf_counter() - t2) * 1000
        answer_text = "".join(full_answer)
        log.info(
            "\033[35m[LLM]\033[0m Respuesta: %d chars (%.0fms)",
            len(answer_text),
            dt_llm,
        )

        yield "data: [DONE]\n\n"

        # Send routing metadata for debug badge
        meta = {
            "retrieve": needs_retrieval,
            "chunks": len(sorted_chunks),
            "sources_sent": bool(req.show_sources and needs_retrieval and sorted_chunks),
        }
        yield f"data: [META] {json.dumps(meta)}\n\n"

        # Send sources metadata if applicable
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

        yield "data: [END]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream"
    )


@router.get("/collections")
def get_collections():
    """List available ChromaDB collections."""
    return {"collections": list_collections()}
