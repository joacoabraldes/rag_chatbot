# -*- coding: utf-8 -*-
"""FastAPI route handlers — chat endpoint with intent routing."""

import asyncio
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import ChatRequest
from core.config import OPENAI_MODEL, RERANK_TOP_N, TOP_K
from core.filter_extractor import extract_retrieval_filter
from core.followups import generate_followups
from core.llm import stream_chat
from core.observability import RequestTrace
from core.prompts import build_direct_system_prompt, build_rag_system_prompt
from core.query_rewriter import rewrite_standalone
from core.reranker import rerank
from core.router import classify_intent
from core.sql_tools import TOOLS as SQL_TOOLS
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
        section = (md.get("section") or "").strip()
        if section:
            header_fields.append(f"Sección: {section}")
        topic_tags = (md.get("topic_tags") or "").strip()
        if topic_tags:
            header_fields.append(f"Temas: {topic_tags}")
        header = " | ".join(header_fields)
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts), sorted_chunks


async def _retrieve_chunks_with_filter_fallback(
    retrieval_query: str,
    where: dict | None,
) -> tuple[list, bool]:
    """Run filtered retrieval and retry without filters if it returns no chunks."""
    chunks = await query_chunks_async(
        retrieval_query,
        k=TOP_K,
        where=where,
    )
    if chunks or where is None:
        return chunks, False

    fallback_chunks = await query_chunks_async(
        retrieval_query,
        k=TOP_K,
        where=None,
    )
    return fallback_chunks, True


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint with intent-based routing."""
    trace = RequestTrace(
        query=req.query,
        show_sources=bool(req.show_sources),
        history_len=len(req.history or []),
    )

    log.info("─" * 60)
    log.info(
        "\033[1m>>> %s\033[0m %s | show_sources=%s",
        trace.request_id,
        req.query[:80],
        req.show_sources,
    )

    # Step 1a: rewrite the query for retrieval if history exists.
    history_for_rewrite = req.history or []
    has_history = any(
        m.get("role") in ("user", "assistant") for m in history_for_rewrite
    )
    with trace.span("rewrite") as sp:
        if has_history:
            t_rewrite = time.perf_counter()
            retrieval_query = await rewrite_standalone(
                req.query, history_for_rewrite, trace=trace
            )
            dt_rewrite = (time.perf_counter() - t_rewrite) * 1000
            rewritten = retrieval_query.strip() != req.query.strip()
            sp.set(rewritten=rewritten, has_history=True)
            log.info(
                "\033[96m[REWRITE]\033[0m %.0fms | %s",
                dt_rewrite,
                f"'{req.query[:60]}' → '{retrieval_query[:60]}'" if rewritten else "sin cambios",
            )
        else:
            retrieval_query = req.query
            sp.set(skipped=True)

    # Step 1b: classify intent + collection summary in parallel.
    async def _classify():
        try:
            return await classify_intent(req.query, trace=trace)
        except Exception:
            logging.exception("Intent classification failed, defaulting to RAG")
            return True

    async def _summary():
        try:
            return await get_collection_summary_async()
        except Exception:
            logging.exception("Collection summary failed; continuing without it")
            return None

    with trace.span("router_summary") as sp:
        t0 = time.perf_counter()
        needs_retrieval, collection_summary = await asyncio.gather(
            _classify(),
            _summary(),
        )
        dt_parallel = (time.perf_counter() - t0) * 1000
        sp.set(
            needs_retrieval=bool(needs_retrieval),
            summary_chunks=(collection_summary or {}).get("total_chunks"),
        )

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

    chunks: list = []
    filter_info = {"where": None, "extracted": {}, "applied": False}

    if needs_retrieval:
        with trace.span("filter_extract") as sp:
            t_filter = time.perf_counter()
            filter_info = await extract_retrieval_filter(
                retrieval_query,
                collection_summary,
                trace=trace,
            )
            dt_filter = (time.perf_counter() - t_filter) * 1000
            sp.set(
                applied=bool(filter_info.get("applied")),
                extracted=filter_info.get("extracted", {}),
            )
        if filter_info.get("applied"):
            log.info(
                "\033[92m[FILTER]\033[0m applied=true (%.0fms) | %s",
                dt_filter,
                json.dumps(filter_info.get("extracted", {}), ensure_ascii=False),
            )
        else:
            log.info("\033[92m[FILTER]\033[0m applied=false (%.0fms)", dt_filter)

        with trace.span("retrieval") as sp:
            t_q = time.perf_counter()
            chunks, used_fallback = await _retrieve_chunks_with_filter_fallback(
                retrieval_query,
                where=filter_info.get("where"),
            )
            dt_q = (time.perf_counter() - t_q) * 1000
            sp.set(k=TOP_K, n_chunks=len(chunks), fallback=used_fallback)
        log.info(
            "\033[36m[VEC]\033[0m %d chunks recuperados (%.0fms)",
            len(chunks),
            dt_q,
        )
        if used_fallback:
            log.info(
                "\033[36m[VEC]\033[0m fallback sin filtros aplicado (consulta filtrada sin resultados)",
            )

    sorted_chunks: list = []

    if needs_retrieval and chunks:
        with trace.span("rerank") as sp:
            n_in = len(chunks)
            t_rerank = time.perf_counter()
            chunks = rerank(retrieval_query, chunks, top_n=RERANK_TOP_N)
            dt_rerank = (time.perf_counter() - t_rerank) * 1000
            top_score = chunks[0].get("rerank_score", 0.0) if chunks else 0.0
            sp.set(n_in=n_in, n_out=len(chunks), top_score=round(float(top_score), 3))
        log.info(
            "\033[34m[RERANK]\033[0m %d → %d chunks (%.0fms)",
            n_in,
            len(chunks),
            dt_rerank,
        )

        context, sorted_chunks = _build_context(chunks)
        trace.add_chunks(sorted_chunks)
        system_prompt = build_rag_system_prompt(
            context, req.show_sources, collection_summary
        )
        for i, ch in enumerate(sorted_chunks, 1):
            md = ch["metadata"]
            log.info(
                "   [%d] %s | %s | p.%s | sec=%s | sim=%.3f | rerank=%.3f | tags=%s",
                i,
                md.get("source_file", "?"),
                md.get("pub_date", "?"),
                md.get("page_number", "?"),
                md.get("section", "?"),
                ch.get("similarity", 1.0 - ch.get("distance", 1.0)),
                ch.get("rerank_score", -1),
                md.get("topic_tags", "")[:50],
            )
    elif needs_retrieval:
        log.info(
            "\033[36m[VEC]\033[0m \033[31m0 chunks\033[0m (%.0fms) → sin contexto",
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
        llm_attrs: dict = {"model": OPENAI_MODEL}
        had_error = False

        try:
            async for text in stream_chat(messages, tools=SQL_TOOLS, trace=trace):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                full_answer.append(text)
                n_chunks += 1
                yield f"data: {json.dumps(text)}\n\n"
        except Exception as e:
            had_error = True
            trace.errors.append(
                {"stage": "llm", "kind": type(e).__name__, "msg": str(e)}
            )
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
        llm_attrs.update(
            {
                "duration_ms": round(dt_llm),
                "ttft_ms": round(dt_ttft) if dt_ttft is not None else None,
                "stream_chunks": n_chunks,
                "answer_chars": len(answer_text),
            }
        )
        if had_error:
            llm_attrs["error"] = True
        trace.spans["llm"] = llm_attrs

        yield "data: [DONE]\n\n"

        # Generate follow-ups before finalizing the trace so they're recorded.
        followups: list = []
        if answer_text.strip() and not answer_text.startswith("[ERROR]"):
            with trace.span("followups") as sp:
                t_fu = time.perf_counter()
                try:
                    followups = await generate_followups(
                        req.query, answer_text, trace=trace
                    )
                except Exception:
                    logging.exception("Follow-ups generation failed")
                    followups = []
                dt_fu = (time.perf_counter() - t_fu) * 1000
                sp.set(n=len(followups))
            if followups:
                log.info(
                    "\033[95m[FOLLOWUPS]\033[0m %d sugerencias (%.0fms)",
                    len(followups),
                    dt_fu,
                )
            else:
                log.info(
                    "\033[95m[FOLLOWUPS]\033[0m sin sugerencias (%.0fms)",
                    dt_fu,
                )

        # Finalize the trace BEFORE emitting [META] so we can attach cost.
        record = trace.finalize(status="error" if had_error else "ok")

        # Send routing metadata for debug badge
        meta = {
            "request_id": trace.request_id,
            "retrieve": needs_retrieval,
            "filter_applied": bool(filter_info.get("applied")),
            "filter": filter_info.get("extracted", {}),
            "chunks": len(sorted_chunks),
            "sources_sent": bool(req.show_sources and needs_retrieval and sorted_chunks),
            "ttft_ms": llm_attrs.get("ttft_ms"),
            "total_ms": record["duration_ms"],
            "stream_chunks": n_chunks,
            "rewritten": (
                retrieval_query if retrieval_query.strip() != req.query.strip() else None
            ),
            "tools": [t["name"] for t in trace.tools],
            "cost_usd": record["cost_usd"],
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
                        "section": md.get("section", ""),
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

        if followups:
            yield (
                f"data: [FOLLOWUPS] "
                f"{json.dumps(followups, ensure_ascii=False)}\n\n"
            )

        log.info(
            "\033[90m[TRACE]\033[0m %s | %dms | $%.4f | tokens %s",
            trace.request_id,
            record["duration_ms"],
            record["cost_usd"],
            json.dumps(
                {m: u["prompt_tokens"] + u["completion_tokens"]
                 for m, u in record["usage"].items()},
                ensure_ascii=False,
            ),
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
    """List available collections (Postgres has a single fixed one)."""
    return {"collections": list_collections()}
