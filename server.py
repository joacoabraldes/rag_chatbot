# -*- coding: utf-8 -*-
"""FastAPI entry point — mounts API routes and serves static files."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

log = logging.getLogger("rag")

_BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload ML models at startup to avoid cold-start on first request."""
    log.info("Precargando modelos...")
    from core.embedder import get_model as get_embedder
    from core.reranker import get_model as get_reranker
    from core.vectorstore import get_collection_summary

    get_embedder()
    get_reranker()
    # Warm the collection-summary cache so the first request doesn't pay the scan.
    try:
        summary = get_collection_summary()
        log.info(
            "Colección: %d docs, %d chunks, fechas %s -> %s",
            summary.get("total_docs", 0),
            summary.get("total_chunks", 0),
            summary.get("date_min"),
            summary.get("date_max"),
        )
    except Exception:
        log.exception("No se pudo precomputar el resumen de la colección")
    log.info("Modelos cargados.")
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def serve_index():
    return FileResponse(_BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
