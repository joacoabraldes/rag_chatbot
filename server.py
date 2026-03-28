# -*- coding: utf-8 -*-
"""FastAPI entry point — mounts API routes and serves static files."""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

_BASE = Path(__file__).parent

app = FastAPI(title="RAG Chatbot API")

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
