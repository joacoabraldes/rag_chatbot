# -*- coding: utf-8 -*-
"""Centralized configuration from environment variables."""

import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", "gpt-4.1-nano")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "informes")
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.environ.get("TOP_K", "8"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))
