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
DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
# Postgres + pgvector (Supabase) is the only vector backend. The chunk
# store and the SQL tools (fx, forex) live in the same database.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.environ.get("TOP_K", "8"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))

# Token-aware semantic chunking. Sentences keep being grouped by cosine
# similarity, but a new chunk is forced once the running total reaches the
# target, and never exceeds the max.
CHUNK_TARGET_TOKENS = int(os.environ.get("CHUNK_TARGET_TOKENS", "400"))
CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "700"))

# Hybrid retrieval: keep up to TOP_K chunks AND drop anything below this
# cosine similarity. Tuned empirically against the golden set.
SIMILARITY_THRESHOLD_RETRIEVAL = float(
    os.environ.get("SIMILARITY_THRESHOLD_RETRIEVAL", "0.30")
)
