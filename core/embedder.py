# -*- coding: utf-8 -*-
"""Embedding utilities using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns a 2-D numpy array (n_texts, dim)."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)


@lru_cache(maxsize=256)
def _cached_embed_query(text: str) -> np.ndarray:
    """LRU-cached version. Strings are hashable and arrays here are
    treated as read-only — callers do ``.tolist()`` or ``.dot()``,
    never mutate in place."""
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0]


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a 1-D numpy array.

    Cached via LRU: identical queries (typical when a user re-tries or a
    follow-up shares the prefix) skip the local model call (~50-80ms).
    """
    return _cached_embed_query(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors (= dot product)."""
    return float(np.dot(a, b))
