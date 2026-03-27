# -*- coding: utf-8 -*-
"""Embedding utilities using sentence-transformers."""

from __future__ import annotations

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


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a 1-D numpy array."""
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors (= dot product)."""
    return float(np.dot(a, b))
