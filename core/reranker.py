# -*- coding: utf-8 -*-
"""Cross-encoder re-ranking for retrieved chunks."""

from __future__ import annotations

from typing import Dict, List

from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_model() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def rerank(query: str, chunks: List[Dict], top_n: int) -> List[Dict]:
    """Re-rank chunks using a cross-encoder and return the top_n best.

    Each chunk dict must have a 'text' key.
    """
    if not chunks or top_n <= 0:
        return chunks

    model = get_model()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]
