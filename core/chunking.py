# -*- coding: utf-8 -*-
"""Semantic chunking — groups sentences by cosine similarity."""

import re
from typing import List

import numpy as np

from core.config import SIMILARITY_THRESHOLD
from core.embedder import cosine_similarity, embed_texts


def split_into_sentences(text: str) -> List[str]:
    """Split text into semantic units (paragraphs or sentences).

    Short paragraphs are kept whole; long ones are split at sentence
    boundaries so each unit carries enough context for embedding.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    units: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) < 500:
            units.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            units.extend(s.strip() for s in sentences if s.strip())
    return units


def semantic_chunk(
    sentences: List[str],
    threshold: float | None = None,
) -> List[List[str]]:
    """Group consecutive sentences into chunks by cosine similarity.

    A new chunk starts whenever the similarity between consecutive
    sentence embeddings drops below *threshold* (default from config).
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences]

    embeddings: np.ndarray = embed_texts(sentences)

    chunks: List[List[str]] = []
    current: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(current)
            current = [sentences[i]]

    if current:
        chunks.append(current)

    return chunks
