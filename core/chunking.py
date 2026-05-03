# -*- coding: utf-8 -*-
"""Semantic chunking with token-size control.

Sentences are grouped by cosine similarity (the chunk grows while
consecutive embeddings stay above ``threshold``), but two extra rules
keep chunk sizes consistent:

- once the running token count reaches ``target_tokens``, a low-similarity
  boundary is preferred (the chunk closes there);
- the running token count never exceeds ``max_tokens`` — when it would,
  the chunk closes regardless of similarity.

Result: chunks of roughly uniform size in tokens that still respect
semantic boundaries instead of cutting mid-thought.
"""

import re
from functools import lru_cache
from typing import List

import numpy as np

from core.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_TARGET_TOKENS,
    SIMILARITY_THRESHOLD,
)
from core.embedder import cosine_similarity, embed_texts


@lru_cache(maxsize=1)
def _encoder():
    """Lazy-load tiktoken once. cl100k_base matches OpenAI gpt-4/5 family
    and gives a stable token count for chunk-size budgeting; we never call
    OpenAI from this module — it's purely for length accounting."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


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
    target_tokens: int | None = None,
    max_tokens: int | None = None,
) -> List[List[str]]:
    """Group consecutive sentences into chunks by cosine similarity + size.

    A new chunk starts when:
    - similarity drops below ``threshold`` AND we've already hit ``target_tokens``
      (semantic boundary preferred once we reach minimum size); or
    - the next sentence would push the running total past ``max_tokens``
      (hard cap).

    Sentences longer than ``max_tokens`` on their own become their own chunk.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if target_tokens is None:
        target_tokens = CHUNK_TARGET_TOKENS
    if max_tokens is None:
        max_tokens = CHUNK_MAX_TOKENS

    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences]

    embeddings: np.ndarray = embed_texts(sentences)
    sentence_tokens = [count_tokens(s) for s in sentences]

    chunks: List[List[str]] = []
    current: List[str] = [sentences[0]]
    current_tokens = sentence_tokens[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        next_tokens = sentence_tokens[i]

        # Hard cap: don't let a chunk grow past max_tokens.
        if current_tokens + next_tokens > max_tokens and current:
            chunks.append(current)
            current = [sentences[i]]
            current_tokens = next_tokens
            continue

        # Soft target: once we've reached target size, honour any drop in
        # similarity below the threshold as a chunk boundary.
        if current_tokens >= target_tokens and sim < threshold:
            chunks.append(current)
            current = [sentences[i]]
            current_tokens = next_tokens
            continue

        current.append(sentences[i])
        current_tokens += next_tokens

    if current:
        chunks.append(current)

    return chunks
