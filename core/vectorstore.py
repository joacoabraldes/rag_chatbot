# -*- coding: utf-8 -*-
"""ChromaDB operations — add, query, list collections."""

from __future__ import annotations

from typing import Dict, List, Optional

import chromadb

from core.config import CHROMA_COLLECTION, CHROMA_DB_DIR
from core.embedder import embed_query, embed_texts

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Lazy-load and cache the ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def get_collection(name: str | None = None) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=name or CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: List[Dict], collection_name: str | None = None) -> None:
    """Add pre-processed chunks to ChromaDB.

    Each chunk dict must have: chunk_id, text, metadata.
    Embeddings are computed here.
    """
    collection = get_collection(collection_name)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(documents).tolist()

    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end],
        )


def query_chunks(
    query: str,
    k: int = 8,
    collection_name: str | None = None,
) -> List[Dict]:
    """Query ChromaDB and return the top-k results as dicts."""
    collection = get_collection(collection_name)
    embedding = embed_query(query).tolist()

    count = collection.count()
    if count == 0:
        return []
    n = min(k, count)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[Dict] = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return chunks


def list_collections() -> List[str]:
    """Return names of all ChromaDB collections."""
    client = get_client()
    return [c.name for c in client.list_collections()]
