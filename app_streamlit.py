# -*- coding: utf-8 -*-
import os, json, datetime, math
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import re, unicodedata
import langdetect

from rag_utils import (
    clean_text, clean_keep_ascii_marks, escape_markdown, normalize_plain,
    extract_date_from_text, parse_date_iso, recency_score, sim_from_distance,
    combined_score, format_source_tag, short_preview, deduplicate_candidates,
    assess_confidence, build_confidence_note, load_prompt, save_history,
    expand_queries_llm, expand_queries_simple, split_k_across, now_iso,
    best_source_name, best_page, boost_keywords, build_context,
    contextualize_query,
)

# ========= App =========

st.set_page_config(page_title="PARROT RAG", layout="wide")
st.title("🦜 PARROT RAG — say stupidities in a fancy way")

DEFAULT_PERSIST   = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBED     = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_PATH       = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_HISTORY   = "./history/chat_history.jsonl"

# How many past messages (user+assistant pairs) to include in the LLM context
MAX_HISTORY_PAIRS = 5

# ----- Sidebar -----
st.sidebar.header("⚙️ Configuración")
persist_dir  = st.sidebar.text_input("Ruta de Chroma (persist)", value=DEFAULT_PERSIST)

client = chromadb.PersistentClient(path=persist_dir)
collections = [c.name for c in client.list_collections()]
if not collections:
    st.sidebar.warning("No hay colecciones. Ingestá documentos con ingest.py")

selected_cols = st.sidebar.multiselect("Colecciones a consultar", options=collections, default=collections[:1] if collections else [])
k_total      = st.sidebar.slider("Top-k total", 1, 30, 10, 1)
temperature  = st.sidebar.slider("Temperature (OpenAI)", 0.0, 1.0, 0.0, 0.1)
openai_model = st.sidebar.text_input("Modelo OpenAI", value=DEFAULT_OAI_MODEL)
embed_model  = st.sidebar.text_input("Modelo de Embeddings (HF)", value=DEFAULT_EMBED)
history_path = st.sidebar.text_input("Archivo de historial (JSONL)", value=DEFAULT_HISTORY)
show_sources = st.sidebar.checkbox("Mostrar fuentes", value=True)
show_scores  = st.sidebar.checkbox("Mostrar score", value=False)
show_preview = st.sidebar.checkbox("Mostrar preview", value=True)

st.sidebar.markdown("### 🗓️ Recencia")
recency_weight    = st.sidebar.slider("Peso de recencia", 0.0, 1.0, 0.35, 0.05)
recency_half_life = st.sidebar.number_input("Half-life (días)", min_value=1, max_value=365, value=30, step=1)
min_date_str      = st.sidebar.text_input("Filtrar desde fecha (YYYY-MM-DD)", value="")

st.sidebar.markdown("### 🔎 Recuperación avanzada")
fetch_factor          = st.sidebar.slider("Fetch factor", 1.0, 4.0, 2.0, 0.5)
use_query_expansion   = st.sidebar.checkbox("Usar expansión de consulta (simple)", value=False)
use_llm_expansion     = st.sidebar.checkbox("Usar expansión de consulta (LLM)", value=True)
confidence_threshold  = st.sidebar.slider("Umbral de similitud (confianza)", 0.1, 0.8, 0.35, 0.05)
min_good_results      = st.sidebar.number_input("Mín. fragmentos relevantes", min_value=1, max_value=10, value=2, step=1)

st.sidebar.markdown("### 💬 Memoria conversacional")
use_memory = st.sidebar.checkbox("Enviar historial al LLM (multi-turn)", value=True)
max_history = st.sidebar.slider("Pares de mensajes a enviar", 1, 15, MAX_HISTORY_PAIRS, 1)

render_plain = st.sidebar.checkbox("Render respuesta como texto plano", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Requiere OPENAI_API_KEY en el entorno.")

# ----- Caches -----
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource(show_spinner=False)
def get_vectordb(persist_path: str, collection_name: str, embedder):
    return Chroma(client=chromadb.PersistentClient(path=persist_path),
                  collection_name=collection_name,
                  embedding_function=embedder)

embedder = get_embedder(embed_model)
vectordbs = {col: get_vectordb(persist_dir, col, embedder) for col in selected_cols}

# ----- Chat render previo -----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_sources and m.get("sources"):
            with st.expander("Fuentes"):
                for s in m["sources"]:
                    st.text(clean_text(s))


# ----- Helper: build conversation history for OpenAI -----
def build_chat_messages(system_prompt: str, user_input: str, max_pairs: int) -> list:
    """Build the messages list including conversation history for multi-turn.

    Includes up to *max_pairs* previous user/assistant exchanges so the LLM
    can resolve follow-up questions like "tell me more" or "what about X?".
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Gather previous turns (skip the current one which hasn't been appended yet)
    history = st.session_state.messages
    # Take the last N pairs (user + assistant = 2 messages per pair)
    recent = history[-(max_pairs * 2):]

    for msg in recent:
        role = msg["role"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg["content"]})

    # Current user question
    messages.append({"role": "user", "content": user_input})
    return messages


# ----- Parse min_date from sidebar -----
def parse_min_date(s: str) -> Optional[datetime.date]:
    if not s or not s.strip():
        return None
    try:
        y, m, d = map(int, s.strip()[:10].split("-"))
        return datetime.date(y, m, d)
    except Exception:
        return None


# ----- Interacción -----
user_input = st.chat_input("Escribí tu pregunta...")
if user_input and selected_cols:
    # mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Recuperación
    with st.spinner("Buscando contexto..."):
        # --- Query contextualisation (multi-turn) ---
        # Rewrite vague follow-ups ("contame más", "un resumen") into
        # self-contained queries using the conversation history so that
        # the vector search retrieves the right documents.
        search_query = user_input
        if use_memory and st.session_state.messages:
            search_query = contextualize_query(
                user_input,
                st.session_state.messages,
                openai_model=openai_model,
                max_pairs=max_history,
            )

        target_iso = extract_date_from_text(search_query)
        target_date = None
        if target_iso:
            try:
                y, m, d = map(int, target_iso.split("-"))
                target_date = datetime.date(y, m, d)
            except Exception:
                target_date = None

        # Parse the min_date filter from sidebar (BUG FIX: was previously unused)
        min_date = parse_min_date(min_date_str)

        ks = split_k_across(len(selected_cols), k_total)
        fetch_k_each = [max(5, int(k * fetch_factor)) for k in ks]
        candidates_all = []

        # Determine query expansion strategy
        if use_llm_expansion:
            queries = expand_queries_llm(search_query, openai_model=openai_model)
        elif use_query_expansion:
            queries = expand_queries_simple(search_query)
        else:
            queries = [search_query]

        for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
            if k_each <= 0:
                continue
            for qexp in queries:
                for doc, dist in vdb.similarity_search_with_score(qexp, k=k_each):
                    md = dict(doc.metadata or {})
                    md["_collection"] = col
                    doc.metadata = md
                    candidates_all.append((doc, dist))

        # Deduplication
        candidates_all = deduplicate_candidates(candidates_all)

        # Apply min_date filter (BUG FIX)
        if min_date:
            candidates_all = [
                (doc, dist) for doc, dist in candidates_all
                if not parse_date_iso(doc.metadata) or parse_date_iso(doc.metadata) >= min_date
            ]

        filtered = []
        if target_date:
            window = 3
            for doc, dist in candidates_all:
                d_iso = parse_date_iso(doc.metadata)
                if d_iso and abs((d_iso - target_date).days) <= window:
                    filtered.append((doc, dist))
        else:
            filtered = candidates_all

        scored = []
        today = datetime.date.today()
        for doc, dist in filtered:
            d_iso = parse_date_iso(doc.metadata)
            cscore = combined_score(dist, d_iso, today, recency_weight, recency_half_life)
            cscore += boost_keywords(doc.page_content)
            scored.append((doc, dist, cscore))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:k_total]
        top_docs = [doc for (doc, _, _) in top]

        context = "\n\n".join(
            clean_keep_ascii_marks(f"[{i}] ({format_source_tag(doc.metadata)})\n{doc.page_content}")
            for i, doc in enumerate(top_docs, 1)
        )

        # Confidence assessment
        confidence = assess_confidence(top, sim_threshold=confidence_threshold,
                                       min_good_results=min_good_results)
        confidence_note = build_confidence_note(confidence)

        system_prompt = load_prompt(PROMPT_PATH).format(
            context=context, question=user_input
        ) + confidence_note

    # LLM call with multi-turn conversation memory
    with st.spinner("Consultando OpenAI..."):
        try:
            client_oa = OpenAI()

            # Build messages: include history if memory is enabled
            if use_memory:
                messages = build_chat_messages(system_prompt, user_input, max_history)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ]

            resp = client_oa.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=temperature,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Error llamando a OpenAI: {e}"

    # Fuentes
    sources = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        tag = format_source_tag(doc.metadata)
        if show_scores:
            tag = f"{tag}\n  dist={dist:.4f}  score={cscore:.4f}"
        if show_preview:
            prev = short_preview(doc.page_content)
            tag = f"{tag}\n> {prev}"
        sources.append(f"[{i}] ({tag})")

    # Respuesta
    with st.chat_message("assistant"):
        # Confidence indicator
        if not confidence["confident"]:
            if confidence["reason"] == "no_results":
                st.warning("⚠️ No se encontraron fragmentos relevantes. La respuesta puede ser limitada.")
            elif confidence["reason"] == "low_relevance":
                st.info(f"ℹ️ Contexto parcial ({confidence['good_count']} fragmentos relevantes, "
                        f"similitud promedio: {confidence['avg_sim']:.2f}). "
                        "Considerá reformular o dar más detalles.")
        if render_plain:
            st.text(normalize_plain(answer))
        else:
            st.markdown(escape_markdown(answer))
        if show_sources and sources:
            with st.expander("Fuentes"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "collections": selected_cols,
    })

    # Guardar historial
    record = {
        "timestamp": now_iso(),
        "query": user_input,
        "answer": answer,
        "sources": sources,
        "collections": selected_cols,
        "k_total": k_total,
        "embed_model": embed_model,
        "openai_model": openai_model,
        "temperature": temperature,
        "recency_weight": recency_weight,
        "recency_half_life": recency_half_life,
        "min_date": min_date_str,
        "confidence": confidence,
        "expansion_mode": "llm" if use_llm_expansion else ("simple" if use_query_expansion else "none"),
        "memory_enabled": use_memory,
    }
    save_history(history_path, record)

elif user_input and not selected_cols:
    st.warning("Seleccioná al menos una colección en la barra lateral.")
