# -*- coding: utf-8 -*-
import os, json, datetime, math, pathlib, uuid, tempfile
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

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from rag_utils import (
    clean_text, clean_keep_ascii_marks, escape_markdown, normalize_plain,
    extract_date_from_text, parse_date_iso, recency_score, sim_from_distance,
    combined_score, format_source_tag, short_preview, deduplicate_candidates,
    assess_confidence, build_confidence_note, load_prompt, save_history,
    expand_queries_llm, expand_queries_simple, split_k_across, now_iso,
    best_source_name, best_page, boost_keywords, build_context,
    contextualize_query,
)
from ingest import (
    chunk_documents, infer_date_iso, try_parse_date_from_string,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

# ========= App =========

st.set_page_config(page_title="PARROT RAG", layout="wide")
st.title("🦜 PARROT RAG — say stupidities in a fancy way")

# Custom CSS for better UX
st.markdown("""
<style>
.stChatMessage { max-width: 90%; }
.feedback-btn { display: inline-block; cursor: pointer; font-size: 1.3rem; margin-right: 8px; opacity:0.6; }
.feedback-btn:hover { opacity:1; }
.suggested-q { padding: 8px 16px; border: 1px solid #444; border-radius: 18px;
    display: inline-block; margin: 4px 6px; cursor: pointer; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

DEFAULT_PERSIST   = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBED     = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
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

with st.sidebar.expander("🤖 Modelos y respuesta", expanded=False):
    k_total      = st.slider("Top-k total", 1, 30, 10, 1)
    temperature  = st.slider("Temperature (OpenAI)", 0.0, 1.0, 0.0, 0.1)
    openai_model = st.text_input("Modelo OpenAI", value=DEFAULT_OAI_MODEL)
    embed_model  = st.text_input("Modelo de Embeddings (HF)", value=DEFAULT_EMBED)
    show_sources = st.checkbox("Mostrar fuentes", value=True)
    show_scores  = st.checkbox("Mostrar score", value=False)
    show_preview = st.checkbox("Mostrar preview", value=True)
    render_plain = st.checkbox("Render respuesta como texto plano", value=False)

with st.sidebar.expander("🗓️ Recencia", expanded=False):
    recency_weight    = st.slider("Peso de recencia", 0.0, 1.0, 0.35, 0.05)
    recency_half_life = st.number_input("Half-life (días)", min_value=1, max_value=365, value=30, step=1)
    min_date_str      = st.text_input("Filtrar desde fecha (YYYY-MM-DD)", value="")

with st.sidebar.expander("🔎 Recuperación avanzada", expanded=False):
    fetch_factor          = st.slider("Fetch factor", 1.0, 4.0, 2.0, 0.5)
    use_query_expansion   = st.checkbox("Usar expansión de consulta (simple)", value=False)
    use_llm_expansion     = st.checkbox("Usar expansión de consulta (LLM)", value=True)
    confidence_threshold  = st.slider("Umbral de similitud (confianza)", 0.1, 0.8, 0.35, 0.05)
    min_good_results      = st.number_input("Mín. fragmentos relevantes", min_value=1, max_value=10, value=2, step=1)

with st.sidebar.expander("💬 Memoria conversacional", expanded=False):
    use_memory = st.checkbox("Enviar historial al LLM (multi-turn)", value=True)
    max_history = st.slider("Pares de mensajes a enviar", 1, 15, MAX_HISTORY_PAIRS, 1)
    history_path = st.text_input("Archivo de historial (JSONL)", value=DEFAULT_HISTORY)

st.sidebar.markdown("---")

# ----- Clear / Export chat -----
col_clear, col_export = st.sidebar.columns(2)
if col_clear.button("🗑️ Limpiar chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.pop("feedback", None)
    st.rerun()
if col_export.button("📤 Exportar chat", use_container_width=True):
    if st.session_state.get("messages"):
        export_data = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2, default=str)
        st.sidebar.download_button(
            label="⬇️ Descargar JSON",
            data=export_data,
            file_name=f"chat_export_{datetime.datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.sidebar.info("No hay mensajes para exportar.")

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

# ----- File Upload & Ingestion -----
st.sidebar.markdown("### 📄 Subir documentos")
upload_collection = st.sidebar.text_input("Colección destino", value=collections[0] if collections else "docs")
use_semantic_chunking = st.sidebar.checkbox("Usar chunking semántico", value=True)
uploaded_files = st.sidebar.file_uploader(
    "Subir .txt o .pdf",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("📥 Ingestar documentos"):
    with st.sidebar:
        with st.spinner("Procesando e ingiriendo..."):
            ingest_embedder = get_embedder(embed_model)
            ingest_client = chromadb.PersistentClient(path=persist_dir)
            ingest_vdb = Chroma(
                client=ingest_client,
                collection_name=upload_collection,
                embedding_function=ingest_embedder,
            )

            all_docs: List[Document] = []
            for uf in uploaded_files:
                file_name = uf.name
                if file_name.lower().endswith(".txt"):
                    text = uf.read().decode("utf-8", errors="ignore")
                    meta = {"source": file_name, "rel_path": file_name, "chunk_id": None}
                    di = infer_date_iso(meta, None)
                    if di:
                        meta["date_iso"] = di
                    all_docs.append(Document(page_content=text, metadata=meta))

                elif file_name.lower().endswith(".pdf"):
                    if not HAS_PYPDF:
                        st.error("Se requiere 'pypdf' para PDFs. Instalá con: pip install pypdf")
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        reader = pypdf.PdfReader(tmp_path)
                        parts = []
                        for page in reader.pages:
                            page_text = (page.extract_text() or "").strip()
                            if page_text:
                                parts.append(page_text)
                        full_text = "\n\n".join(parts)
                        if not full_text.strip():
                            st.warning(f"⚠️ {file_name}: no se pudo extraer texto")
                            continue
                        meta = {
                            "source": file_name,
                            "rel_path": file_name,
                            "chunk_id": None,
                            "total_pages": len(reader.pages),
                        }
                        di = infer_date_iso(meta, None)
                        if di:
                            meta["date_iso"] = di
                        all_docs.append(Document(page_content=full_text, metadata=meta))
                    finally:
                        os.unlink(tmp_path)

            if not all_docs:
                st.warning("No se encontraron documentos válidos.")
            else:
                # Chunking
                if use_semantic_chunking:
                    semantic_splitter = SemanticChunker(
                        ingest_embedder,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=75,
                    )
                    chunked_docs: List[Document] = []
                    for doc in all_docs:
                        splits = semantic_splitter.split_text(doc.page_content)
                        for ci, piece in enumerate(splits):
                            md = dict(doc.metadata)
                            md["chunk_id"] = ci
                            md["chunk_total"] = len(splits)
                            chunked_docs.append(Document(page_content=piece, metadata=md))
                    all_docs = chunked_docs
                else:
                    all_docs = chunk_documents(all_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

                # Ingest into ChromaDB
                texts = [d.page_content for d in all_docs]
                metas = [d.metadata for d in all_docs]
                ids = [str(uuid.uuid4()) for _ in all_docs]

                batch_size = 500
                for i in range(0, len(texts), batch_size):
                    ingest_vdb.add_texts(
                        texts=texts[i:i+batch_size],
                        metadatas=metas[i:i+batch_size],
                        ids=ids[i:i+batch_size],
                    )

                st.success(f"✅ {len(all_docs)} chunks ingresados en '{upload_collection}'")

                # Clear caches so the new collection/docs appear
                get_vectordb.clear()
                st.rerun()

# ----- Chat render previo -----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # msg_idx -> "up" | "down"

# Suggested questions for onboarding (empty chat)
if not st.session_state.messages:
    st.markdown("##### 👋 ¡Hola! Preguntame sobre economía. Algunas ideas:")
    SUGGESTED = [
        "¿Cómo está la economía argentina esta semana?",
        "¿Qué decidió el BCE sobre tasas de interés?",
        "¿Cuál es la producción actual de la OPEP?",
        "¿Qué medidas de estímulo tomó China?",
        "Resumen de la economía de EE.UU.",
    ]
    cols_suggested = st.columns(len(SUGGESTED))
    for idx, q in enumerate(SUGGESTED):
        if cols_suggested[idx].button(q, key=f"suggested_{idx}", use_container_width=True):
            st.session_state["_suggested_q"] = q
            st.rerun()

for midx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            # Feedback buttons
            fb_key = str(midx)
            fcol1, fcol2, fcol3 = st.columns([0.06, 0.06, 0.88])
            current_fb = st.session_state.feedback.get(fb_key)
            if fcol1.button("👍" if current_fb != "up" else "✅", key=f"fb_up_{midx}", help="Buena respuesta"):
                st.session_state.feedback[fb_key] = "up"
                st.rerun()
            if fcol2.button("👎" if current_fb != "down" else "❌", key=f"fb_down_{midx}", help="Mala respuesta"):
                st.session_state.feedback[fb_key] = "down"
                st.rerun()
        if show_sources and m.get("sources"):
            with st.expander("📚 Fuentes"):
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
# Handle suggested question clicks
if "_suggested_q" in st.session_state:
    user_input = st.session_state.pop("_suggested_q")
else:
    user_input = st.chat_input("Escribí tu pregunta...")
if user_input and selected_cols:
    # mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Recuperación
    status_container = st.status("Procesando tu pregunta...", expanded=True)
    with status_container:
        st.write("🔍 Contextualizando consulta...")
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

        st.write("📡 Buscando en colecciones...")
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

        st.write(f"📊 Ordenando y seleccionando los mejores {k_total} fragmentos...")
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

        st.write("🤖 Generando respuesta...")
    status_container.update(label="✅ Contexto listo", state="complete", expanded=False)

    # LLM call with streaming
    with st.chat_message("assistant"):
        # Confidence indicator (before answer)
        if not confidence["confident"]:
            if confidence["reason"] == "no_results":
                st.warning("⚠️ No se encontraron fragmentos relevantes. La respuesta puede ser limitada.")
            elif confidence["reason"] == "low_relevance":
                st.info(f"ℹ️ Contexto parcial ({confidence['good_count']} fragmentos relevantes, "
                        f"similitud promedio: {confidence['avg_sim']:.2f}). "
                        "Considerá reformular o dar más detalles.")

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

            # Streaming response for better UX
            if render_plain:
                # Non-streaming fallback for plain text mode
                resp = client_oa.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                    temperature=temperature,
                )
                answer = resp.choices[0].message.content
                st.text(normalize_plain(answer))
            else:
                stream = client_oa.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                answer = st.write_stream(
                    (chunk.choices[0].delta.content or ""
                     for chunk in stream
                     if chunk.choices[0].delta.content is not None)
                )
        except Exception as e:
            answer = f"⚠️ Error llamando a OpenAI: {e}"
            st.error(answer)

        # Fuentes (built and displayed inside the same chat message)
        sources = []
        for i, (doc, dist, cscore) in enumerate(top, 1):
            tag = format_source_tag(doc.metadata)
            if show_scores:
                tag = f"{tag}\n  dist={dist:.4f}  score={cscore:.4f}"
            if show_preview:
                prev = short_preview(doc.page_content)
                tag = f"{tag}\n> {prev}"
            sources.append(f"[{i}] ({tag})")

        if show_sources and sources:
            with st.expander("📚 Fuentes"):
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
