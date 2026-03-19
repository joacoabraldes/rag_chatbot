# -*- coding: utf-8 -*-
import os, json, datetime, math, pathlib, uuid, tempfile, time, sys
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI, APITimeoutError, APIConnectionError
import re, unicodedata
import langdetect

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from rag_utils import (
    clean_keep_ascii_marks, escape_markdown, normalize_plain,
    extract_date_from_text, parse_date_iso, recency_score, sim_from_distance,
    combined_score, format_source_tag, short_preview, deduplicate_candidates,
    assess_confidence, build_confidence_note, load_prompt, save_history,
    expand_queries_llm, expand_queries_simple, split_k_across, now_iso,
    best_source_name, best_page, boost_keywords, build_context,
    contextualize_query, retrieve_from_collections_parallel,
    make_cache_key, get_cached_results, set_cached_results, clear_retrieval_cache,
    trim_history_to_budget, estimate_tokens, render_with_citations,
    generate_suggested_questions, KEYWORDS_BOOST,
)
from ingest import (
    chunk_documents, infer_date_iso, try_parse_date_from_string,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

# ========= Configuration (read from env — never shown in UI) =========

DEFAULT_PERSIST      = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBED        = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_OAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_PATH          = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_HISTORY      = "./history/chat_history.jsonl"
TEMPERATURE          = float(os.environ.get("TEMPERATURE", "0.0"))
RECENCY_WEIGHT       = float(os.environ.get("RECENCY_WEIGHT", "0.35"))
RECENCY_HALF_LIFE    = int(os.environ.get("RECENCY_HALF_LIFE", "30"))
FETCH_FACTOR         = float(os.environ.get("FETCH_FACTOR", "2.0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
MIN_GOOD_RESULTS     = int(os.environ.get("MIN_GOOD_RESULTS", "2"))
TOKEN_BUDGET         = int(os.environ.get("TOKEN_BUDGET", "2000"))
K_TOTAL              = int(os.environ.get("K_TOTAL", "10"))
USE_LLM_EXPANSION    = True
USE_MEMORY           = True
MAX_HISTORY_PAIRS    = 5

# ========= Spanish Error Messages =========

_ERROR_MESSAGES = {
    "chroma_connect": "No pudimos conectar con la base de datos de documentos. Por favor, verifica que el sistema este correctamente configurado.",
    "embed_load": "Hubo un problema al preparar el sistema de busqueda. Por favor, intenta de nuevo mas tarde.",
    "retrieval_fail": "No pudimos buscar en los documentos en este momento. Por favor, intenta de nuevo.",
    "llm_timeout": "El servicio tardo demasiado en responder. Por favor, intenta de nuevo en unos momentos.",
    "llm_connection": "No pudimos conectarnos al servicio de IA. Verifica tu conexion e intenta de nuevo.",
    "llm_generic": "Ocurrio un error inesperado al generar la respuesta. Por favor, intenta de nuevo.",
    "ingest_init": "No pudimos preparar el sistema para procesar los documentos. Intenta de nuevo mas tarde.",
    "ingest_chunk": "Hubo un problema al dividir el documento en fragmentos. Intenta con otro archivo.",
    "ingest_batch": "Ocurrio un error al guardar los fragmentos. Algunos documentos pueden no haberse procesado.",
    "pdf_missing": "Se requiere la libreria 'pypdf' para procesar archivos PDF. Contacta al administrador.",
    "no_text_extracted": "No se pudo extraer texto del archivo. Verifica que el PDF contenga texto legible.",
    "no_collections": "No hay colecciones de documentos disponibles. Subi documentos para comenzar.",
    "no_selection": "Selecciona al menos una fuente de documentos en el panel lateral.",
    "generic": "Ocurrio un error inesperado. Por favor, intenta de nuevo.",
}


def _friendly_error(key: str, technical_detail: str = "") -> str:
    """Return a friendly Spanish message. Log technical detail to stderr."""
    if technical_detail:
        print(f"[ERROR:{key}] {technical_detail}", file=sys.stderr)
    return _ERROR_MESSAGES.get(key, _ERROR_MESSAGES["generic"])


# ========= App =========

st.set_page_config(page_title="RAG Analyst", layout="wide")

# Finance-themed CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #1a365d;
    --primary-light: #2c5282;
    --accent: #c09f50;
    --accent-light: #d4b96a;
    --bg: #f7fafc;
    --surface: #ffffff;
    --text: #1a202c;
    --text-muted: #718096;
    --success: #38a169;
    --warning: #d69e2e;
    --error: #e53e3e;
    --border: #e2e8f0;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.stApp > header { background: transparent; }

.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.stChatMessage {
    max-width: 92%;
    border-radius: 12px;
}

/* Suggested question pills */
.suggested-q {
    padding: 8px 18px;
    border: 1px solid var(--border);
    border-radius: 24px;
    display: inline-block;
    margin: 4px 6px;
    cursor: pointer;
    font-size: 0.88rem;
    color: var(--primary);
    background: var(--surface);
    transition: all 0.2s ease;
    font-weight: 500;
}
.suggested-q:hover {
    background: #fdf6e3;
    border-color: var(--accent);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
    border-right: 2px solid var(--accent);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary);
}

/* Status containers */
div[data-testid="stStatus"] {
    border-radius: 10px;
    border-left: 3px solid var(--accent);
}

/* Buttons */
.stButton > button {
    border: 1.5px solid var(--primary);
    color: var(--primary);
    background: transparent;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--primary);
}

/* Welcome card */
.welcome-card {
    text-align: center;
    padding: 2.5rem 2rem;
    background: linear-gradient(135deg, var(--surface) 0%, #f0f4f8 100%);
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}
.welcome-card h2 {
    color: var(--primary);
    margin-bottom: 0.5rem;
    font-size: 1.6rem;
}
.welcome-card p {
    color: var(--text-muted);
    font-size: 1rem;
    margin-bottom: 0;
}
.welcome-card .icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

/* Source cards */
.source-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    background: var(--surface);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.source-card .source-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.source-card .source-idx {
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--primary);
}
.source-card .source-name {
    font-size: 0.88rem;
    color: var(--text);
    font-weight: 500;
}
.source-card .source-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
}
.source-card .source-preview {
    margin-top: 6px;
    padding: 6px 10px;
    background: #f8f9fa;
    border-left: 3px solid var(--accent);
    font-size: 0.82rem;
    color: #555;
    border-radius: 0 4px 4px 0;
    line-height: 1.4;
}

/* Confidence badge */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Sidebar brand */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.sidebar-brand h2 {
    color: var(--primary);
    font-size: 1.3rem;
    margin: 0;
    font-weight: 700;
}
.sidebar-brand .brand-icon {
    font-size: 2rem;
    margin-bottom: 4px;
}
.sidebar-brand .brand-sub {
    color: var(--text-muted);
    font-size: 0.78rem;
}

/* Footer */
.sidebar-footer {
    text-align: center;
    padding: 1rem 0;
    color: var(--text-muted);
    font-size: 0.72rem;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}

/* Feedback buttons */
.feedback-btn {
    display: inline-block; cursor: pointer; font-size: 1.2rem;
    margin-right: 6px; opacity: 0.55; transition: opacity 0.15s, transform 0.1s;
}
.feedback-btn:hover { opacity: 1; transform: scale(1.15); }

/* Responsive */
@media (max-width: 768px) {
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    .welcome-card { padding: 1.5rem 1rem; }
    .welcome-card h2 { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)

# ========= Sidebar =========

# Branded header
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="brand-icon">📊</div>
    <h2>RAG Analyst</h2>
    <div class="brand-sub">Inteligencia para economia y finanzas</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# ChromaDB connection
try:
    client = chromadb.PersistentClient(path=DEFAULT_PERSIST)
    collections = [c.name for c in client.list_collections()]
except Exception as e:
    st.sidebar.error(_friendly_error("chroma_connect", str(e)))
    collections = []

if not collections:
    st.sidebar.warning(_ERROR_MESSAGES["no_collections"])

selected_cols = st.sidebar.multiselect(
    "Fuentes de documentos",
    options=collections,
    default=collections[:1] if collections else [],
)

# ----- Caches -----
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    try:
        return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
    except Exception as e:
        st.error(_friendly_error("embed_load", str(e)))
        return None

@st.cache_resource(show_spinner=False)
def get_vectordb(persist_path: str, collection_name: str, embedder):
    try:
        return Chroma(client=chromadb.PersistentClient(path=persist_path),
                      collection_name=collection_name,
                      embedding_function=embedder)
    except Exception as e:
        st.error(_friendly_error("chroma_connect", f"Collection '{collection_name}': {e}"))
        return None

embedder = get_embedder(DEFAULT_EMBED)
vectordbs = {}
if embedder is not None:
    for col in selected_cols:
        vdb = get_vectordb(DEFAULT_PERSIST, col, embedder)
        if vdb is not None:
            vectordbs[col] = vdb

st.sidebar.markdown("---")

# ----- Clear / Export chat -----
col_clear, col_export = st.sidebar.columns(2)
if col_clear.button("🗑️ Limpiar chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.pop("feedback", None)
    st.rerun()
if col_export.button("📤 Exportar chat", use_container_width=True):
    if st.session_state.get("messages"):
        export_msgs = []
        for m in st.session_state.messages:
            entry = {"timestamp": now_iso(), "role": m["role"], "content": m["content"]}
            if m.get("sources"):
                entry["sources"] = m["sources"]
            export_msgs.append(entry)
        export_data = json.dumps(export_msgs, ensure_ascii=False, indent=2, default=str)
        st.sidebar.download_button(
            label="⬇️ Descargar JSON",
            data=export_data,
            file_name=f"chat_export_{datetime.datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.sidebar.info("No hay mensajes para exportar.")

st.sidebar.markdown("---")

# ----- File Upload & Ingestion -----
st.sidebar.markdown("### 📄 Subir documentos")
upload_collection = collections[0] if collections else "docs"
uploaded_files = st.sidebar.file_uploader(
    "Subi archivos .txt o .pdf",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("📥 Procesar documentos"):
    with st.sidebar:
        with st.spinner("Procesando documentos..."):
            try:
                ingest_embedder = get_embedder(DEFAULT_EMBED)
                if ingest_embedder is None:
                    st.error(_friendly_error("embed_load"))
                    st.stop()
                ingest_client = chromadb.PersistentClient(path=DEFAULT_PERSIST)
                ingest_vdb = Chroma(
                    client=ingest_client,
                    collection_name=upload_collection,
                    embedding_function=ingest_embedder,
                )
            except Exception as e:
                st.error(_friendly_error("ingest_init", str(e)))
                st.stop()

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
                        st.error(_friendly_error("pdf_missing"))
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
                            st.warning(f"⚠️ {file_name}: {_ERROR_MESSAGES['no_text_extracted']}")
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
                st.warning("No se encontraron documentos validos.")
            else:
                # Semantic chunking (always enabled)
                try:
                    semantic_splitter = SemanticChunker(
                        ingest_embedder,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=75,
                    )
                except Exception as e:
                    st.error(_friendly_error("ingest_chunk", str(e)))
                    st.stop()
                chunked_docs: List[Document] = []
                for doc in all_docs:
                    try:
                        splits = semantic_splitter.split_text(doc.page_content)
                    except Exception as e:
                        print(f"[ingest_chunk] {doc.metadata.get('source', '?')}: {e}", file=sys.stderr)
                        st.warning(f"No se pudo procesar '{doc.metadata.get('source', '?')}'. Se omitio.")
                        continue
                    for ci, piece in enumerate(splits):
                        md = dict(doc.metadata)
                        md["chunk_id"] = ci
                        md["chunk_total"] = len(splits)
                        chunked_docs.append(Document(page_content=piece, metadata=md))
                all_docs = chunked_docs

                # Ingest into ChromaDB
                texts = [d.page_content for d in all_docs]
                metas = [d.metadata for d in all_docs]
                ids = [str(uuid.uuid4()) for _ in all_docs]

                batch_size = 500
                ingested_count = 0
                ingest_error = None
                for i in range(0, len(texts), batch_size):
                    try:
                        ingest_vdb.add_texts(
                            texts=texts[i:i+batch_size],
                            metadatas=metas[i:i+batch_size],
                            ids=ids[i:i+batch_size],
                        )
                        ingested_count += len(texts[i:i+batch_size])
                    except Exception as e:
                        ingest_error = e
                        st.error(_friendly_error("ingest_batch", str(e)))
                        break

                if ingested_count > 0:
                    st.success(f"✅ {ingested_count} fragmentos procesados y guardados correctamente.")
                elif ingest_error is None:
                    st.warning("No se procesaron fragmentos.")

                # Clear caches so the new collection/docs appear
                get_vectordb.clear()
                clear_retrieval_cache()
                st.rerun()

# Sidebar footer
st.sidebar.markdown("""
<div class="sidebar-footer">
    RAG Analyst · Inteligencia documental<br>
    Potenciado por IA
</div>
""", unsafe_allow_html=True)

# ----- Chat state -----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ----- Title -----
st.title("RAG Analyst")

# ----- Welcome screen (empty chat) -----
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">📊</div>
        <h2>Bienvenido a RAG Analyst</h2>
        <p>Tu asistente de inteligencia para economia y finanzas.<br>
        Pregunta lo que necesites sobre tus documentos.</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_suggested(col_key: tuple) -> list:
        return generate_suggested_questions(vectordbs, openai_model=DEFAULT_OAI_MODEL) if col_key else []

    SUGGESTED = _get_suggested(tuple(sorted(selected_cols))) if vectordbs else [
        "Cuales son las ultimas decisiones de tasas del banco central?",
        "Resumime las condiciones macroeconomicas actuales.",
        "Cual es la perspectiva de inflacion?",
        "Factores de riesgo clave en los informes mas recientes?",
        "Compara la politica monetaria de las principales economias.",
    ]
    for row_start in range(0, len(SUGGESTED), 3):
        row_items = SUGGESTED[row_start:row_start + 3]
        cols_suggested = st.columns(len(row_items))
        for i, q in enumerate(row_items):
            if cols_suggested[i].button(q, key=f"suggested_{row_start + i}", use_container_width=True):
                st.session_state["_suggested_q"] = q
                st.rerun()

# ----- Render chat history -----
for midx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            fb_key = str(midx)
            fcol1, fcol2, fcol3 = st.columns([0.06, 0.06, 0.88])
            current_fb = st.session_state.feedback.get(fb_key)
            if fcol1.button("👍" if current_fb != "up" else "✅", key=f"fb_up_{midx}", help="Buena respuesta"):
                st.session_state.feedback[fb_key] = "up"
                st.rerun()
            if fcol2.button("👎" if current_fb != "down" else "❌", key=f"fb_down_{midx}", help="Mala respuesta"):
                st.session_state.feedback[fb_key] = "down"
                st.rerun()
        if m.get("sources"):
            with st.expander("📚 Fuentes"):
                for s in m["sources"]:
                    st.markdown(s)


# ----- Helper: build conversation history for OpenAI -----
def build_chat_messages(system_prompt: str, user_input: str, max_pairs: int, tok_budget: int = 2000) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    history = st.session_state.messages
    recent = history[-(max_pairs * 2):]
    recent = trim_history_to_budget(recent, token_budget=tok_budget)
    for msg in recent:
        role = msg["role"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    return messages


# ----- Interaction -----
# Always render chat_input so the widget is visible after every run
_chat_value = st.chat_input("Hace tu pregunta sobre los documentos...")

if "_suggested_q" in st.session_state:
    user_input = st.session_state.pop("_suggested_q")
else:
    user_input = _chat_value

if user_input and selected_cols:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    t_start = time.perf_counter()

    # Retrieval
    _retrieval_error = None
    top = []
    system_prompt = ""
    confidence = {"confident": False, "reason": "no_results", "avg_sim": 0.0, "good_count": 0}
    confidence_note = ""
    status_container = st.status("Procesando tu pregunta...", expanded=True)
    with status_container:
        try:
            st.write("🔍 Analizando tu pregunta...")
            search_query = user_input
            if USE_MEMORY and st.session_state.messages:
                search_query = contextualize_query(
                    user_input,
                    st.session_state.messages,
                    openai_model=DEFAULT_OAI_MODEL,
                    max_pairs=MAX_HISTORY_PAIRS,
                )

            target_iso = extract_date_from_text(search_query)
            target_date = None
            if target_iso:
                try:
                    y, m, d = map(int, target_iso.split("-"))
                    target_date = datetime.date(y, m, d)
                except Exception:
                    target_date = None

            st.write("📡 Buscando en los documentos...")
            ks = split_k_across(len(selected_cols), K_TOTAL)
            fetch_k_each = [max(5, int(k * FETCH_FACTOR)) for k in ks]

            if USE_LLM_EXPANSION:
                queries = expand_queries_llm(search_query, openai_model=DEFAULT_OAI_MODEL)
            else:
                queries = [search_query]

            cache_key = make_cache_key(search_query, selected_cols, K_TOTAL, RECENCY_WEIGHT)
            cached = get_cached_results(cache_key)
            if cached is not None:
                candidates_all = cached
            else:
                candidates_all = retrieve_from_collections_parallel(
                    vectordbs, queries, fetch_k_each
                )
                set_cached_results(cache_key, candidates_all)

            candidates_all = deduplicate_candidates(candidates_all)

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
                cscore = combined_score(dist, d_iso, today, RECENCY_WEIGHT, RECENCY_HALF_LIFE)
                cscore += boost_keywords(doc.page_content)
                scored.append((doc, dist, cscore))

            st.write(f"📊 Seleccionando los fragmentos mas relevantes...")
            scored.sort(key=lambda x: x[2], reverse=True)
            top = scored[:K_TOTAL]
            top_docs = [doc for (doc, _, _) in top]

            context = "\n\n".join(
                clean_keep_ascii_marks(f"[{i}] ({format_source_tag(doc.metadata)})\n{doc.page_content}")
                for i, doc in enumerate(top_docs, 1)
            )

            confidence = assess_confidence(top, sim_threshold=CONFIDENCE_THRESHOLD,
                                           min_good_results=MIN_GOOD_RESULTS)
            confidence_note = build_confidence_note(confidence)

            try:
                system_prompt = load_prompt(PROMPT_PATH).format(
                    context=context, question=user_input
                ) + confidence_note
            except Exception:
                system_prompt = f"Context:\n{context}\n\nQuestion: {user_input}" + confidence_note

            st.write("🤖 Generando la respuesta...")
        except Exception as e:
            _retrieval_error = e
            st.error(_friendly_error("retrieval_fail", str(e)))
    t_retrieval = time.perf_counter()
    if _retrieval_error is not None:
        status_container.update(label="❌ Error en la busqueda", state="error", expanded=True)
    else:
        status_container.update(label="✅ Listo", state="complete", expanded=False)

    # LLM call with streaming
    if _retrieval_error is None:
        with st.chat_message("assistant"):
            # Confidence badge (simplified Spanish labels)
            def _confidence_badge(conf: dict) -> str:
                rsn = conf.get("reason", "")
                if conf["confident"] and conf.get("avg_sim", 0) >= 0.6:
                    bg, fg, icon = "#d4edda", "#1a7f4e", "●"
                    label = "Alta confianza en las fuentes"
                elif conf["confident"]:
                    bg, fg, icon = "#fff3cd", "#856404", "◐"
                    label = "Confianza moderada en las fuentes"
                elif rsn == "no_results":
                    bg, fg, icon = "#f8d7da", "#721c24", "○"
                    label = "No se encontraron fuentes relevantes"
                else:
                    bg, fg, icon = "#f8d7da", "#721c24", "○"
                    label = "Pocas fuentes relevantes"
                return (
                    f'<div class="confidence-badge" style="'
                    f'background:{bg};color:{fg};border:1px solid {fg}33">'
                    f'<span>{icon}</span><span>{label}</span></div>'
                )
            st.markdown(_confidence_badge(confidence), unsafe_allow_html=True)

            try:
                client_oa = OpenAI(timeout=60.0)

                if USE_MEMORY:
                    messages = build_chat_messages(system_prompt, user_input, MAX_HISTORY_PAIRS, TOKEN_BUDGET)
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ]

                stream = client_oa.chat.completions.create(
                    model=DEFAULT_OAI_MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    stream=True,
                )
                stream_placeholder = st.empty()

                def _safe_stream(s):
                    for chunk in s:
                        try:
                            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                        except Exception:
                            continue

                answer = st.write_stream(_safe_stream(stream))
            except APITimeoutError:
                answer = _ERROR_MESSAGES["llm_timeout"]
                st.error(f"⏱️ {_ERROR_MESSAGES['llm_timeout']}")
            except APIConnectionError:
                answer = _ERROR_MESSAGES["llm_connection"]
                st.error(f"🔌 {_ERROR_MESSAGES['llm_connection']}")
            except Exception as e:
                print(f"[OpenAI error] {type(e).__name__}: {e}", file=sys.stderr)
                answer = _ERROR_MESSAGES["llm_generic"]
                st.error(_ERROR_MESSAGES["llm_generic"])

            t_total = time.perf_counter()

            # Sources — structured data
            sources_data = []
            for i, (doc, dist, cscore) in enumerate(top, 1):
                md = doc.metadata
                sources_data.append({
                    "index": i,
                    "fname": best_source_name(md),
                    "page": best_page(md),
                    "date_iso": md.get("date_iso", ""),
                    "preview": short_preview(doc.page_content),
                })

            # Plain-text source strings for citations and session state
            sources = []
            for sd in sources_data:
                line = f"[{sd['index']}] {sd['fname']}"
                if sd["page"]:
                    line += f" p.{sd['page']}"
                if sd["date_iso"]:
                    line += f" ({sd['date_iso']})"
                sources.append(line)

            # Inline citation superscripts
            if sources and isinstance(answer, str):
                cited = render_with_citations(answer, sources)
                if cited != answer:
                    stream_placeholder.markdown(cited, unsafe_allow_html=True)

            # Source cards (simplified)
            if sources_data:
                def _render_source_card(sd: dict) -> str:
                    page_str = (f"<span class='source-meta'>p.{sd['page']}</span>"
                                if sd["page"] else "")
                    date_str = (f"<span class='source-meta'>{sd['date_iso']}</span>"
                                if sd["date_iso"] else "")
                    preview_html = (
                        f"<div class='source-preview'>{sd['preview']}</div>"
                        if sd["preview"] else ""
                    )
                    return (
                        f"<div class='source-card'>"
                        f"<div class='source-header'>"
                        f"<span class='source-idx'>[{sd['index']}]</span>"
                        f"<span class='source-name'>{sd['fname']}</span>"
                        f"{page_str}{date_str}"
                        f"</div>"
                        f"{preview_html}</div>"
                    )

                with st.expander("📚 Fuentes", expanded=False):
                    html_cards = "".join(_render_source_card(sd) for sd in sources_data)
                    st.markdown(html_cards, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "collections": selected_cols,
            "confidence": confidence,
            "timing": {
                "retrieval": t_retrieval - t_start,
                "generation": t_total - t_retrieval,
            },
        })

        # Save history (clean record)
        record = {
            "timestamp": now_iso(),
            "query": user_input,
            "answer": answer,
            "sources": sources,
        }
        try:
            save_history(DEFAULT_HISTORY, record)
        except Exception:
            pass
    else:
        answer = _friendly_error("retrieval_fail", str(_retrieval_error))
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": [], "collections": selected_cols})

elif user_input and not selected_cols:
    st.warning(_ERROR_MESSAGES["no_selection"])
