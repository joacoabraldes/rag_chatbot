# -*- coding: utf-8 -*-
import os, json, datetime, math
from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import re, unicodedata
import langdetect

# ========= Utils =========
import unicodedata, re

def clean_text(s: str) -> str:
    # normaliza unicode y limpia caracteres que rompen el render
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")       # NBSP
    s = s.replace("\u00AD", "")        # soft hyphen
    s = s.replace("\u200B", "")        # zero width space
    s = re.sub(r"[ \t]+", " ", s)      # colapsa espacios
    # evita que ## o * activen markdown si terminás usando markdown
    s = s.replace("#", "＃").replace("*", "＊").replace("_", "﹎")
    return s

def clean_keep_ascii_marks(s: str) -> str:
    # versión que solo normaliza sin reemplazar #/*/_ (útil para markdown escapado)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u00AD", "").replace("\u200B", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s

_MD_SPECIALS = r"\`*_{}[]()#+\-.!|>"

def escape_markdown(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>])", r"\\\1", s)

def normalize_plain(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return s
""
DATE_TEXT_PATTERNS = [
    # 15-10-2025 / 15/10/2025 / 15.10.2025
    re.compile(r"(?P<d>[0-3]?\d)[\-\/\.](?P<m>[01]?\d)[\-\/\.](?P<y>\d{4})"),
    # 2025-10-15
    re.compile(r"(?P<y>\d{4})[\-\/\.](?P<m>[01]?\d)[\-\/\.](?P<d>[0-3]?\d)"),
]

def normalize_iso(y:int,m:int,d:int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"

def extract_date_from_text(text: str) -> Optional[str]:
    t = text.lower()
    # nombres de meses en español
    meses = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
             "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,
             "noviembre":11,"diciembre":12}
    # 1) números
    for pat in DATE_TEXT_PATTERNS:
        m = pat.search(t)
        if m:
            y = int(m.group("y")); mth = int(m.group("m")); d = int(m.group("d"))
            return normalize_iso(y,mth,d)
    # 2) “15 de octubre de 2025”
    m = re.search(r"(?P<d>[0-3]?\d)\s+de\s+(?P<mes>\w+)\s+de\s+(?P<y>\d{4})", t)
    if m:
        d = int(m.group("d")); y=int(m.group("y"))
        mes_name = m.group("mes")
        if mes_name in meses:
            return normalize_iso(y, meses[mes_name], d)
    return None

KEYWORDS_ANY = ["mep","ccl","mulc","oficial","contado con liqui","dólar","dolar","blue","brecha"]
def keyword_score(text: str) -> int:
    tl = text.lower()
    return sum(1 for kw in KEYWORDS_ANY if kw in tl)

def now_iso() -> str:
    return datetime.datetime.now().isoformat()

def load_prompt(path: str) -> str:
    """Plantilla segura: sintetiza SOLO con el contexto; si falta info, resp. fija."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return (
            "Eres un asistente que responde SOLO con el CONTEXTO provisto.\n"
            "Puedes sintetizar y combinar fragmentos del contexto.\n"
            "Si falta información crítica para responder con precisión, di exactamente:\n"
            "\"No se puede responder con el contexto disponible.\"\n\n"
            "=== CONTEXTO ===\n{context}\n\n=== PREGUNTA ===\n{question}\n"
        )

SOURCE_KEYS = ["rel_path", "source", "file", "filename", "doc", "title", "name", "path", "uri", "url", "id"]
PAGE_KEYS   = ["page", "page_number", "pageIndex", "page_num", "pageno", "pageNo"]

def best_source_name(md: Dict[str, Any]) -> str:
    import os as _os
    for k in SOURCE_KEYS:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            try:
                return _os.path.basename(v) or v
            except Exception:
                return v
    return "desconocido"

def best_page(md: Dict[str, Any]) -> Optional[str]:
    for k in PAGE_KEYS:
        v = md.get(k)
        if v not in (None, ""):
            return str(v)
    return None

def parse_date_iso(md: Dict[str, Any]) -> Optional[datetime.date]:
    s = md.get("date_iso")
    if not isinstance(s, str):
        return None
    try:
        y, m, d = map(int, s[:10].split("-"))
        return datetime.date(y, m, d)
    except Exception:
        return None

def recency_score(d: Optional[datetime.date], today: datetime.date, half_life_days: int) -> float:
    """1.0 = hoy, decae exponencialmente con 'half_life_days'."""
    if not d:
        return 0.0
    age_days = max(0, (today - d).days)
    return math.exp(-age_days / max(1, half_life_days))

def sim_from_distance(dist: float) -> float:
    """convierte distancia (menor=mejor) en similitud ~[0..1)."""
    return 1.0 / (1.0 + dist)

def combined_score(dist: float, date_: Optional[datetime.date], today: datetime.date,
                   weight: float, half_life_days: int) -> float:
    """score final = (1-w)*sim + w*recency (mayor es mejor)."""
    s = sim_from_distance(dist)
    r = recency_score(date_, today, half_life_days)
    w = max(0.0, min(1.0, weight))
    return (1.0 - w) * s + w * r

def format_source_tag(md: Dict[str, Any]) -> str:
    col = md.get("_collection", "?")
    name = best_source_name(md)
    page = best_page(md)
    chunk_id = md.get("chunk_id")
    start = md.get("chunk_start"); end = md.get("chunk_end")
    tag = f"{col}) {name}"
    if page: tag += f"#{page}"
    if chunk_id is not None:
        tag += f"  [chunk:{chunk_id}"
        if start is not None and end is not None:
            tag += f" @ {start}:{end}"
        tag += "]"
    d = md.get("date_iso")
    if d: tag += f"  (fecha: {d})"
    return tag

def short_preview(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return (t[:n] + "…") if len(t) > n else t

def similarity_search_with_score(vdb: Chroma, query: str, k: int):
    return vdb.similarity_search_with_score(query, k=k)

def split_k_across(n_items: int, k: int) -> List[int]:
    base = k // n_items; rem = k % n_items
    sizes = [base] * n_items
    for i in range(rem): sizes[i] += 1
    return sizes

def save_history(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def expand_queries(q: str) -> list[str]:
    """Expansión simple para español (sinónimos/variantes macro)."""
    base = q.strip()
    expansions = {base}
    lower = base.lower()
    if "régimen" in lower and ("monetario" in lower or "cambiario" in lower):
        expansions |= {
            lower.replace("monetario", "cambiario"),
            lower.replace("régimen", "esquema"),
            "crawling peg", "crawling-peg", "deslizamiento cambiario",
            "bandas cambiarias", "ancla nominal", "programa monetario",
            "liberalización cambiaria", "CEPO", "crawling",
        }
    if "abril" in lower:
        expansions |= {"mediados de abril", "segunda quincena de abril"}
    return [base] + [e for e in expansions if e != base]

# ========= App =========

# Título que pediste
st.set_page_config(page_title="PARROT RAG", layout="wide")
st.title("🦜 PARROT RAG — say stupidities in a fancy way")

DEFAULT_PERSIST   = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBED     = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_PATH       = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
DEFAULT_HISTORY   = "./history/chat_history.jsonl"

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
use_query_expansion   = st.sidebar.checkbox("Usar expansión de consulta", value=True)

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
                for s in sources:
                    st.text(clean_text(s))


# ----- Interacción -----
user_input = st.chat_input("Escribí tu pregunta...")
if user_input and selected_cols:
    # mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Recuperación
    with st.spinner("Buscando contexto..."):
        target_iso = extract_date_from_text(user_input)
        target_date = None
        if target_iso:
            try:
                y, m, d = map(int, target_iso.split("-"))
                target_date = datetime.date(y, m, d)
            except Exception:
                target_date = None

        ks = split_k_across(len(selected_cols), k_total)
        fetch_k_each = [max(5, int(k * fetch_factor)) for k in ks]
        candidates_all = []

        for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
            if k_each <= 0:
                continue
            for qexp in expand_queries(user_input) if use_query_expansion else [user_input]:
                for doc, dist in vdb.similarity_search_with_score(qexp, k=k_each):
                    md = dict(doc.metadata or {})
                    md["_collection"] = col
                    doc.metadata = md
                    candidates_all.append((doc, dist))

        filtered = []
        if target_date:
            window = 3
            for doc, dist in candidates_all:
                d_iso = parse_date_iso(doc.metadata)
                if d_iso and abs((d_iso - target_date).days) <= window:
                    filtered.append((doc, dist))
        else:
            filtered = candidates_all

        def boost_keywords(text: str) -> float:
            boost_terms = ["mep", "ccl", "mulc", "tipo de cambio", "dólar", "cotización", "oficial"]
            text_l = text.lower()
            return 0.10 * sum(kw in text_l for kw in boost_terms)

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

        # context = "\n\n".join(
        #     f"[{i}] ({format_source_tag(doc.metadata)})\n{doc.page_content}" for i, doc in enumerate(top_docs, 1)
        # )
        system_prompt = load_prompt(PROMPT_PATH).format(context=context, question=user_input)
        # try:
        #     user_lang = langdetect.detect(user_input)
        # except:
        #     user_lang = "es"

        # lang_instructions = {
        #     "en": "Answer in English, keeping the same tone and style as the question.",
        #     "es": "Responde en español, manteniendo el mismo tono y estilo de la pregunta."
        # }

        # system_prompt = (
        #     load_prompt(PROMPT_PATH).format(context=context, question=user_input)
        #     + "\n\n"
        #     + lang_instructions.get(user_lang, lang_instructions["es"])
        # )


    # LLM OpenAI (system_prompt ya existe en este scope)
    with st.spinner("Consultando OpenAI..."):
        try:
            client_oa = OpenAI()
            resp = client_oa.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=temperature
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
        if render_plain:
            st.text(normalize_plain(answer))  # texto plano
        else:
            st.markdown(escape_markdown(answer))  # markdown escapado
        if show_sources and sources:
            with st.expander("Fuentes"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "collections": selected_cols
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
    }
    save_history(history_path, record)

elif user_input and not selected_cols:
    st.warning("Seleccioná al menos una colección en la barra lateral.")
