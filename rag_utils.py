# -*- coding: utf-8 -*-
"""
Shared utilities for the RAG chatbot.
Centralises functions used by query.py, app_streamlit.py and server.py.
"""

import os, sys, json, datetime, math, re, unicodedata, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from openai import OpenAI, APITimeoutError, APIConnectionError

# ─── Date helpers ────────────────────────────────────────────────────────────

DATE_TEXT_PATTERNS = [
    # DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
    re.compile(r"(?P<d>[0-3]?\d)[-/.](?P<m>[01]?\d)[-/.](?P<y>\d{4})"),
    # YYYY-MM-DD
    re.compile(r"(?P<y>\d{4})[-/.](?P<m>[01]?\d)[-/.](?P<d>[0-3]?\d)"),
]

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def normalize_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_date_from_text(text: str) -> Optional[str]:
    """Try to find a date inside free text (numeric patterns + Spanish month names)."""
    t = text.lower()
    for pat in DATE_TEXT_PATTERNS:
        hit = pat.search(t)
        if hit:
            y, mth, d = int(hit.group("y")), int(hit.group("m")), int(hit.group("d"))
            try:
                datetime.date(y, mth, d)
                return normalize_iso(y, mth, d)
            except ValueError:
                continue
    # "15 de octubre de 2025"
    hit = re.search(r"(?P<d>[0-3]?\d)\s+de\s+(?P<mes>\w+)\s+de\s+(?P<y>\d{4})", t)
    if hit:
        d = int(hit.group("d")); y = int(hit.group("y"))
        mes_name = hit.group("mes")
        if mes_name in MESES:
            return normalize_iso(y, MESES[mes_name], d)
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


# ─── Scoring helpers ─────────────────────────────────────────────────────────

def recency_score(d: Optional[datetime.date], today: datetime.date, half_life_days: int) -> float:
    """1.0 = today, exponential decay with *half_life_days*."""
    if not d:
        return 0.0
    age_days = max(0, (today - d).days)
    return math.exp(-age_days / max(1, half_life_days))


def sim_from_distance(dist: float) -> float:
    """Convert distance (lower = better) to similarity ~[0..1)."""
    return 1.0 / (1.0 + dist)


def combined_score(
    dist: float,
    date_: Optional[datetime.date],
    today: datetime.date,
    weight: float,
    half_life_days: int,
) -> float:
    """Final score = (1-w)*similarity + w*recency (higher is better)."""
    s = sim_from_distance(dist)
    r = recency_score(date_, today, half_life_days)
    w = max(0.0, min(1.0, weight))
    return (1.0 - w) * s + w * r


# ─── Text helpers ────────────────────────────────────────────────────────────

def short_preview(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return (t[:n] + "…") if len(t) > n else t


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u00AD", "").replace("\u200B", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("#", "＃").replace("*", "＊").replace("_", "﹎")
    return s


def clean_keep_ascii_marks(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u00AD", "").replace("\u200B", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def escape_markdown(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>])", r"\\\1", s)


def normalize_plain(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return s


# ─── Source / page helpers ───────────────────────────────────────────────────

SOURCE_KEYS = ["rel_path", "source", "file", "filename", "doc", "title", "name", "path", "uri", "url", "id"]
PAGE_KEYS = ["page", "page_number", "pageIndex", "page_num", "pageno", "pageNo"]


def best_source_name(md: Dict[str, Any]) -> str:
    for k in SOURCE_KEYS:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            try:
                return os.path.basename(v) or v
            except Exception:
                return v
    return "desconocido"


def best_page(md: Dict[str, Any]) -> Optional[str]:
    for k in PAGE_KEYS:
        v = md.get(k)
        if v not in (None, ""):
            return str(v)
    return None


def format_source_tag(md: Dict[str, Any]) -> str:
    col = md.get("_collection", "?")
    name = best_source_name(md)
    page = best_page(md)
    chunk_id = md.get("chunk_id")
    start = md.get("chunk_start")
    end = md.get("chunk_end")
    tag = f"{col}) {name}"
    if page:
        tag += f"#{page}"
    if chunk_id is not None:
        tag += f"  [chunk:{chunk_id}"
        if start is not None and end is not None:
            tag += f" @ {start}:{end}"
        tag += "]"
    d = md.get("date_iso")
    if d:
        tag += f"  (fecha: {d})"
    return tag


# ─── Prompt loading ─────────────────────────────────────────────────────────

_FALLBACK_PROMPT = (
    "Eres un asistente que responde SOLO con el CONTEXTO provisto.\n"
    "Puedes sintetizar y combinar fragmentos del contexto.\n"
    "Si falta información crítica para responder con precisión, di exactamente:\n"
    "\"No se puede responder con el contexto disponible.\"\n\n"
    "=== CONTEXTO ===\n{context}\n\n=== PREGUNTA ===\n{question}\n"
)


def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return _FALLBACK_PROMPT


# ─── Deduplication ───────────────────────────────────────────────────────────

def deduplicate_candidates(candidates: list, content_window: int = 150) -> list:
    """Deduplicate chunks by content + source to avoid repeated results."""
    seen = set()
    unique = []
    for doc, dist in candidates:
        key = (
            doc.page_content[:content_window],
            doc.metadata.get("rel_path", ""),
            doc.metadata.get("chunk_id", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append((doc, dist))
    return unique


# ─── Confidence assessment ──────────────────────────────────────────────────

def assess_confidence(
    top_results: list,
    sim_threshold: float = 0.35,
    min_good_results: int = 2,
) -> dict:
    if not top_results:
        return {"confident": False, "reason": "no_results", "avg_sim": 0.0, "good_count": 0}
    sims = [sim_from_distance(dist) for (_, dist, _) in top_results]
    avg_sim = sum(sims) / len(sims)
    good_count = sum(1 for s in sims if s >= sim_threshold)
    if good_count < min_good_results:
        return {"confident": False, "reason": "low_relevance", "avg_sim": avg_sim, "good_count": good_count}
    return {"confident": True, "reason": "ok", "avg_sim": avg_sim, "good_count": good_count}


def build_confidence_note(conf: dict) -> str:
    if conf["confident"]:
        return ""
    if conf["reason"] == "no_results":
        return (
            "\n\n⚠️ NOTA INTERNA: No se encontraron resultados relevantes en los documentos. "
            "Indicá que no hay información disponible y preguntá "
            "al usuario si puede reformular la pregunta o dar más detalles "
            "(período, región, indicador específico, etc.)."
        )
    if conf["reason"] == "low_relevance":
        return (
            f"\n\n⚠️ NOTA INTERNA: Solo {conf['good_count']} fragmentos tienen "
            f"relevancia aceptable (similitud promedio: {conf['avg_sim']:.2f}). "
            "La información podría ser parcial. Respondé con lo que tengas "
            "pero advertí al usuario y pedile que precise su consulta "
            "(ej: país, período, indicador específico)."
        )
    return ""


# ─── Query contextualisation (multi-turn → self-contained query) ────────────

_CONTEXTUALISE_SYSTEM = (
    "You receive a conversation history and the user's latest message.\n"
    "Your ONLY job is to rewrite the latest message into a SELF-CONTAINED search "
    "query that includes all necessary context (topic, country, indicator, time "
    "period, etc.) drawn from the conversation history.\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else.\n"
    "- Keep the same language as the user.\n"
    "- If the latest message is already self-contained, return it unchanged.\n"
    "- Do NOT answer the question — just reformulate it."
)


def contextualize_query(
    user_input: str,
    history: list,
    openai_model: str = "gpt-4o-mini",
    max_pairs: int = 5,
) -> str:
    """Rewrite a vague follow-up into a self-contained search query.

    Uses the last *max_pairs* conversation turns so the retrieval step
    searches for the right topic even when the user says things like
    \"tell me more\" or \"give me a summary\".

    Returns the original query unchanged if there is no history or on error.
    """
    if not history:
        return user_input

    # Build a compact version of the recent conversation
    recent = history[-(max_pairs * 2):]
    turns = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            turns.append(f"{role}: {content[:500]}")

    if not turns:
        return user_input

    conversation_block = "\n".join(turns)
    user_prompt = (
        f"=== CONVERSATION HISTORY ===\n{conversation_block}\n\n"
        f"=== LATEST USER MESSAGE ===\n{user_input}\n\n"
        "Rewritten self-contained query:"
    )

    try:
        client = OpenAI(timeout=15.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _CONTEXTUALISE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else user_input
    except (APITimeoutError, APIConnectionError):
        return user_input
    except Exception:
        return user_input


# ─── Query expansion (LLM-powered) ──────────────────────────────────────────

_EXPANSION_SYSTEM = (
    "You are a search-query expansion assistant for a Spanish-language RAG system "
    "about economics and finance.\n"
    "Given the user's query, generate 3-5 alternative phrasings/search queries that "
    "capture the same intent but use different vocabulary, synonyms, or related terms.\n"
    "Return ONLY a JSON array of strings. Example: [\"query 1\", \"query 2\", \"query 3\"]\n"
    "Rules:\n"
    "- Keep queries in the same language as the original.\n"
    "- Include relevant acronyms, technical synonyms, and colloquial variants.\n"
    "- Do NOT add explanations, just the JSON array."
)


def expand_queries_llm(
    query: str,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> List[str]:
    """Use an LLM to generate alternative search queries.

    Falls back to the original query on any error.
    """
    try:
        client = OpenAI(timeout=20.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _EXPANSION_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Parse the JSON array
        expansions = json.loads(raw)
        if isinstance(expansions, list):
            return [query] + [e for e in expansions if isinstance(e, str) and e.strip()]
    except (APITimeoutError, APIConnectionError):
        pass
    except Exception:
        pass
    # Fallback: just the original query
    return [query]


def expand_queries_simple(q: str) -> List[str]:
    """Legacy hardcoded expansion (kept as fallback)."""
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
    dollar_types = {"dólar", "dolar", "dollar"}
    if dollar_types & set(lower.split()):
        expansions |= {
            "dólar oficial", "dólar MEP", "dólar CCL", "dólar blue",
            "contado con liquidación", "cotización del dólar",
            "tipo de cambio", "MULC",
        }
    return [base] + [e for e in expansions if e != base]


# ─── History ─────────────────────────────────────────────────────────────────

def save_history(path: str, record: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# ─── Context building ───────────────────────────────────────────────────────

def build_context(top_docs, formatter=format_source_tag) -> str:
    """Build the context string from a list of (doc, ...) tuples or documents."""
    parts = []
    for i, item in enumerate(top_docs, 1):
        # Accept both bare documents and tuples (doc, dist, score)
        doc = item[0] if isinstance(item, (tuple, list)) else item
        tag = formatter(doc.metadata)
        parts.append(f"[{i}] ({tag})\n{doc.page_content}")
    return "\n\n".join(parts)


# ─── Misc ────────────────────────────────────────────────────────────────────

def split_k_across(n_items: int, k: int) -> List[int]:
    base = k // n_items
    rem = k % n_items
    sizes = [base] * n_items
    for i in range(rem):
        sizes[i] += 1
    return sizes


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


KEYWORDS_BOOST = ["mep", "ccl", "mulc", "tipo de cambio", "dólar", "cotización", "oficial"]


def boost_keywords(text: str, keywords: Optional[List[str]] = None) -> float:
    kws = keywords if keywords is not None else KEYWORDS_BOOST
    tl = text.lower()
    return 0.10 * sum(kw in tl for kw in kws)


# ─── Parallel retrieval ──────────────────────────────────────────────────────

def retrieve_from_collections_parallel(
    vectordbs: Dict[str, Any],
    queries: List[str],
    fetch_k_each: List[int],
    max_workers: int = 4,
) -> list:
    """Query all (collection, query) pairs concurrently and merge results.

    ChromaDB PersistentClient is read-safe across threads, so no locking needed.
    Returns a flat list of (doc, dist) tuples ready for deduplication.
    """
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
            if k_each <= 0:
                continue
            for qexp in queries:
                future = executor.submit(
                    vdb.similarity_search_with_score, qexp, k_each
                )
                futures_map[future] = col

        candidates: list = []
        try:
            for future in as_completed(futures_map, timeout=30):
                col = futures_map[future]
                try:
                    results = future.result(timeout=30)
                    for doc, dist in results:
                        md = dict(doc.metadata or {})
                        md["_collection"] = col
                        doc.metadata = md
                        candidates.append((doc, dist))
                except TimeoutError:
                    print(f"[RAG] Collection '{col}' timed out", file=sys.stderr)
                except Exception as e:
                    print(f"[RAG] Collection '{col}' retrieval error: {e}", file=sys.stderr)
        except TimeoutError:
            print("[RAG] retrieve_from_collections_parallel overall timeout", file=sys.stderr)

    return candidates


# ─── Retrieval result cache ──────────────────────────────────────────────────

_retrieval_cache: Dict[str, Any] = {}
RETRIEVAL_CACHE_TTL = 300   # seconds
RETRIEVAL_CACHE_MAX_SIZE = 100


def make_cache_key(query: str, cols: List[str], k: int, recency_weight: float) -> str:
    payload = f"{query}|{sorted(cols)}|{k}|{recency_weight:.2f}"
    return hashlib.md5(payload.encode()).hexdigest()


def get_cached_results(key: str) -> Optional[list]:
    entry = _retrieval_cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if (datetime.datetime.now().timestamp() - ts) > RETRIEVAL_CACHE_TTL:
        del _retrieval_cache[key]
        return None
    return results


def set_cached_results(key: str, results: list) -> None:
    if len(_retrieval_cache) >= RETRIEVAL_CACHE_MAX_SIZE:
        oldest_key = min(_retrieval_cache, key=lambda k: _retrieval_cache[k][0])
        del _retrieval_cache[oldest_key]
    _retrieval_cache[key] = (datetime.datetime.now().timestamp(), results)


def clear_retrieval_cache() -> None:
    _retrieval_cache.clear()


# ─── Token budget helpers ────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters for mixed Spanish/English."""
    return max(1, len(text) // 4)


def trim_history_to_budget(history: list, token_budget: int = 2000) -> list:
    """Return the most recent history messages that fit within token_budget."""
    total = 0
    result = []
    for msg in reversed(history):
        tokens = estimate_tokens(msg.get("content", ""))
        if total + tokens > token_budget:
            break
        result.insert(0, msg)
        total += tokens
    return result


# ─── Inline citation rendering ───────────────────────────────────────────────

def generate_suggested_questions(
    vectordbs: Dict[str, Any],
    n: int = 5,
    openai_model: str = "gpt-4o-mini",
) -> List[str]:
    """Sample corpus metadata and use LLM to generate contextually relevant questions.

    Falls back to generic prompts on any error so the UI always has something to show.
    """
    _FALLBACK = [
        "Cuales son las ultimas decisiones de tasas del banco central?",
        "Resumime las condiciones macroeconomicas actuales.",
        "Cual es la perspectiva de inflacion?",
        "Factores de riesgo clave en los informes mas recientes?",
        "Compara la politica monetaria de las principales economias.",
    ]
    try:
        snippets = []
        for col, vdb in list(vectordbs.items())[:3]:
            try:
                result = vdb.get(limit=8, include=["metadatas", "documents"])
                for doc_text, meta in zip(
                    result.get("documents") or [],
                    result.get("metadatas") or [],
                ):
                    name = best_source_name(meta or {})
                    date = (meta or {}).get("date_iso", "")
                    preview = short_preview(doc_text or "", 120)
                    snippets.append(f"[{col}] {name} {date}: {preview}")
                    if len(snippets) >= 12:
                        break
            except Exception:
                continue
            if len(snippets) >= 12:
                break

        if not snippets:
            return _FALLBACK

        corpus_summary = "\n".join(snippets[:12])
        prompt = (
            f"Based on these document excerpts from an economics/finance knowledge base:\n\n"
            f"{corpus_summary}\n\n"
            f"Generate exactly {n} concise, specific questions a finance or economics analyst "
            f"would want to ask. Return ONLY a JSON array of {n} strings."
        )
        client = OpenAI(timeout=20.0)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        questions = json.loads(raw)
        if isinstance(questions, list) and len(questions) >= n:
            return [str(q) for q in questions[:n]]
    except Exception:
        pass
    return _FALLBACK


def render_with_citations(answer: str, sources: list) -> str:
    """Replace [N] citation markers with HTML superscripts with source tooltips.

    Sources is a list of markdown strings produced by the source card builder.
    Out-of-range markers are left as plain text.
    """
    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(sources):
            # Extract first line of source card for tooltip (strip markdown)
            tip = re.sub(r"[*`>\[\]]", "", sources[n - 1].split("\n")[0]).strip()
            return f'<sup title="{tip}">[{n}]</sup>'
        return match.group(0)

    return re.sub(r"\[(\d+)\]", _replace, answer)
