# Asistente Económico — RAG Chatbot

Chatbot RAG en español que responde preguntas sobre informes económicos
diarios (PDFs) combinando recuperación semántica con datos duros desde SQL.
Mobile-first; UI 100% vanilla.

## Stack

| Componente     | Tecnología                                                     |
|----------------|----------------------------------------------------------------|
| Frontend       | HTML, CSS, JavaScript vanilla (mobile-first)                   |
| Backend        | Python 3.11+ · FastAPI · Uvicorn                               |
| Vector DB      | Postgres + [pgvector](https://github.com/pgvector/pgvector) (Supabase) |
| Embeddings     | sentence-transformers · `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| LLM principal  | OpenAI `gpt-5-mini`                                            |
| LLM auxiliar   | OpenAI `gpt-4.1-nano` (clasificador, rewrite, filter, follow-ups) |
| Re-ranking     | Cross-encoder `ms-marco-MiniLM-L-6-v2` (local)                 |

## Arquitectura

Pipeline por request `/chat`:

1. **Query rewriting** (sólo con historial): convierte la pregunta en standalone.
2. **Router** (paralelo): clasifica si necesita RAG y refresca el resumen de la colección.
3. **Filter extractor**: detecta filtros de archivo / fecha / sección / tema.
4. **Retrieval híbrido**: top-K + piso de similitud, traduce filtros estilo Chroma a SQL.
5. **Re-ranking** con cross-encoder local.
6. **Fusión RAG + tools SQL**: el LLM puede llamar funciones (`get_fx`, `get_forex_*`)
   que ejecutan SQL parametrizado contra Postgres. La narrativa viene de los chunks,
   los números vienen de SQL — en conflicto, gana SQL.
7. **Streaming SSE** de la respuesta + `[META]` con request_id, costo y debug info.
8. **Follow-ups**: 3 preguntas sugeridas como chips clickeables.

Cada request produce un trace con duración por etapa, tokens reales por modelo,
costo en USD, tools llamadas y chunks usados, persistido en `logs/traces.jsonl`.

## Estructura

```
.
├── api/                FastAPI routes + Pydantic models
├── core/               Pipeline modules (router, retrieval, rerank, llm, …)
│   ├── observability.py  RequestTrace + Span + price table
│   ├── prompts.py        System prompts (chatbot, classifier, rewriter, …)
│   ├── sql_tools.py      Tools SQL expuestos al LLM via function calling
│   └── vectorstore.py    Postgres + pgvector backend
├── schema/             DDL para Supabase (report_chunks, fx, forex)
├── scripts/            CLIs: ingest, retag, trace_stats, load_sql
├── static/             SPA del chatbot (index.html, styles.css, app.js)
├── server.py           FastAPI entry point
└── requirements.txt
```

## Setup

### Requisitos

- Python 3.11+
- Una base Postgres con la extensión `vector` habilitada (Supabase u otra)
- Una OpenAI API key

### Instalación

```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env y completar OPENAI_API_KEY + DATABASE_URL
```

### Schema

```bash
python scripts/load_sql.py schema/00_setup.sql schema/01_tables.sql
```

> Las tablas `fx` y `forex` que usan las tools SQL se llenan con seeds locales
> (no versionados). Cargalos con `python scripts/load_sql.py path/to/dump.sql`
> o pegándolos en el SQL Editor de Supabase.

### Indexar PDFs

Colocar los PDFs en `./docs/` (o setear `DOCS_DIR`) y correr:

```bash
python scripts/ingest.py --pdf-dir ./docs
python scripts/ingest.py --pdf-dir ./docs --reset    # vacía la tabla primero
```

## Ejecutar

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Abrir `http://localhost:8000`.

## Observabilidad

Cada `/chat` genera una entrada en `logs/traces.jsonl` con duración por etapa,
tokens, costo y errores. Para resumir:

```bash
python scripts/trace_stats.py                 # todo
python scripts/trace_stats.py --last 50       # últimas 50 requests
python scripts/trace_stats.py --since 2026-05-07
python scripts/trace_stats.py --errors        # solo errores
```

La tabla de precios USD/1M tokens vive en [core/observability.py](core/observability.py)
y debe actualizarse cuando OpenAI mueva tarifas.

## Variables de entorno

Ver [.env.example](.env.example).

| Variable                          | Default                                              |
|-----------------------------------|------------------------------------------------------|
| `OPENAI_API_KEY`                  | —                                                    |
| `OPENAI_MODEL`                    | `gpt-5-mini`                                         |
| `OPENAI_MODEL_FAST`               | `gpt-4.1-nano`                                       |
| `EMBEDDING_MODEL`                 | `paraphrase-multilingual-MiniLM-L12-v2`              |
| `DATABASE_URL`                    | —                                                    |
| `DOCS_DIR`                        | `./docs`                                             |
| `SIMILARITY_THRESHOLD`            | `0.75`                                               |
| `SIMILARITY_THRESHOLD_RETRIEVAL`  | `0.30`                                               |
| `CHUNK_TARGET_TOKENS`             | `400`                                                |
| `CHUNK_MAX_TOKENS`                | `700`                                                |
| `TOP_K`                           | `8`                                                  |
| `RERANK_TOP_N`                    | `5`                                                  |
