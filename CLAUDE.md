# RAG Chatbot — Asistente Económico

## Descripción

Chatbot RAG económico para uso empresarial. Los usuarios hacen preguntas sobre informes económicos en PDF que se indexan periódicamente. La aplicación es mobile-first pero funciona en web también. Idioma: español únicamente.

## Stack Técnico

| Componente    | Tecnología                                             |
|---------------|--------------------------------------------------------|
| Frontend      | HTML, CSS, JavaScript vanilla (mobile-first)           |
| Backend       | Python + FastAPI                                       |
| Vector DB     | ChromaDB (persistente, espacio coseno)                 |
| Embeddings    | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`   |
| LLM respuesta | OpenAI `gpt-5-mini` (respuestas + chat)                |
| LLM rápido    | OpenAI `gpt-4.1-nano` (clasificador + topic tags)      |
| Servidor      | Uvicorn                                                |

## Estructura de Carpetas

```
rag_chatbot/
├── CLAUDE.md              # Este archivo — leer al inicio de cada sesión
├── .env                   # Variables de entorno (NO commitear)
├── .env.example           # Template de variables de entorno
├── .gitignore
├── requirements.txt       # Dependencias Python
├── server.py              # Entry point FastAPI (monta rutas + sirve estáticos)
├── api/
│   ├── __init__.py
│   ├── models.py          # Modelos Pydantic (ChatRequest)
│   └── routes.py          # Endpoint /chat (streaming SSE) + /collections
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuración centralizada desde env vars
│   ├── embedder.py        # Embeddings con sentence-transformers
│   ├── chunking.py        # Chunking semántico por similitud coseno
│   ├── vectorstore.py     # Operaciones ChromaDB (add, query, list)
│   ├── router.py          # Clasificador de intención (retrieve true/false)
│   ├── llm.py             # Streaming OpenAI + extracción de topic tags
│   └── prompts.py         # System prompts (chatbot + clasificador)
├── scripts/
│   └── ingest.py          # CLI de indexación de PDFs
├── static/
│   ├── index.html         # SPA del chatbot
│   ├── styles.css         # Estilos mobile-first
│   └── app.js             # Lógica del frontend
└── docs/                  # Directorio para PDFs a indexar
```

## Flujo de la Aplicación

1. El usuario escribe una pregunta en el chat
2. El frontend envía `POST /chat` con `{ query, show_sources, history }`
3. El backend clasifica la intención con un LLM liviano:
   - `{"retrieve": true}` → pregunta sobre datos económicos → va a ChromaDB
   - `{"retrieve": false}` → saludo o conversación general → responde directo
4. Si retrieve es true:
   - Se buscan los top-K chunks más relevantes en ChromaDB
   - Se ordenan cronológicamente por `pub_date`
   - Se arma el contexto y se pasa al LLM con el system prompt + modo de citas
5. Si retrieve es false:
   - El LLM responde directamente sin contexto de documentos
6. La respuesta se envía por streaming (SSE) al frontend
7. Si `show_sources` es true, se envían las fuentes como metadata al final del stream

## Decisiones de Arquitectura

- **Sin langchain**: Se usan las librerías directamente (sentence-transformers, chromadb, openai) para reducir complejidad y dependencias.
- **Clasificador de intención antes de RAG**: Evita búsquedas innecesarias en ChromaDB para saludos y conversación general, ahorrando latencia y costo.
- **Chunking semántico**: En lugar de cortar por tamaño fijo, se agrupan oraciones consecutivas por similitud coseno. Esto produce chunks más coherentes.
- **Topic tags con LLM**: Durante la ingesta, un LLM extrae etiquetas temáticas por chunk. Se almacenan como metadata en ChromaDB.
- **Modo de citas dinámico**: El `show_sources` booleano se envía en cada request y controla el system prompt del LLM para incluir o excluir citas.
- **Streaming SSE**: Respuestas en tiempo real para mejor UX.
- **Dos modelos LLM**: `gpt-5-mini` para respuestas (calidad), `gpt-4.1-nano` para clasificación de intención y topic tags (velocidad/costo). No usar `temperature` con GPT-5+.

## Convenciones de Código

- Python: snake_case, type hints, docstrings en español o inglés
- Frontend: camelCase para JS, BEM-like para CSS
- Archivos UTF-8
- No hardcodear valores — todo desde variables de entorno
- No agregar features que no estén en el spec

## Variables de Entorno

Ver `.env.example` para la lista completa:
- `OPENAI_API_KEY` — API key de OpenAI
- `OPENAI_MODEL` — Modelo LLM (default: gpt-5-mini)
- `EMBEDDING_MODEL` — Modelo de embeddings
- `CHROMA_DB_DIR` — Ruta de persistencia de ChromaDB
- `CHROMA_COLLECTION` — Nombre de la colección
- `DOCS_DIR` — Directorio de PDFs
- `SIMILARITY_THRESHOLD` — Umbral para chunking semántico
- `TOP_K` — Cantidad de chunks a recuperar

## Cómo Ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Indexar PDFs
python scripts/ingest.py --pdf-dir ./docs

# Iniciar servidor
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## Instrucción para Claude

Actualizar este archivo al final de cada sesión o cuando se agregue una feature, cambie el stack, o se tome una decisión de arquitectura relevante.
