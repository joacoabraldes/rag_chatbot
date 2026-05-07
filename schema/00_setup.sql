-- =========================================================================
-- 00_setup.sql — vector store setup for the RAG chatbot
--
-- Idempotente: se puede correr varias veces sin romper.
-- Aplicar antes del primer ingest de PDFs.
--
-- Cómo correr:
--   psql "$DATABASE_URL" -f schema/00_setup.sql
--   o pegar en el SQL Editor de Supabase y "Run".
-- =========================================================================

-- pgvector ya quedó habilitado desde el dashboard, pero por idempotencia:
CREATE EXTENSION IF NOT EXISTS vector;

-- -------------------------------------------------------------------------
-- report_chunks — un row por chunk indexado.
--
-- Schema espejado del metadata que produce scripts/ingest.py.
-- vector(384) coincide con la dim del embedder local
-- (paraphrase-multilingual-MiniLM-L12-v2). Si en el futuro migramos a
-- text-embedding-3-small habrá que crear una columna nueva o re-crear
-- la tabla con vector(1536).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS report_chunks (
    id            bigserial PRIMARY KEY,
    chunk_id      text UNIQUE NOT NULL,
    source_file   text NOT NULL,
    pub_date      date,
    page_number   int,
    block_index   int,
    block_header  text,
    section       text NOT NULL DEFAULT 'resumen',
    chunk_index   int,
    text_chunk    text NOT NULL,
    -- topic_keys como array nativo en lugar de string CSV — habilita índice GIN.
    topic_keys    text[] DEFAULT ARRAY[]::text[],
    -- topic_tags se mantiene como string legible para el header del prompt.
    topic_tags    text DEFAULT '',
    embedding     vector(384) NOT NULL,
    created_at    timestamptz DEFAULT now()
);

-- -------------------------------------------------------------------------
-- Índices
-- -------------------------------------------------------------------------

-- B-tree sobre fechas: usado en filtros tipo BETWEEN y en la summary query.
CREATE INDEX IF NOT EXISTS rc_pub_date_idx
    ON report_chunks (pub_date);

-- B-tree sobre section: pre-filtro estructural.
CREATE INDEX IF NOT EXISTS rc_section_idx
    ON report_chunks (section);

-- B-tree sobre source_file: filtros por archivo específico.
CREATE INDEX IF NOT EXISTS rc_source_file_idx
    ON report_chunks (source_file);

-- GIN sobre topic_keys (array): consultas con topic_keys && ARRAY['fx', 'tasas'].
CREATE INDEX IF NOT EXISTS rc_topic_keys_gin_idx
    ON report_chunks USING gin (topic_keys);

-- HNSW sobre embedding: búsqueda vectorial con coseno.
-- ops = vector_cosine_ops; el embedder produce vectores normalizados.
-- Si la colección crece >100k filas conviene tunear m / ef_construction.
CREATE INDEX IF NOT EXISTS rc_embedding_hnsw_idx
    ON report_chunks USING hnsw (embedding vector_cosine_ops);

-- -------------------------------------------------------------------------
-- Sanity checks (no rompen si ya existen)
-- -------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector no está habilitado. Habilitarlo desde Database > Extensions en Supabase.';
    END IF;
END $$;
