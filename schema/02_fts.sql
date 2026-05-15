-- =========================================================================
-- 02_fts.sql — Full-text search column + GIN index para retrieval híbrido (RRF)
--
-- Agrega un tsvector generado sobre text_chunk con el stemmer 'spanish'.
-- El retrieval actual fusiona dos rankers via Reciprocal Rank Fusion (RRF):
--   1. pgvector cosine sobre `embedding` (capta paráfrasis y similitud
--      semántica).
--   2. BM25-style FTS sobre este `text_tsv` (capta entidades exactas,
--      siglas, números, fechas literales — todo lo que el embedder
--      multilingual MiniLM tiende a perder).
--
-- Es idempotente — se puede correr varias veces sin romper.
--
-- Cómo correr:
--   psql "$DATABASE_URL" -f schema/02_fts.sql
--   o pegar en el SQL Editor de Supabase y "Run".
-- =========================================================================


-- -------------------------------------------------------------------------
-- text_tsv — columna generada (STORED), se actualiza automáticamente con
-- cada INSERT/UPDATE de text_chunk. NO hay que poblarla manualmente ni
-- tocarla desde el ingest.
-- -------------------------------------------------------------------------
ALTER TABLE report_chunks
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('spanish', coalesce(text_chunk, ''))
    ) STORED;


-- -------------------------------------------------------------------------
-- Índice GIN — `text_tsv @@ plainto_tsquery(...)` queda en milisegundos
-- incluso con cientos de miles de chunks.
-- -------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS rc_text_tsv_gin_idx
    ON report_chunks USING gin (text_tsv);


-- -------------------------------------------------------------------------
-- Sanity check
-- -------------------------------------------------------------------------
DO $$
DECLARE
    total bigint;
    populated bigint;
BEGIN
    SELECT count(*) INTO total FROM report_chunks;
    SELECT count(*) INTO populated FROM report_chunks WHERE text_tsv IS NOT NULL;
    RAISE NOTICE 'report_chunks: % filas (% con text_tsv poblado)', total, populated;
    IF total > 0 AND populated <> total THEN
        RAISE WARNING 'Hay filas con text_tsv NULL — verificá manualmente.';
    END IF;
END $$;
