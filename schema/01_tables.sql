-- =========================================================================
-- 01_tables.sql — DDL para tablas de datos duros (fx, forex)
--
-- Las columnas reflejan el formato de los dumps que cargamos como seeds
-- (schema/fx.sql y schema/forex.sql, no versionados).
--
-- Cómo correr:
--   psql "$DATABASE_URL" -f schema/01_tables.sql
--   o pegar en el SQL Editor de Supabase y "Run".
--
-- IMPORTANTE: aplicar ANTES de cargar los seeds — los INSERTs asumen que
-- las tablas ya existen.
-- =========================================================================


-- -------------------------------------------------------------------------
-- public.fx — cotizaciones agregadas diarias del mercado argentino.
--
-- Una fila por fecha. Todas las columnas (excepto la PK) admiten NULL
-- porque hay fechas históricas con datos parciales.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.fx (
    "date"     date PRIMARY KEY,
    ccl        numeric,   -- contado con liqui
    mepal      numeric,   -- MEP via AL30
    mepgd      numeric,   -- MEP via GD30
    cclgd      numeric,   -- CCL via GD30
    a3500      numeric,   -- Comunicación A3500 (BCRA reference)
    canje      numeric,   -- ratio CCL/MEP
    brecha     numeric,   -- ratio MEP/oficial - 1
    last_mlc   numeric    -- último precio MLC del día
);

CREATE INDEX IF NOT EXISTS fx_date_idx ON public.fx ("date");


-- -------------------------------------------------------------------------
-- public.forex — operaciones por instrumento intradiarias del MAE.
--
-- Granularidad: una fila por (date, rueda, instrumento, hora). NO hay PK
-- natural porque puede haber duplicados por settle. Usamos un id sintético.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.forex (
    id                       bigserial PRIMARY KEY,
    "date"                   date NOT NULL,
    rueda                    text,
    instrumento              text,
    currency_out             text,
    currency_in              text,
    settle                   int,            -- T+N (0, 1, 2)
    settle_date              date,
    monto                    numeric,
    cotizacion               numeric,
    hora                     time,
    -- columnas vestigiales — siempre NULL en la data actual, dejadas por
    -- compatibilidad con los INSERT existentes:
    descripcion              text,
    tipo_emision             text,
    codigo_segmento          text,
    codigo_plazo             text,
    moneda                   text,
    monto_acumulado          numeric,
    precio_ultimo            numeric,
    ultima_tasa              numeric,
    precio_cierre_anterior   numeric,
    precio_minimo            numeric,
    precio_maximo            numeric,
    open_interest            numeric,
    variacion                numeric
);

-- Índices que probablemente vamos a usar desde las tools SQL:
CREATE INDEX IF NOT EXISTS forex_date_idx        ON public.forex ("date");
CREATE INDEX IF NOT EXISTS forex_instrumento_idx ON public.forex (instrumento);
CREATE INDEX IF NOT EXISTS forex_currency_idx    ON public.forex (currency_out, currency_in);
CREATE INDEX IF NOT EXISTS forex_rueda_idx       ON public.forex (rueda);


-- -------------------------------------------------------------------------
-- Sanity check
-- -------------------------------------------------------------------------
DO $$
BEGIN
    RAISE NOTICE 'Tablas fx y forex listas. Cargá los seeds con scripts/load_sql.py.';
END $$;
