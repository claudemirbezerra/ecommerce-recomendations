-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store encoding context metadata (used when generating vectors)
-- Helps track which model/context version produced the vectors
CREATE TABLE IF NOT EXISTS encoding_context (
    id SERIAL PRIMARY KEY,
    dimensions INTEGER NOT NULL,
    num_categories INTEGER NOT NULL,
    num_colors INTEGER NOT NULL,
    categories_index JSONB,
    colors_index JSONB,
    min_price DECIMAL(12, 2),
    max_price DECIMAL(12, 2),
    min_age INTEGER,
    max_age INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Product vectors table
-- Vector dimension 256: enough for 2 (price+age) + categories + colors
-- Shorter vectors are padded with zeros by the application
CREATE TABLE IF NOT EXISTS product_vectors (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    vector vector(256) NOT NULL,
    meta JSONB,
    context_id INTEGER REFERENCES encoding_context(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(product_id)
);

-- Index for similarity search (cosine distance)
-- Use ivfflat for approximate nearest neighbor - good for production
-- lists = sqrt(row_count) is a common heuristic; min 1 for small datasets
CREATE INDEX IF NOT EXISTS product_vectors_vector_idx
ON product_vectors
USING ivfflat (vector vector_cosine_ops)
WITH (lists = 5);

-- Index for lookup by product_id
CREATE INDEX IF NOT EXISTS product_vectors_product_id_idx ON product_vectors(product_id);

-- Table to track migration versions
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- Record this migration
INSERT INTO schema_migrations (version) VALUES ('001_create_product_vectors')
ON CONFLICT (version) DO NOTHING;
