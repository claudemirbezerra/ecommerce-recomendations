/**
 * Repository for product vectors in PostgreSQL (pgvector).
 */

import pg from 'pg';
import pgvector from 'pgvector/pg';

const VECTOR_DIM = 256;

function toArray(vector) {
  if (Array.isArray(vector)) return vector;
  if (vector && typeof vector[Symbol.iterator] === 'function') return [...vector];
  if (vector && typeof vector === 'object') return Array.from(Object.values(vector));
  return [];
}

function padVector(vector) {
  const arr = toArray(vector);
  if (arr.length >= VECTOR_DIM) return arr.slice(0, VECTOR_DIM);
  return [...arr, ...Array(VECTOR_DIM - arr.length).fill(0)];
}

export class VectorRepository {
  constructor(connectionString) {
    this.pool = new pg.Pool({ connectionString });
    this.pool.on('connect', async (client) => pgvector.registerTypes(client));
  }

  async saveVectors(productVectors, context = null) {
    const client = await this.pool.connect();
    try {
      let contextId = null;
      if (context) {
        const ctxResult = await client.query(
          `INSERT INTO encoding_context (
            dimensions, num_categories, num_colors,
            categories_index, colors_index, min_price, max_price, min_age, max_age
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          RETURNING id`,
          [
            context.dimensions,
            context.numCategories,
            context.numColors,
            JSON.stringify(context.categoriesIndex || {}),
            JSON.stringify(context.colorsIndex || {}),
            context.minPrice ?? null,
            context.maxPrice ?? null,
            context.minAge ?? null,
            context.maxAge ?? null,
          ]
        );
        contextId = ctxResult.rows[0].id;
      }

      for (const pv of productVectors) {
        const productId = pv.meta?.id ?? pv.meta?.product_id ?? 0;
        const paddedVector = padVector(pv.vector);

        await client.query(
          `INSERT INTO product_vectors (product_id, product_name, vector, meta, context_id)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (product_id) DO UPDATE SET
             vector = EXCLUDED.vector,
             meta = EXCLUDED.meta,
             context_id = EXCLUDED.context_id,
             created_at = NOW()`,
          [
            productId,
            pv.name,
            pgvector.toSql(paddedVector),
            JSON.stringify(pv.meta || {}),
            contextId,
          ]
        );
      }

      return { success: true, contextId };
    } finally {
      client.release();
    }
  }

  async getVectors() {
    const result = await this.pool.query(
      `SELECT product_id, product_name, vector, meta
       FROM product_vectors
       ORDER BY product_id`
    );

    return result.rows.map((row) => ({
      name: row.product_name,
      meta: row.meta || {},
      vector: Array.isArray(row.vector) ? row.vector : parseVector(String(row.vector)),
    }));
  }

  async clearVectors() {
    await this.pool.query('DELETE FROM product_vectors');
    await this.pool.query('DELETE FROM encoding_context');
    return { success: true };
  }
}

function parseVector(str) {
  if (Array.isArray(str)) return str;
  if (typeof str === 'string') {
    const cleaned = str.replace(/^\[|\]$/g, '').trim();
    return cleaned ? cleaned.split(',').map(Number) : [];
  }
  return [];
}
