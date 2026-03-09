/**
 * Express API server for vector storage.
 * Serves product vectors from PostgreSQL (pgvector).
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import swaggerUi from 'swagger-ui-express';
import { swaggerSpec } from './swagger.js';
import { createVectorsRouter } from './api/vectors.js';
import { VectorRepository } from './repository/vectorRepository.js';

const PORT = process.env.PORT || 3001;
const DATABASE_URL =
  process.env.DATABASE_URL ||
  'postgresql://ecommerce:ecommerce_secret@localhost:5432/ecommerce_recommendations';

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: '10mb' }));

const vectorRepo = new VectorRepository(DATABASE_URL);
app.use('/api/vectors', createVectorsRouter(vectorRepo));

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));
app.get('/api-docs.json', (_, res) => res.json(swaggerSpec));

app.get('/health', (_, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Vector API running at http://localhost:${PORT}`);
  console.log(`Swagger UI at http://localhost:${PORT}/api-docs`);
});
