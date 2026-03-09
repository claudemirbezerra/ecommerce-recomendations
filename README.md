# E-commerce Recommendation System

A web application that displays user profiles and product listings, with the ability to track user purchases for future machine learning recommendations using TensorFlow.js.

## Project Structure

- `index.html` - Main HTML file for the application
- `src/index.js` - Entry point for the frontend application
- `src/view/` - Contains classes for managing the DOM and templates
- `src/controller/` - Contains controllers to connect views and services
- `src/service/` - Contains business logic for data handling
- `src/events/` - Event bus for component communication
- `src/workers/` - Web Workers for ML model training (TensorFlow.js)
- `data/` - Contains JSON files with user and product data

### Server Structure

```
server/
├── index.js              # Express API entry point (port 3001)
├── api/
│   └── vectors.js        # REST routes for product vectors (POST, GET, DELETE)
└── repository/
    └── vectorRepository.js   # PostgreSQL/pgvector data access layer
```

**Endpoints:**
- `POST /api/vectors` - Save product vectors (body: `{ productVectors, context }`)
- `GET /api/vectors` - List all product vectors
- `DELETE /api/vectors` - Clear all vectors
- `GET /health` - Health check

**Documentação API (Swagger):** `http://localhost:3001/api-docs`

### Database Structure

PostgreSQL com extensão **pgvector** (Docker: `pgvector/pgvector:pg16`).

**Tabelas:**

| Tabela | Descrição |
|--------|-----------|
| `encoding_context` | Metadados do contexto de encoding (dimensões, índices de categorias/cores, faixas de preço/idade) |
| `product_vectors` | Vetores de produtos (256 dimensões) para busca por similaridade |
| `schema_migrations` | Controle de versões das migrations |

**Schema principal (`product_vectors`):**
- `product_id` (INTEGER, UNIQUE) - ID do produto
- `product_name` (VARCHAR) - Nome do produto
- `vector` (vector(256)) - Embedding para similaridade (índice ivfflat com cosine distance)
- `meta` (JSONB) - Metadados adicionais
- `context_id` (FK → encoding_context) - Contexto usado na geração do vetor

## Setup and Run

1. Install dependencies:
```bash
npm install
```

2. Start the database (PostgreSQL + pgvector):
```bash
npm run db:up
npm run migrate
```
Ou em um comando: `npm run db:init`

3. Start the API server (porta 3001):
```bash
npm run server
```

4. Start the frontend (porta 3000):
```bash
npm start
```

5. Open your browser and navigate to `http://localhost:3000`

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage

## Future Enhancements

- TensorFlow.js-based recommendation engine
- User similarity analysis
- Product recommendation based on purchase history
