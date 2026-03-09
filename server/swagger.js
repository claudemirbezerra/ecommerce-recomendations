/**
 * Swagger/OpenAPI configuration for the Vector API.
 */

import swaggerJsdoc from 'swagger-jsdoc';

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'E-commerce Recommendations API',
      version: '1.0.0',
      description: 'API para armazenamento e consulta de vetores de produtos (pgvector)',
    },
    servers: [
      { url: 'http://localhost:3001', description: 'Servidor local' },
    ],
    paths: {
      '/health': {
        get: {
          summary: 'Health check',
          description: 'Verifica se a API está em execução',
          tags: ['Health'],
          responses: {
            200: {
              description: 'API está funcionando',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: { ok: { type: 'boolean', example: true } },
                  },
                },
              },
            },
          },
        },
      },
      '/api/vectors': {
        get: {
          summary: 'Listar vetores',
          description: 'Retorna todos os vetores de produtos armazenados',
          tags: ['Vectors'],
          responses: {
            200: {
              description: 'Lista de vetores',
              content: {
                'application/json': {
                  schema: {
                    type: 'array',
                    items: {
                      type: 'object',
                      properties: {
                        name: { type: 'string', description: 'Nome do produto' },
                        meta: { type: 'object', description: 'Metadados do produto' },
                        vector: {
                          type: 'array',
                          items: { type: 'number' },
                          description: 'Vetor de embedding (256 dimensões)',
                        },
                      },
                    },
                  },
                },
              },
            },
            500: { description: 'Erro interno do servidor' },
          },
        },
        post: {
          summary: 'Salvar vetores',
          description: 'Salva ou atualiza vetores de produtos no banco',
          tags: ['Vectors'],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['productVectors'],
                  properties: {
                    productVectors: {
                      type: 'array',
                      description: 'Lista de vetores de produtos',
                      items: {
                        type: 'object',
                        required: ['vector', 'name'],
                        properties: {
                          vector: {
                            type: 'array',
                            items: { type: 'number' },
                            description: 'Vetor de embedding',
                          },
                          name: { type: 'string', description: 'Nome do produto' },
                          meta: {
                            type: 'object',
                            description: 'Metadados (id, product_id, etc.)',
                          },
                        },
                      },
                    },
                    context: {
                      type: 'object',
                      description: 'Contexto de encoding (opcional)',
                      properties: {
                        dimensions: { type: 'number' },
                        numCategories: { type: 'number' },
                        numColors: { type: 'number' },
                        categoriesIndex: { type: 'object' },
                        colorsIndex: { type: 'object' },
                        minPrice: { type: 'number' },
                        maxPrice: { type: 'number' },
                        minAge: { type: 'number' },
                        maxAge: { type: 'number' },
                      },
                    },
                  },
                },
              },
            },
          },
          responses: {
            200: {
              description: 'Vetores salvos com sucesso',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      contextId: { type: 'number', nullable: true },
                    },
                  },
                },
              },
            },
            400: { description: 'productVectors array é obrigatório' },
            500: { description: 'Erro interno do servidor' },
          },
        },
        delete: {
          summary: 'Limpar vetores',
          description: 'Remove todos os vetores e contextos do banco',
          tags: ['Vectors'],
          responses: {
            200: {
              description: 'Vetores removidos',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: { success: { type: 'boolean', example: true } },
                  },
                },
              },
            },
            500: { description: 'Erro interno do servidor' },
          },
        },
      },
    },
  },
  apis: [],
};

export const swaggerSpec = swaggerJsdoc(options);
