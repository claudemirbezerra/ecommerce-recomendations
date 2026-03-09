/**
 * API routes for product vectors.
 */

import { Router } from 'express';

export function createVectorsRouter(vectorRepo) {
  const router = Router();

  router.post('/', async (req, res) => {
    try {
      const { productVectors, context } = req.body;
      if (!productVectors || !Array.isArray(productVectors)) {
        return res.status(400).json({ error: 'productVectors array is required' });
      }
      const result = await vectorRepo.saveVectors(productVectors, context);
      res.json(result);
    } catch (err) {
      console.error('Error saving vectors:', err);
      res.status(500).json({ error: err.message });
    }
  });

  router.get('/', async (req, res) => {
    try {
      const vectors = await vectorRepo.getVectors();
      res.json(vectors);
    } catch (err) {
      console.error('Error fetching vectors:', err);
      res.status(500).json({ error: err.message });
    }
  });

  router.delete('/', async (req, res) => {
    try {
      await vectorRepo.clearVectors();
      res.json({ success: true });
    } catch (err) {
      console.error('Error clearing vectors:', err);
      res.status(500).json({ error: err.message });
    }
  });

  return router;
}
