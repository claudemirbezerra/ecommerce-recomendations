import { config } from "../config.js";

export class VectorService {
    async persistVectors(productVectors, context) {
        try {
            const res = await fetch(`${config.apiUrl}/api/vectors`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ productVectors, context }),
            });
            if (res.ok) {
                console.log('Vectors persisted to database');
            } else {
                console.warn('Failed to persist vectors:', await res.text());
            }
        } catch (err) {
            console.warn('Could not persist vectors (is the API server running?):', err.message);
        }
    }

    async getVectors() {
        try {
            const res = await fetch(`${config.apiUrl}/api/vectors`);
            if (res.ok) {
                return await res.json();
            }
        } catch (err) {
            console.warn('Could not fetch vectors from DB, using in-memory:', err.message);
        }
        return null;
    }
}
