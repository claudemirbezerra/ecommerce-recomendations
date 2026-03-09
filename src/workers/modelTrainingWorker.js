/**
 * Model Training Worker
 *
 * Web Worker that trains a neural network for product recommendations.
 * Runs in a separate thread to avoid blocking the main UI.
 * Uses TensorFlow.js for machine learning operations.
 */

import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');

/** Global context holding products, users, and encoding metadata. Persists across worker messages. */
let _globalCtx = {};

/** Trained neural network model. Used for predictions after training completes. */
let _model = null;

/**
 * Feature weights for the recommendation model.
 * Higher values = stronger influence on recommendations.
 * Category and color have more weight; age has the least.
 */
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

/**
 * Normalizes a value to the range [0, 1] using min-max scaling.
 * Formula: (value - min) / (max - min)
 * @param {number} value - Raw value to normalize
 * @param {number} min - Minimum value in the dataset
 * @param {number} max - Maximum value in the dataset
 * @returns {number} - Value between 0 and 1 (or 0 if min === max to avoid division by zero)
 */
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

/**
 * Builds the encoding context from products and users.
 * Extracts min/max ranges, unique categories/colors, and creates lookup indices.
 * Also computes the average age of buyers per product (normalized).
 *
 * @param {Array} products - List of products with name, price, color, category
 * @param {Array} users - List of users with age and purchases
 * @returns {Object} - Context object with all data needed for encoding
 */
function makeContext(products, users) {
    // Extract all ages and prices for min/max calculation
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    // new Set removes duplicates; spread operator converts back to array
    const colors = [...new Set(products.map(c => c.color))]
    const categories = [...new Set(products.map(c => c.category))]

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    // Map each color/category string to a numeric index for one-hot encoding
    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    )
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )

    // Compute the average age of buyers per product (for products with purchase history)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    // Normalized average age per product: 0 = youngest buyers, 1 = oldest buyers
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimensions: 2 + categories.length + colors.length,  // price(1) + age(1) + categories + colors
        productAvgAgeNorm,
    }
}

/**
 * Creates a one-hot encoded vector with a weight applied.
 * Example: tf.oneHot(2, 5) → [0, 0, 1, 0, 0]; then multiply by weight
 *
 * @param {number} index - Index of the "hot" position (1)
 * @param {number} length - Total length of the vector
 * @param {number} weight - Multiplier applied to the one-hot vector
 * @returns {tf.Tensor} - Weighted one-hot tensor
 */
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

/**
 * Encodes a product into a fixed-length feature vector.
 * Vector structure: [price, age, category_one_hot..., color_one_hot...]
 * Each feature is weighted according to WEIGHTS.
 *
 * @param {Object} product - Product with price, color, category, name
 * @param {Object} context - Encoding context from makeContext
 * @returns {tf.Tensor} - 1D tensor of shape [dimensions]
 */
function encodeProduct(product, context) {
    // Normalized price (0–1) scaled by price weight
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice) * WEIGHTS.price
    ])

    // Average age of buyers for this product (or 0.5 if unknown), scaled by age weight
    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    // One-hot encoded category (e.g., "Electronics" → [0, 1, 0, 0])
    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    // One-hot encoded color (e.g., "Blue" → [0, 0, 1, 0])
    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    // Concatenate all features into a single 1D vector
    return tf.concat1d([price, age, category, color])
}

/**
 * Encodes a user into a feature vector.
 * If user has purchases: average of all purchased product vectors (represents user preference).
 * If user has no purchases: fallback vector using only user age.
 *
 * @param {Object} user - User with age and purchases array
 * @param {Object} context - Encoding context from makeContext
 * @returns {tf.Tensor} - 2D tensor of shape [1, dimensions]
 */
function encodeUser(user, context) {
    if (user.purchases.length) {
        // Stack all purchased product vectors, then take the mean (average preference)
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
        .mean(0)  // Average across the stacked dimension
        .reshape([
            1,
            context.dimensions
        ])
    }

    // No purchases: use user age only, zeros for category and color
    return tf.concat1d(
        [
            tf.zeros([1]),  // price placeholder
            tf.tensor1d(
                [
                    normalize(user.age, context.minAge, context.maxAge)
                    * WEIGHTS.age
                ]
            ),
            tf.zeros([context.numCategories]),
            tf.zeros([context.numColors]),
        ]
    ).reshape([
        1,
        context.dimensions
    ])
}

/**
 * Creates training data for the neural network.
 * For each user with purchases and each product: input = [userVector, productVector], label = 1 if purchased, 0 otherwise.
 * This is a binary classification setup: predict whether the user would buy the product.
 *
 * @param {Object} context - Encoding context with products and users
 * @returns {Object} - { xs: input tensor, ys: label tensor, inputdimensions: size of each input row }
 */
function creatingTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.users
        .filter(user => user.purchases.length)  // Only users who bought something
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()  // Convert tensor to JS array
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                // Label: 1 if user bought this product, 0 otherwise
                const label = user.purchases.some(
                    purchase => purchase.name === product.name
                ) ? 1 : 0

                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })

    const rowLength = context.dimensions * 2  // userVector length + productVector length

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputdimensions: rowLength
    }
}

/**
 * Creates and trains the neural network.
 * Architecture: 4 dense layers (128 → 64 → 32 → 1) with ReLU, then sigmoid for binary output.
 * Uses Adam optimizer and binary cross-entropy loss.
 * Sends training progress (loss, accuracy) to the main thread after each epoch.
 *
 * @param {Object} trainingData - { xs, ys, inputdimensions }
 * @returns {Promise<tf.LayersModel>} - Trained model
 */
async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential()

    // Input layer: receives concatenated user+product vector
    model.add(tf.layers.dense({
        inputShape: [trainingData.inputdimensions],
        units: 128,
        activation: 'relu'  // ReLU: outputs max(0, x); removes negative values
    }))

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }))

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }))

    // Output layer: single neuron, sigmoid outputs probability between 0 and 1
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    model.compile({
        optimizer: tf.train.adam(0.01),  // Adam: adaptive learning rate
        loss: 'binaryCrossentropy',     // For binary classification
        metrics: ['accuracy']
    })

    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}

/**
 * Main training pipeline: loads products, builds context, creates training data, trains model.
 * Sends progress updates and trainingComplete when done.
 *
 * @param {Object} params - { users: Array }
 */
async function trainModel({ users }) {
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(products, users)
    _globalCtx = context

    // Pre-compute product vectors for fast recommendations later
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

    const trainingData = creatingTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainingData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
    postMessage({
        type: workerEvents.vectorsReady,
        productVectors: context.productVectors,
        context: {
            dimensions: context.dimensions,
            numCategories: context.numCategories,
            numColors: context.numColors,
            categoriesIndex: context.categoriesIndex,
            colorsIndex: context.colorsIndex,
            minPrice: context.minPrice,
            maxPrice: context.maxPrice,
            minAge: context.minAge,
            maxAge: context.maxAge,
        },
    });
}

/**
 * Generates product recommendations for a user.
 * Concatenates user vector with each product vector, runs model prediction, sorts by score.
 * Sends results to main thread via postMessage.
 * Uses productVectors from param (from DB) or from ctx (in-memory fallback).
 *
 * @param {Object} user - User to recommend for
 * @param {Object} ctx - Global context (products, dimensions, productVectors for fallback)
 * @param {Array} [productVectors] - Product vectors from DB (preferred over ctx.productVectors)
 */
function recommend(user, ctx, productVectors) {
    const vectors = productVectors ?? ctx?.productVectors;
    if (_model == null || !ctx?.products?.length || ctx.dimensions == null) {
        console.warn('Recommend called before model training. Train the model first.')
        return
    }
    if (!vectors?.length) {
        console.warn('No product vectors available. Train the model or ensure vectors are loaded from DB.')
        return
    }

    const userVector = encodeUser(user, ctx).dataSync()
    const dim = ctx.dimensions;
    // Vectors from DB are padded to 256; trim to actual dimensions for model input
    const trimVector = (v) => (v.length > dim ? v.slice(0, dim) : v);

    // Build input matrix: each row = [userVector, productVector]
    const inputs = vectors.map(({vector}) => {
        return [...userVector, ...trimVector(vector)]
    })

    const inputsTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputsTensor)

    const scores = predictions.dataSync()
    const recommendations = vectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index]
        }
    })

    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score)

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    })
}

/** Maps event types to handler functions */
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx, d.productVectors),
};

/** Message listener: dispatches incoming messages to the appropriate handler */
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
