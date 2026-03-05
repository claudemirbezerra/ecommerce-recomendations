import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)
    // new Set remove duplicados
    const colors = [...new Set(products.map(c => c.color))]
    const categories = [...new Set(products.map(c => c.category))]

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    )
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )

    // Computar a media de idade dos compradores por produto
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}
    
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

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
        dimensions: 2 + categories.length + colors.length,
        productAvgAgeNorm,
    }
}

// tf.oneHot(2, 5) → [0, 0, 1, 0, 0]
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price, 
            context.minPrice, 
            context.maxPrice) * WEIGHTS.price
    ]) 

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color], 
        context.numColors, 
        WEIGHTS.color
    )  

    return tf.concat1d([price, age, category, color])
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
        .mean(0)
        .reshape([
            1,
            context.dimensions
        ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), 
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

function creatingTrainingData(context) {
    const inputs = [];
    const labels = [];
    context.users
    .filter(user => user.purchases.length)
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync()
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync()

            const label = user.purchases.some(
                purchase => purchase.name === product.name
            ) ? 1 : 0
            
            inputs.push([...userVector,...productVector])
            labels.push(label)
        })
    })

    const rowLength = context.dimensions * 2

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputdimensions: rowLength
        // tamanho = uservector + productVector
    }
}

async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential()

    // adiciona varias camadas para ir filtrando melhor os dados
    model.add(tf.layers.dense({
        inputShape: [trainingData.inputdimensions],
        units: 128,
        activation: 'relu'
    }))

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }))

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }))

    //relu tira negativos, sigmoid retorna ordenado entre 0 e 1
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
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

async function trainModel({ users }) {
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();
    
    const context = makeContext(products, users)
    _globalCtx = context
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
}
function recommend(user, ctx) {
    if (_model == null || !ctx?.products?.length || ctx.dimensions == null) {
        console.warn('Recommend called before model training. Train the model first.')
        return
    }
    
    const userVector = encodeUser(user, ctx).dataSync()

    // concat user vector with product vector
    // will return a tensor with the shape [ctx.products.length, ctx.dimensions]
    // and the predictions will be the shape [ctx.products.length, 1]

    // vectors should be in a postgres with vector extension
    const inputs = ctx.productVectors.map(({vector}) => {
        return [...userVector, ...vector]
    })

    const inputsTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputsTensor)

    const scores = predictions.dataSync()
    const recommendations = ctx.productVectors.map((item, index) => {
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


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
