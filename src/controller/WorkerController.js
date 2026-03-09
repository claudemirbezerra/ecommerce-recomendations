import { workerEvents } from "../events/constants.js";

export class WorkerController {
    #worker;
    #events;
    #vectorService;
    #alreadyTrained = false;
    constructor({ worker, events, vectorService }) {
        this.#worker = worker;
        this.#events = events;
        this.#vectorService = vectorService;
        this.#alreadyTrained = false;
        this.init();
    }

    async init() {
        this.setupCallbacks();
    }

    static init(deps) {
        return new WorkerController(deps);
    }

    setupCallbacks() {
        this.#events.onTrainModel((data) => {
            this.#alreadyTrained = false;
            this.triggerTrain(data);
        });
        this.#events.onTrainingComplete(() => {
            this.#alreadyTrained = true;
        });

        this.#events.onRecommend(async (data) => {
            if (!this.#alreadyTrained) return;
            await this.triggerRecommend(data);
        });

        const eventsToIgnoreLogs = [
            workerEvents.progressUpdate,
            workerEvents.trainingLog,
            workerEvents.tfVisData,
            workerEvents.tfVisLogs,
            workerEvents.trainingComplete,
            workerEvents.vectorsReady,
        ];
        this.#worker.onmessage = (event) => {
            if (!eventsToIgnoreLogs.includes(event.data.type))
                console.log(event.data);

            if (event.data.type === workerEvents.progressUpdate) {
                this.#events.dispatchProgressUpdate(event.data.progress);
            }

            if (event.data.type === workerEvents.trainingComplete) {
                this.#events.dispatchTrainingComplete(event.data);
            }

            if (event.data.type === workerEvents.vectorsReady) {
                this.#vectorService.persistVectors(event.data.productVectors, event.data.context);
            }

            // Handle tfvis data from the worker for initial visualization
            if (event.data.type === workerEvents.tfVisData) {
                this.#events.dispatchTFVisorData(event.data.data);
            }

            // Handle tfvis recommendation data
            if (event.data.type === workerEvents.trainingLog) {
                this.#events.dispatchTFVisLogs(event.data);
            }
            if (event.data.type === workerEvents.recommend) {
                this.#events.dispatchRecommendationsReady(event.data);
            }
        };
    }

    triggerTrain(users) {
        this.#worker.postMessage({ action: workerEvents.trainModel, users });
    }

    async triggerRecommend(data) {
        const user = data?.user ?? data;
        const productVectors = await this.#vectorService.getVectors();
        this.#worker.postMessage({
            action: workerEvents.recommend,
            user,
            productVectors,
        });
    }
}