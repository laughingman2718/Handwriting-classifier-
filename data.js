
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

// Use 5/6 of the data for training, 1/6 for testing.
const TRAIN_TEST_RATIO = 5 / 6;
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS  = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

// URLs for the sprite image and label file.
const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex  = 0;
  }

  async load() {
    // 1) Download the sprite image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const imgLoadPromise = new Promise((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = err => reject(err);
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
    await imgLoadPromise;

    // 2) Prepare a buffer for all pixel data (Float32 per pixel)
    const datasetBytesBuffer =
        new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * Float32Array.BYTES_PER_ELEMENT);

    // 3) We slice the sprite into CHUNK_SIZE-high strips so no canvas exceeds browser limits.
    const CHUNK_SIZE = 5000;
    const numChunks  = Math.ceil(img.height / CHUNK_SIZE);

    // Create one small canvas we’ll reuse for every strip.
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width  = img.width;
    smallCanvas.height = CHUNK_SIZE;
    const smallCtx = smallCanvas.getContext('2d', { willReadFrequently: true });

    // 4) Loop over each chunk, draw it, and pull its pixels into our buffer.
    let byteOffset = 0;
    for (let i = 0; i < numChunks; i++) {
      // How many rows remain in this final chunk?
      const thisChunkHeight =
          Math.min(CHUNK_SIZE, img.height - i * CHUNK_SIZE);

      // Draw that stripe directly from the sprite image
      smallCtx.clearRect(0, 0, smallCanvas.width, smallCanvas.height);
      smallCtx.drawImage(
        img,
        0,               // sx
        i * CHUNK_SIZE,  // sy
        img.width,       // sw
        thisChunkHeight, // sh
        0,               // dx
        0,               // dy
        img.width,       // dw
        thisChunkHeight  // dh
      );

      // Pull out its RGBA data
      const imageData = smallCtx.getImageData(
        0, 0,
        img.width,
        thisChunkHeight
      );

      // View into our big buffer for just this chunk’s floats
      const floatView = new Float32Array(
        datasetBytesBuffer,
        byteOffset,
        IMAGE_SIZE * thisChunkHeight
      );

      // Convert RGBA→grayscale [0..1] and copy into the floatView
      for (let j = 0; j < IMAGE_SIZE * thisChunkHeight; j++) {
        floatView[j] = imageData.data[j * 4] / 255;
      }

      // Advance the byteOffset by the number of bytes we just wrote
      byteOffset += IMAGE_SIZE * thisChunkHeight * Float32Array.BYTES_PER_ELEMENT;
    }

    // 5) Store the full image buffer
    this.datasetImages = new Float32Array(datasetBytesBuffer);

    // 6) Fetch and store the one-hot labels
    const labelsResponse = await fetch(MNIST_LABELS_PATH);
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // 7) Create shuffled index arrays for train / test
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices  = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // 8) Slice images & labels into train vs test sets
    this.trainImages = this.datasetImages.slice(
      0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages  = this.datasetImages.slice(
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS);

    this.trainLabels = this.datasetLabels.slice(
      0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels  = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [ this.trainImages, this.trainLabels ],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [ this.testImages, this.testLabels ],
      () => {
        this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
        return this.testIndices[this.shuffledTestIndex];
      }
    );
  }

  nextBatch(batchSize, data, indexFn) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = indexFn();
      const imageSlice = data[0].slice(
        idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(imageSlice, i * IMAGE_SIZE);

      const labelSlice = data[1].slice(
        idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(labelSlice, i * NUM_CLASSES);
    }

    const xs     = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}
