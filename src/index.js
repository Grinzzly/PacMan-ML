import * as tf from '@tensorflow/tfjs';

import * as ui from './modules/pm-utils/utils';
import { WebcamController } from './modules/pm-controllers/webcamController';
import { DatasetController } from './modules/pm-controllers/datasetController';

/* The number of classes that will be predicted is 4 for up, down, left, and right. */
const NUM_CLASSES = 4;

const webcam = new WebcamController(document.getElementById('webcam'));
const datasetController = new DatasetController(NUM_CLASSES);

let mobilenet;
let model;

async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  /* Return a model that outputs an internal activation. */
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

/*
* When the UI buttons are pressed, read a frame from the webcam and associate
* it with the class label given by the button. up, down, left, right are
* labels 0, 1, 2, 3 respectively.
*/
ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    datasetController.addExample(mobilenet.predict(img), label);

    ui.drawThumb(img, label);
  });
});

async function train() {
  if (datasetController.xs == null) {
    throw new Error('Add some examples before training!');
  }

  /*
  * Creates a 2-layer fully connected model. By creating a separate model,
  * rather than adding layers to the mobilenet model, we "freeze" the weights
  * of the mobilenet model, and only train weights from the new model.
  */
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),

      /* Layer 1 */
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),

      /*
      * Layer 2. The number of units of the last layer should correspond
      * to the number of classes we want to predict.
      */
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const optimizer = tf.train.adam(ui.getLearningRate());

  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  const batchSize =
    Math.floor(datasetController.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
      `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  model.fit(datasetController.xs, datasetController.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();

  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);

      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});

document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  /*
  * Warm up the model. This uploads weights to the GPU and compiles the WebGL
  * programs so the first time we collect data from the webcam it will be
  * quick.
  */
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

init();
