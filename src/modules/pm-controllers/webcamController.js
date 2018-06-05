import * as tf from '@tensorflow/tfjs';

export class WebcamController {
  /** @param {HTMLVideoElement} webcam. */
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
  }

  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  capture() {
    return tf.tidy(() => {
      /* Reads the image as a Tensor from the webcam <video> element. */
      const webcamImage = tf.fromPixels(this.webcamElement);

      const croppedImage = this.cropImage(webcamImage);

      /* Expand the outer most dimension so there is a batch size of 1. */
      const batchedImage = croppedImage.expandDims(0);

      /* Normalize the image between -1 and 1. */
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

   /** @param {Tensor4D} img. */
  cropImage(img) {
    /* Crops the image using the center square of the rectangular. */
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  /**
   * Adjusts the video size to make a centered square crop without
   * including whitespace.
   * @param {number} width.
   * @param {number} height.
   */
  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  async setup() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia(
            {video: true},
            stream => {
              this.webcamElement.srcObject = stream;
              this.webcamElement.addEventListener('loadeddata', async () => {
                this.adjustVideoSize(
                    this.webcamElement.videoWidth,
                    this.webcamElement.videoHeight);
                resolve();
              }, false);
            },
            error => {
              console.debug('Houston we have a problem!', error);
              document.querySelector('#no-webcam').style.display = 'block';
            });
      } else {
        reject();
      }
    });
  }
}
