import * as tf from '@tensorflow/tfjs';

export class DatasetController {
    constructor(numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * Adds an example to the dataset controller.
     * @param {Tensor} example.
     * @param {number} label.
     */
    addExample(example, label) {
        /* One-hot encode the label. */
        const y = tf.tidy(
            () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

        if (this.xs == null) {

            /* For the first example that gets added, keep example and y so that the
            * DatasetController owns the memory of the inputs. This makes sure that
            * if addExample() is called in a tf.tidy(), these Tensors will not get
            * disposed.
            */
            this.xs = tf.keep(example);
            this.ys = tf.keep(y);
        } else {
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(example, 0));

            const oldY = this.ys;
            this.ys = tf.keep(oldY.concat(y, 0));

            oldX.dispose();
            oldY.dispose();
            y.dispose();
        }
    }
}
