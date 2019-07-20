"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from ordered_attention.ordered_attention_layer import OrderedAttentionLayer


if __name__ == "__main__":

    inputs = tf.random.normal([1, 7, 16, 16])
    targets = tf.random.normal([1, 7, 16, 16])

    layer = OrderedAttentionLayer(2, 2, 4, 16)
    outputs = layer([inputs, inputs, inputs])
    print("[{}, {}]".format(outputs.numpy().max(), outputs.numpy().min()))

    assert all(x == y for x, y in zip(targets.shape, outputs.shape))

    optimizer = tf.keras.optimizers.Adam()
    for i in range(10):
        loss_function = lambda: tf.reduce_mean(
            tf.losses.mean_squared_error(
                targets,
                layer([inputs, inputs, inputs])))
        optimizer.minimize(loss_function, layer.trainable_variables)
        print("Test Loss: {}".format(loss_function().numpy()))
