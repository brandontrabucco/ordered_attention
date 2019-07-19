"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from ordered_attention.attention_layer import AttentionLayer


if __name__ == "__main__":

    layer = AttentionLayer(2, 2, 4, 16)
    data = tf.random.normal([1, 7, 16, 16])
    target = tf.random.normal([1, 7, 16, 16])
    optimizer = tf.keras.optimizers.Adam()

    for i in range(100):

        loss_function = lambda: tf.reduce_mean(
            tf.losses.mean_squared_error(
                target,
                layer([data, data, data])))
        optimizer.minimize(loss_function, layer.trainable_variables)
        print(loss_function())
