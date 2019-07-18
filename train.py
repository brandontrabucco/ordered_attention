"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from ordered_attention.ordered_attention_layer import OrderedAttentionLayer


if __name__ == "__main__":

    layer = OrderedAttentionLayer(2, 2, 4, 16)

    data = tf.random.normal([1, 7, 16, 16])
    target = tf.random.normal([1, 7, 16, 16])

    for i in range(100):

        with tf.GradientTape() as tape:

            tape.watch(data)
            y = layer(data, data, data)

            loss = tf.reduce_mean(tf.losses.mean_squared_error(target, y))
            print(loss)

            data -= tape.gradient(loss, data)[0]