"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


class PartialDenseLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            output_right_size,
            output_left_size,
            **kwargs
    ):
        super(PartialDenseLayer, self).__init__()
        self.dense_right = tf.keras.layers.Dense(output_right_size, **kwargs)
        self.dense_left = tf.keras.layers.Dense(output_left_size, **kwargs)

    def call(
            self,
            inputs,
            **kwargs
    ):
        return tf.linalg.matrix_transpose(self.dense_left(
            tf.linalg.matrix_transpose(self.dense_right(inputs, **kwargs)), **kwargs))
