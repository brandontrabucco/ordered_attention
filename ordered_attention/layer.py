"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


class Layer(tf.keras.layers.Layer):

    def __init__(
            self,
            left_num_heads,
            right_num_heads,
            hidden_size,
            output_size,
            **kwargs
    ):
        super(Layer, self).__init__(**kwargs)

        self.left_num_heads = left_num_heads
        self.right_num_heads = right_num_heads
        self.hidden_size = hidden_size

        self.query_right = tf.keras.layers.Dense(
            hidden_size * right_num_heads, use_bias=False)
        self.query_left = tf.keras.layers.Dense(
            hidden_size * left_num_heads, use_bias=False)

        self.key_right = tf.keras.layers.Dense(
            hidden_size * right_num_heads, use_bias=False)
        self.key_left = tf.keras.layers.Dense(
            hidden_size * left_num_heads, use_bias=False)

        self.value_right = tf.keras.layers.Dense(
            hidden_size * right_num_heads, use_bias=False)
        self.value_left = tf.keras.layers.Dense(
            hidden_size * left_num_heads, use_bias=False)

        self.output_right = tf.keras.layers.Dense(
            output_size, use_bias=False)
        self.output_left = tf.keras.layers.Dense(
            output_size, use_bias=False)

    def call(
            self,
            queries,
            keys,
            values
    ):
        batch_size = tf.shape(queries)[0]
        query_size = tf.shape(queries)[1]
        seq_size = tf.shape(values)[1]

        queries = tf.linalg.matrix_transpose(self.query_left(
            tf.linalg.matrix_transpose(self.query_right(queries))))
        keys = tf.linalg.matrix_transpose(self.key_left(
            tf.linalg.matrix_transpose(self.key_right(keys))))
        values = tf.linalg.matrix_transpose(self.value_left(
            tf.linalg.matrix_transpose(self.value_right(values))))

        extension = [self.left_num_heads, self.hidden_size,
                     self.right_num_heads, self.hidden_size]
        queries = tf.reshape(queries, [batch_size, query_size, 1] + extension)
        keys = tf.reshape(keys, [batch_size, 1, seq_size] + extension)
        values = tf.reshape(values, [batch_size, 1, seq_size] + extension)

        indices = [0, 3, 5, 1, 2, 4, 6]
        queries = tf.transpose(queries, indices)
        keys = tf.transpose(keys, indices)
        values = tf.transpose(values, indices)

        unscaled_weights = tf.linalg.norm(tf.matmul(queries, keys),
                                          ord=2, axis=[-2, -1], keepdims=True)
        weights = tf.math.softmax(unscaled_weights / tf.math.sqrt(
            float(self.hidden_size)), axis=-2)
        seq_splits = tf.split(tf.linalg.expm(values * weights), query_size, axis=4)

        output = seq_splits[0]
        for y in seq_splits[1:]:
            output = tf.matmul(output, y)

        output = tf.cast(output, tf.complex64)
        output = tf.linalg.logm(output)
        output = tf.cast(output, tf.float32)

        output = tf.transpose(output, [0, 3, 4, 1, 5, 2, 6])
        output = tf.reshape(output, [batch_size, query_size,
                                     self.left_num_heads * self.hidden_size,
                                     self.right_num_heads * self.hidden_size])
        return tf.linalg.matrix_transpose(self.output_left(
            tf.linalg.matrix_transpose(self.output_right(output))))
