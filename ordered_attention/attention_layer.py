"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from ordered_attention import partial_dense_layer
from ordered_attention import similarity_methods
from ordered_attention import merge_methods


def separate_heads(
        inputs,
        left_num_heads,
        right_num_heads,
        hidden_size
):
    outputs = tf.reshape(inputs, tf.concat([
        tf.shape(inputs)[:(-2)], [left_num_heads, hidden_size,
                                  right_num_heads, hidden_size]], 0))
    return tf.transpose(outputs, tf.concat([
        tf.range(tf.shape(tf.shape(outputs))[0])[:(-4)],
        [tf.shape(tf.shape(outputs))[0] - 4, tf.shape(tf.shape(outputs))[0] - 2,
         tf.shape(tf.shape(outputs))[0] - 3, tf.shape(tf.shape(outputs))[0] - 1]], 0))


def merge_heads(
        inputs,
        left_num_heads,
        right_num_heads,
        hidden_size
):
    outputs = tf.transpose(inputs, tf.concat([
        tf.range(tf.shape(tf.shape(inputs))[0])[:(-4)],
        [tf.shape(tf.shape(inputs))[0] - 4, tf.shape(tf.shape(inputs))[0] - 2,
         tf.shape(tf.shape(inputs))[0] - 3, tf.shape(tf.shape(inputs))[0] - 1]], 0))
    return tf.reshape(outputs, tf.concat([
        tf.shape(inputs)[:(-4)], [left_num_heads * hidden_size,
                                  right_num_heads * hidden_size]], 0))


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            left_num_heads,
            right_num_heads,
            hidden_size,
            output_size,
            similarity_method=similarity_methods.method_one,
            merge_method=merge_methods.method_one,
            **kwargs
    ):
        super(AttentionLayer, self).__init__(**kwargs)
        self.left_num_heads = left_num_heads
        self.right_num_heads = right_num_heads
        self.hidden_size = hidden_size
        self.similarity_method = similarity_method
        self.merge_method = merge_method
        self.dense_query = partial_dense_layer.PartialDenseLayer(
            right_num_heads * hidden_size, left_num_heads * hidden_size)
        self.dense_key = partial_dense_layer.PartialDenseLayer(
            right_num_heads * hidden_size, left_num_heads * hidden_size)
        self.dense_value = partial_dense_layer.PartialDenseLayer(
            right_num_heads * hidden_size, left_num_heads * hidden_size)
        self.dense_output = partial_dense_layer.PartialDenseLayer(
            output_size, output_size)

    def call(
            self,
            inputs,
            **kwargs
    ):
        queries, keys, values = inputs
        args = (self.left_num_heads, self.right_num_heads, self.hidden_size)
        queries = separate_heads(self.dense_query(queries) / float(self.hidden_size), *args)
        keys = separate_heads(self.dense_key(keys) / float(self.hidden_size), *args)
        values = separate_heads(self.dense_value(values) / float(self.hidden_size), *args)
        weights = tf.math.softmax(self.similarity_method(
            tf.expand_dims(queries, -5),
            tf.expand_dims(keys, -6)) / float(self.hidden_size), axis=(-5))
        outputs = self.merge_method(weights, tf.expand_dims(values, -6))
        return self.dense_output(merge_heads(outputs, *args)) / float(self.hidden_size)
