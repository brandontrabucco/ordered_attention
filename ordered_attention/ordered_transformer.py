"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from ordered_attention import partial_dense_layer
from ordered_attention import ordered_attention_layer


class OrderedTransformer(tf.keras.Model):

    def __init__(
            self,
            right_num_heads,
            left_num_heads,
            attention_hidden_size,
            dense_hidden_size,
            num_layers,
            hidden_size,
            output_size,
            **kwargs
    ):
        super(OrderedTransformer, self).__init__()
        self.input_layer = partial_dense_layer.PartialDenseLayer(
            hidden_size, hidden_size, use_bias=False)
        self.input_two_layer = partial_dense_layer.PartialDenseLayer(
            hidden_size, hidden_size, use_bias=False)
        self.attention_layers = [
            ordered_attention_layer.OrderedAttentionLayer(
                right_num_heads,
                left_num_heads,
                attention_hidden_size,
                hidden_size,
                use_mask=True,
                use_bias=False,
                **kwargs)
            for i in range(num_layers)]
        self.attention_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.dense_hidden_layers = [
            partial_dense_layer.PartialDenseLayer(
                dense_hidden_size, dense_hidden_size)
            for i in range(num_layers)]
        self.dense_output_layers = [
            partial_dense_layer.PartialDenseLayer(
                hidden_size, hidden_size)
            for i in range(num_layers)]
        self.dense_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.output_layer = partial_dense_layer.PartialDenseLayer(
            output_size, output_size, use_bias=False)

    def __call__(self, sequence, **kwargs):
        for (attend_layer,
                attend_norm_layer,
                hidden_layer,
                output_layer,
                dense_norm_layer) in zip(
                    self.attention_layers,
                    self.attention_norms,
                    self.dense_hidden_layers,
                    self.dense_output_layers,
                    self.dense_norms):
            sequence = attend_norm_layer(
                sequence + attend_layer([
                    sequence,
                    sequence,
                    sequence]))
            sequence = dense_norm_layer(
                sequence + output_layer(
                    tf.nn.relu(hidden_layer(sequence))))
        return self.output_layer(sequence)
