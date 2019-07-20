"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from ordered_attention import partial_dense_layer
from ordered_attention import ordered_attention_layer


class ConditionalOrderedTransformer(tf.keras.Model):

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
        super(ConditionalOrderedTransformer, self).__init__()
        self.input_one_layer = partial_dense_layer.PartialDenseLayer(
            hidden_size, hidden_size, use_bias=False)
        self.input_two_layer = partial_dense_layer.PartialDenseLayer(
            hidden_size, hidden_size, use_bias=False)
        self.attention_one_layers = [
            ordered_attention_layer.OrderedAttentionLayer(
                right_num_heads,
                left_num_heads,
                attention_hidden_size,
                hidden_size,
                use_mask=False,
                use_bias=False,
                **kwargs)
            for i in range(num_layers)]
        self.attention_one_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.dense_one_hidden_layers = [
            partial_dense_layer.PartialDenseLayer(
                dense_hidden_size, dense_hidden_size)
            for i in range(num_layers)]
        self.dense_one_output_layers = [
            partial_dense_layer.PartialDenseLayer(
                hidden_size, hidden_size)
            for i in range(num_layers)]
        self.dense_one_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.attention_two_layers = [
            ordered_attention_layer.OrderedAttentionLayer(
                right_num_heads,
                left_num_heads,
                attention_hidden_size,
                hidden_size,
                use_mask=True,
                use_bias=False,
                **kwargs)
            for i in range(num_layers)]
        self.attention_two_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.dense_two_hidden_layers = [
            partial_dense_layer.PartialDenseLayer(
                dense_hidden_size, dense_hidden_size)
            for i in range(num_layers)]
        self.dense_two_output_layers = [
            partial_dense_layer.PartialDenseLayer(
                hidden_size, hidden_size)
            for i in range(num_layers)]
        self.dense_two_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.attention_three_layers = [
            ordered_attention_layer.OrderedAttentionLayer(
                right_num_heads,
                left_num_heads,
                attention_hidden_size,
                hidden_size,
                use_mask=False,
                use_bias=False,
                **kwargs)
            for i in range(num_layers)]
        self.attention_three_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.dense_three_hidden_layers = [
            partial_dense_layer.PartialDenseLayer(
                dense_hidden_size, dense_hidden_size)
            for i in range(num_layers)]
        self.dense_three_output_layers = [
            partial_dense_layer.PartialDenseLayer(
                hidden_size, hidden_size)
            for i in range(num_layers)]
        self.dense_three_norms = [
            tf.keras.layers.BatchNormalization()
            for i in range(num_layers)]
        self.output_layer = partial_dense_layer.PartialDenseLayer(
            output_size, output_size, use_bias=False)

    def __call__(self, inputs, **kwargs):
        sequence_one, sequence_two = inputs
        for (attend_one_layer,
                attend_norm_one_layer,
                hidden_layer,
                output_layer,
                dense_norm_layer) in zip(
                    self.attention_one_layers,
                    self.attention_one_norms,
                    self.dense_one_hidden_layers,
                    self.dense_one_output_layers,
                    self.dense_one_norms):
            sequence_one = attend_norm_one_layer(
                sequence_one + attend_one_layer([
                    sequence_one,
                    sequence_one,
                    sequence_one]))
            sequence_one = dense_norm_layer(
                sequence_one + output_layer(
                    tf.nn.relu(hidden_layer(sequence_one))))
        for (attend_two_layer,
                attend_norm_two_layer,
                attend_three_layer,
                attend_norm_three_layer,
                hidden_layer,
                output_layer,
                dense_norm_layer) in zip(
                    self.attention_two_layers,
                    self.attention_two_norms,
                    self.attention_three_layers,
                    self.attention_three_norms,
                    self.dense_one_hidden_layers,
                    self.dense_one_output_layers,
                    self.dense_one_norms):
            sequence_two = attend_norm_two_layer(
                sequence_two + attend_two_layer([
                    sequence_two,
                    sequence_two,
                    sequence_two]))
            sequence_two = attend_norm_three_layer(
                sequence_two + attend_three_layer([
                    sequence_two,
                    sequence_one,
                    sequence_one]))
            sequence_two = dense_norm_layer(
                sequence_two + output_layer(
                    tf.nn.relu(hidden_layer(sequence_two))))
        return self.output_layer(sequence_two)
