"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def atomic_merge_operation(
        index,
        sequence,
        accumulation
):
    return [index + 1, sequence, tf.matmul(
        accumulation, sequence[..., index, :, :, :, :])]


def reduce_matmul(
        sequence
):
    condition = (lambda idx, seq, acc: idx < tf.shape(seq)[-5])
    body = atomic_merge_operation
    loop_vars = [tf.zeros([], dtype=tf.int32),
                 sequence,
                 tf.eye(tf.shape(sequence)[-1],
                        batch_shape=tf.shape(sequence[..., 0, :, :, :, :])[:(-2)])]
    return tf.while_loop(condition, body, loop_vars)[2]


def method_one(
        weights,
        sequence
):
    return reduce_matmul(tf.linalg.expm(weights * sequence))
