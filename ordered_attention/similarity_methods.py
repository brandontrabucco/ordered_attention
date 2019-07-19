"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def method_one(
        left_matrix,
        right_matrix
):
    return tf.reduce_sum(left_matrix * right_matrix,
                         axis=[-2, -1], keepdims=True)


def method_two(
        left_matrix,
        right_matrix
):
    return 1.0 - tf.reduce_sum(tf.math.square(left_matrix - right_matrix),
                               axis=[-2, -1], keepdims=True)


def method_three(
        left_matrix,
        right_matrix,
        order=2
):
    return tf.linalg.norm(tf.matmul(left_matrix, right_matrix),
                          ord=order, axis=[-2, -1], keepdims=True)


def method_four(
        left_matrix,
        right_matrix,
        order=2
):
    return tf.linalg.norm(tf.matmul(left_matrix, right_matrix) +
                          tf.matmul(right_matrix, left_matrix),
                          ord=order, axis=[-2, -1], keepdims=True)


def method_five(
        left_matrix,
        right_matrix,
        order=2
):
    return 1.0 - tf.linalg.norm(tf.matmul(left_matrix, right_matrix) -
                                tf.matmul(right_matrix, left_matrix),
                                ord=order, axis=[-2, -1], keepdims=True)
