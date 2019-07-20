"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import argparse
import os
from ordered_attention.ordered_transformer import OrderedTransformer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="vocab.txt")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="training_features/")
    parser.add_argument(
        "--window_size",
        type=int,
        default=100)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=32)
    args = parser.parse_args()

    with tf.io.gfile.GFile(args.vocab_file, "r") as f:
        reverse_vocab = f.read().strip().split("\n")

    words = tf.constant(reverse_vocab)
    ids = tf.range(tf.size(words))
    words_to_ids = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys=words, values=ids),
        default_value=tf.constant(1),
        name="words_to_ids")
    ids_to_words = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys=ids, values=words),
        default_value=tf.constant("<unk>"),
        name="ids_to_words")

    feature_files = tf.io.gfile.glob(
        os.path.join(args.features_dir, "*.txt.npy"))
    features = [np.load(
        file) for file in feature_files]

    splits = np.zeros([0, 3], dtype=np.int32)
    for i, ids in enumerate(features):
        splits = np.concatenate([splits, np.stack([
            np.full([ids.size - args.window_size - 1], i),
            np.arange(ids.size - args.window_size - 1),
            args.window_size + 1 + np.arange(ids.size - args.window_size - 1)], 1)], 0)

    def generate_split_backend(split):
        return (features[split[0]][split[1]:(split[2] - 1)],
                features[split[0]][(1 + split[1]):split[2]])

    def generate_split(split):
        begin, end = tf.py_function(generate_split_backend, [split], [tf.int32, tf.int32])
        return tf.reshape(begin, [args.window_size]), tf.reshape(end, [args.window_size])

    embeddings = tf.Variable(
        tf.initializers.glorot_uniform()([len(reverse_vocab), 16, 16]))
    model = OrderedTransformer(
        right_num_heads=2,
        left_num_heads=2,
        attention_hidden_size=4,
        dense_hidden_size=16,
        num_layers=2,
        hidden_size=16,
        output_size=16)
    logits = tf.keras.layers.Dense(len(reverse_vocab))

    optimizer = tf.keras.optimizers.Adam()

    dataset = tf.data.Dataset.from_tensor_slices(splits)
    dataset = dataset.map(
        generate_split, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(args.batch_size * 32)
    dataset = dataset.repeat(args.num_epochs)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
        "/gpu:0", buffer_size=tf.data.experimental.AUTOTUNE))

    for begin, end in dataset:

        def loss_function():
            x = tf.nn.embedding_lookup(embeddings, begin)
            x = model(x)
            x = logits(tf.reshape(x, tf.concat([tf.shape(x)[:(-2)], [-1]], 0)))
            return tf.losses.sparse_categorical_crossentropy(end, x)

        optimizer.minimize(
            loss_function,
            [embeddings] + logits.trainable_variables + model.trainable_variables)
