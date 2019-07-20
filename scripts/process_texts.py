"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import nltk
import numpy as np
import argparse
import os
from collections import defaultdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=5)
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="vocab.txt")
    parser.add_argument(
        "--texts_dir",
        type=str,
        default="training/")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="training_features/")
    args = parser.parse_args()

    text_files = tf.io.gfile.glob(
        os.path.join(args.texts_dir, "*.txt"))
    text_tokens = []
    for file in text_files:
        with tf.io.gfile.GFile(file, "r") as f:
            text_tokens.append(
                nltk.word_tokenize(f.read().strip().lower()))

    token_frequencies = defaultdict(int)
    for text in text_tokens:
        for token in text:
            token_frequencies[token] += 1
    sorted_tokens, counts = list(zip(*list(sorted(
        token_frequencies.items(),
        key=(lambda x: x[1]),
        reverse=True))))
    split = 0
    for split, c in enumerate(counts):
        if c < args.min_frequency:
            break
    reverse_vocab = ("<pad>", "<unk>", "<start>", "<end>") + sorted_tokens[:split]
    with tf.io.gfile.GFile(args.vocab_file, "w") as f:
        f.write("\n".join(reverse_vocab))

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

    tf.io.gfile.makedirs(args.features_dir)
    for text, file in zip(text_tokens, text_files):
        np.save(
            os.path.join(args.features_dir, os.path.basename(file) + ".npy"),
            words_to_ids.lookup(tf.constant(text)).numpy())
