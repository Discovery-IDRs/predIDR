"""Build and run CNN to inpaint target disordered regions using only context."""

import os
from math import floor

import Bio.SeqIO as SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf


class BatchGenerator(tf.keras.utils.Sequence):
    """Label, batch, and pad protein sequence data.

    Only complete batches are returned, so a single epoch may not train on every example.
    """
    def __init__(self, context, seq, weight, shuffle=True, seed=None):
        if len(context) != len(seq) != len(weight):
            raise ValueError('context, seq, weight are unequal lengths')
        self.context = context
        self.target = seq
        self.weight = weight
        self.indices = np.arange(len(self.context))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches."""
        return floor(len(self.context) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        indices_batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        context_batch = self.context[indices_batch]
        target_batch = self.target[indices_batch]
        weight_batch = self.weight[indices_batch]

        return context_batch, target_batch, weight_batch

    def on_epoch_end(self):
        """Shuffles data after each epoch."""
        if self.shuffle:
            self.rng.shuffle(self.indices)


# Utility functions
def load_data(seq_path, label_path):
    """Return sequences and labels from FASTA files as arrays.

    :param seq_path: path for FASTA file of amino acid sequences
    :param label_path: path for FASTA file of labels of amino acid sequences where disordered residues are labeled as 1
        and ordered residues are labeled as 0
    :return: array of one hot encoded sequences and array of labels
    """
    ohes = []
    labels = []
    for record_seq, record_label in zip(SeqIO.parse(seq_path, 'fasta'), SeqIO.parse(label_path, 'fasta')):
        # Convert string seq to one-hot-encoded seq
        ohe = seq_to_OHE(str(record_seq.seq))
        ohes.append(ohe)

        # Convert string label to int label
        label = [int(sym) for sym in record_label]
        labels.append(label)

    return np.array(ohes), np.array(labels)


def get_context_weight(ohe, label):
    """Return context and weight from one-hot-encoded sequences and labels.

    :param ohe: one hot ended 2D arrays of sequences
    :param label: array of labels corresponding to seq_ohc
    :return: context and weight according to ohc and label
    """
    weight = np.expand_dims(label, axis=2)  # weight is binary classification of data (1:target 0: context)
    context = (np.invert(weight) + 2) * ohe  # context is the opposite of the weight (0:target 1: context)

    return context, weight


def seq_to_OHE(seq):
    """Return amino acid sequence as one-hot-encoded vector.

    :param seq: amino acid sequence as string
    :return: one-hot-encoded string as 2D array
    """
    ohe = np.array([sym2idx[sym] for sym in seq])
    ohe = tf.keras.utils.to_categorical(ohe, num_classes=len(alphabet), dtype='int32')
    return ohe


def OHE_to_seq(ohe):
    """Return one-hot-encoded vector as amino acid sequence.

    :param ohe: one-hot-encoded string as 2D array
    :return: amino acid sequence as list
    """
    x = np.argmax(ohe, axis=1)
    seq = []
    for idx in x:
        sym = alphabet[idx]
        seq.append(sym)
    return seq


# MODELS
def make_generative_model():
    """Return generative model.

    DCGAN architecture from "Protein Loop Modeling Using Deep Generative Adversarial Network."
    10.1109/ICTAI.2017.00166

    :return: model instance of generative model
    """
    # Convolution
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(180, len(alphabet))))

    model.add(tf.keras.layers.Conv1D(2, 2, strides=1, padding='same', name='C1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(4, 2, strides=1, padding='same', name='C2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(8, 2, strides=1, padding='same', name='C3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(16, 2, strides=1, padding='same', name='C4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Deconvolution
    model.add(tf.keras.layers.Conv1DTranspose(8, 2, strides=1, padding='same', name='D1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(4, 2, strides=1, padding='same', name='D2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(2, 2, strides=1, padding='same', name='D3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Last layer transforms filters to probability classes
    model.add(tf.keras.layers.Conv1DTranspose(len(alphabet), 3, strides=1, padding='same', activation='softmax', name='D4'))

    return model


# PARAMETERS
batch_size = 30
epoch_num = 300
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}
idx2sym = {idx: sym for idx, sym in enumerate(alphabet)}


# MODEL
generator = make_generative_model()
generator.summary()
print()  # Newline after model summary

generator.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

# LOAD DATA AND TRAIN
train_seq_path = '../split_data/out/train_seqs.fasta'
train_label_path = '../split_data/out/train_labels.fasta'
valid_seq_path = '../split_data/out/validation_seqs.fasta'
valid_label_path = '../split_data/out/validation_labels.fasta'

train_seq, train_label = load_data(train_seq_path, train_label_path)
valid_seq, valid_label = load_data(valid_seq_path, valid_label_path)

train_context, train_weight = get_context_weight(train_seq, train_label)
valid_context, valid_weight = get_context_weight(valid_seq, valid_label)

train_batches = BatchGenerator(train_context, train_seq, train_weight, seed=1)
valid_batches = BatchGenerator(valid_context, valid_seq, valid_weight, seed=1)

history = generator.fit(train_batches, validation_data=valid_batches,
                        epochs=epoch_num, batch_size=batch_size)

# SAVE DATA
if not os.path.exists("out/"):
    os.mkdir("out/")

pd.DataFrame(history.history).to_csv('out/metrics.tsv', index_label='epoch', sep='\t')
generator.save('out/generator_model.h5')
