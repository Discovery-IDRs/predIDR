"""Utilities for inpainting_mobidb"""

import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf


def load_data(seq_path, label_path, sym2idx):
    """Return sequences and labels from FASTA files as arrays.

    :param seq_path: path for FASTA file of amino acid sequences
    :param label_path: path for FASTA file of labels of amino acid sequences where disordered residues are labeled as 1
        and ordered residues are labeled as 0
    :param sym2idx: mapping from symbol to index as dictionary
    :return: array of one hot encoded sequences and array of labels
    """
    ohes = []
    labels = []
    for record_seq, record_label in zip(SeqIO.parse(seq_path, 'fasta'), SeqIO.parse(label_path, 'fasta')):
        # Convert string seq to one-hot-encoded seq
        ohe = seq_to_OHE(str(record_seq.seq), sym2idx)
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


def seq_to_OHE(seq, sym2idx):
    """Return amino acid sequence as one-hot-encoded vector.

    :param sym2idx:
    :param seq: amino acid sequence as string
    :return: one-hot-encoded string as 2D array
    """
    ohe = np.array([sym2idx[sym] for sym in seq])
    ohe = tf.keras.utils.to_categorical(ohe, num_classes=len(sym2idx), dtype='int32')
    return ohe


def OHE_to_seq(ohe, idx2sym):
    """Return one-hot-encoded vector as amino acid sequence.

    :param idx2sym:
    :param ohe: one-hot-encoded string as 2D array
    :return: amino acid sequence as list
    """
    x = np.argmax(ohe, axis=1)
    seq = []
    for idx in x:
        sym = idx2sym[idx]
        seq.append(sym)
    return seq
