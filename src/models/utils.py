"""Utilities for training models."""

from math import ceil, floor

import numpy as np
import tensorflow as tf
from src.utils import read_fasta


class BatchGenerator(tf.keras.utils.Sequence):
    """Label, batch, and pad protein sequence data.

    Only complete batches are returned, so a single epoch may not train on every example."""
    def __init__(self, records, batch_size, alphabet, weights, shuffle=True, all_records=False):
        self.records = records
        self.indices = np.arange(len(self.records))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_records = all_records
        self.sym2idx = {}
        for idx, sym in enumerate(alphabet):
            self.sym2idx[sym] = idx
        self.weights = weights
        self.label2idx = {}
        for idx, label in enumerate(weights):
            self.label2idx[label] = idx
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches."""
        if self.all_records:
            return ceil(len(self.records) / self.batch_size)
        else:
            return floor(len(self.records) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        if index >= len(self):
            raise IndexError('batch index out of range')
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        records = [self.records[i] for i in indices]
        batch_size = len(records)
        max_len = max([len(record[0]) for record in records])
        x = np.zeros((batch_size, max_len))
        y = np.zeros((batch_size, max_len))
        w = np.zeros((batch_size, max_len))
        for i, (syms, labels) in enumerate(records):
            x[i, :len(syms)] = [self.sym2idx.get(sym, 0) for sym in syms]
            y[i, :len(syms)] = [self.label2idx.get(label, 0) for label in labels]
            w[i, :len(syms)] = [self.weights.get(label, 0) for label in labels]

        x = tf.keras.utils.to_categorical(x, num_classes=len(self.sym2idx))
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.label2idx))
        for i, (syms, _) in enumerate(records):
            x[i, len(syms):, :] = 0
            y[i, len(syms):, :] = 0

        return x, y, w

    def on_epoch_end(self):
        """Shuffles data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data(seqs_path, labels_path):
    # Load files
    seqs = {}
    for header, seq in read_fasta(seqs_path):
        accession = header.split('|')[0]
        seqs[accession] = seq
    labels = {}
    for header, label in read_fasta(labels_path):
        accession = header.split('|')[0]
        labels[accession] = label

    # Bundle seqs and labels into single object
    records = []
    for accession, seq in seqs.items():
        records.append((seq, labels[accession]))

    return records
