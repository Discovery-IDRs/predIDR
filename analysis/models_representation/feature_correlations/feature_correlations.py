"""Calculate correlations between learned and known features."""

import os
from math import floor

import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf


class BatchGenerator(tf.keras.utils.Sequence):
    """Label, batch, and pad protein sequence data.

    Only complete batches are returned, so a single epoch may not train on every example."""
    def __init__(self, records, batch_size, alphabet, shuffle=True):
        self.records = records
        self.indices = np.arange(len(self.records))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sym2idx = {}
        for idx, sym in enumerate(alphabet):
            self.sym2idx[sym] = idx
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches."""
        return floor(len(self.records) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        records = [self.records[i] for i in indices]
        max_len = max([len(record[0]) for record in records])
        x = np.zeros((self.batch_size, max_len))
        y = np.zeros((self.batch_size, max_len))
        for i, (syms, labels) in enumerate(records):
            x[i, :len(syms)] = [self.sym2idx.get(sym, 0) for sym in syms]
            y[i, :len(syms)] = [int(label) if label in ["0", "1"] else 2 for label in labels]

        x = tf.keras.utils.to_categorical(x, num_classes=len(self.sym2idx))
        y = tf.keras.utils.to_categorical(y, num_classes=3)
        for i, (syms, _) in enumerate(records):
            x[i, len(syms):, :] = 0
            y[i, len(syms):, :] = 0

        return x, y

    def on_epoch_end(self):
        """Shuffles data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class MaskedConv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.supports_masking = True


def load_data(seqs_path, labels_path):
    # Load files
    seqs = {}
    for record in SeqIO.parse(seqs_path, 'fasta'):
        accession = record.description.split('|')[0]
        seq = str(record.seq)
        seqs[accession] = seq
    labels = {}
    for record in SeqIO.parse(labels_path, 'fasta'):
        accession = record.description.split('|')[0]
        label = str(record.seq)
        labels[accession] = label

    # Bundle seqs and labels into single object
    records = []
    for accession, seq in seqs.items():
        records.append((seq, labels[accession]))

    return records


alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Load data
train_records = load_data('../../mobidb-pdb_validation/split_data/out/train_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/train_labels.fasta')
validation_records = load_data('../../mobidb-pdb_validation/split_data/out/validation_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/validation_labels.fasta')
test_records = load_data('../../mobidb-pdb_validation/split_data/out/test_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/test_labels.fasta')

# Batch data
train_batches = BatchGenerator(train_records, 32, alphabet)
validation_batches = BatchGenerator(validation_records, 32, alphabet)
test_batches = BatchGenerator(test_records, 32, alphabet)

# Load model
model_data = [('../../models/mobidb-pdb_cnn_6_1/out_model/mobidb-pdb_cnn_6_1.h5', 'conv1d2')]
for model_path, layer_name in model_data:
    model = tf.keras.models.load_model(model_path, custom_objects={"MaskedConv1D": MaskedConv1D})
    layer_dict = {layer.name: layer for layer in model.layers}
    layer = layer_dict[layer_name]

# for each example in the data
# calculate feature maps using model
# calculate feature maps using known features

# calculate correlations between all pairs of features in the two sets
# visualize it somehow