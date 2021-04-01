"""Test of building and training of a simple LSTM under the Keras API."""

from itertools import product
from math import floor

import Bio.SeqIO as SeqIO
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class BatchGenerator(keras.utils.Sequence):
    """Label, batch, and pad protein sequence data.

    Only complete batches are returned, so a single epoch may not train on every example."""
    def __init__(self, records, batch_size, sym_codes, label_codes, shuffle=True):
        self.records = records
        self.indices = np.arange(len(self.records))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctable, i = {}, 1  # Category table
        for p in product(sym_codes, label_codes):
            self.ctable[p] = i
            i += 1
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
        for i, (syms, labels) in enumerate(records):
            x[i, :len(syms)] = [self.ctable[(sym, label)] for sym, label in zip(syms, labels)]
        x = keras.utils.to_categorical(x, num_classes=len(self.ctable)+1)
        return x, x

    def on_epoch_end(self):
        """Shuffles data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


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


def decode(batch, sym_codes, label_codes):
    """Decodes a one-hot encoded vector or a probability distribution over the same categories."""
    ctable, i = {0: ('X', 'X')}, 1
    for p in product(sym_codes, label_codes):
        ctable[i] = p
        i += 1
    records = []
    for indices in np.argmax(batch, axis=2):
        syms = []
        labels = []
        for index in indices:
            sym, label = ctable[index]
            syms.append(sym)
            labels.append(label)
        records.append((''.join(syms), ''.join(labels)))
    return records


# Parameters
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
label_codes = ['0', '1']
category_num = len(sym_codes) * len(label_codes) + 1

# Load data
train_records = load_data('../../mobidb_validation/split_data/out/train_as_fasta.fasta', '../../mobidb_validation/split_data/out/train_labels_as_fasta.fasta')
validation_records = load_data('../../mobidb_validation/split_data/out/val_as_fasta.fasta', '../../mobidb_validation/split_data/out/val_labels_as_fasta.fasta')
test_records = load_data('../../mobidb_validation/split_data/out/test_as_fasta.fasta', '../../mobidb_validation/split_data/out/test_labels_as_fasta.fasta')

# Batch data
train_batches = BatchGenerator(train_records, 32, sym_codes, label_codes)
validation_batches = BatchGenerator(validation_records, 32, sym_codes, label_codes)
test_batches = BatchGenerator(test_records, 32, sym_codes, label_codes)

# Build model
model = keras.Sequential(name='simple_lstm')
model.add(layers.Masking(mask_value=0, input_shape=(None, category_num), name='mask1'))
model.add(layers.LSTM(128, return_sequences=True, name='lstm1'))
model.add(layers.Dense(category_num, activation='softmax', name='dense1'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()

# Train model
model.fit(train_batches, epochs=1)

# Evaluate metrics
sym_num = 0
sym_true_num = 0
label_num = 0
label_true_num = 0
for batch in test_batches:
    true_records = decode(batch[0], sym_codes, label_codes)
    pred_records = decode(model.predict(batch), sym_codes, label_codes)
    for true_record, pred_record in zip(true_records, pred_records):
        true_seq, true_labels = true_record
        pred_seq, pred_labels = pred_record
        for true_sym, pred_sym in zip(true_seq.rstrip('X'), pred_seq):
            sym_num += 1
            if true_sym == pred_sym:
                sym_true_num += 1
        for true_label, pred_label in zip(true_labels.rstrip('X'), pred_labels):
            label_num += 1
            if true_label == pred_label:
                label_true_num += 1

print('SYMBOL ACCURACY', sym_true_num / sym_num)
print('LABEL ACCURACY', label_true_num / label_num)
