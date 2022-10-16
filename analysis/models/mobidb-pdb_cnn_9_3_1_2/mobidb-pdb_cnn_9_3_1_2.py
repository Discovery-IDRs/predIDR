# software setup:
# python 3.7
# tensorflow 2.1.0
# cuda 10.1
# cudnn 7.6

# Purpose:
# Examine the performance of a multiheaded model with a 500 kernal head that
# has only 1 1D conv layer and a 100 kernal head that has only 1 1D conv
# layer with a x75 disorder weight trained for x100 epochs.

# Architecture:
# disorder weight: x75
# layers: Head 1
#             x1 1D conv layers with 128 filter and 50 kernal
#         Head 2
#             x1 1D conv layers with 128 filter and 100 kernal
#         x1 dense layer with 128 nodes
# epoch: 100

# Significance:
# Building upon mobidb-pdb_cnn_9_3_2, this was part of the further development
# of mobidb-pdb_cnn_3_6_1 which was undertaken due to interest in the use of a
# x75 disorder weight. x75 disorder weight was of interest because it was the
# smallest weight tested which seemed to properly counteract overfitting. Other
# models tested did implement even larger disorder weights and did not appear 
# to demonstrate any abnormal behavior while also counteracting overfitting.
# However, there was a desire to limit the size of the disorder weight as much
# as possible in order to minimize manipulation of the dataset. This, along with
# the fact that these other models did not appear to demonstrate any significant
# increases in performance, made x75 disorder weight appear to be the best choice
# going forward. This particular model was part of a series of models which were
# specifically looking at testing out various different implementations of 
# multiheaded architectures. It was hoped that the increased complexity of a
# multiheaded architecture could lead to performance increases. In general,
# this series of models seemed to demonstrate that the performance of a multiheaded
# model was simply the average of the performances of the individual models which 
# made up said multiheaded model. Note that this was not always true. This model
# looked to try to basically do the equivalent of combining a model that had a
# kernal of 50 with a model that had a kernal of 100 (these were the two most 
# successful candidates from the 7 series of models). Additionally, this model
# was constructed so that each head had only 1 1D conv layer. It was hoped that this
# would allow the model to have greater generalizability while also taking advantage
# of the additional perspectives offered by the multiple heads. This model had
# decreased accuracy, MCC, specificity, precision, and F1 performance and increased
# sensitivity performance when compared to both its counterpart mobidb-pdb_cnn_9_3_2
# and baseline model mobidb-pdb_cnn_6_2. Its training curves appear to be normal 
# (features rapid initial growth of accuracy and specificity that eventually levels 
# off, something which would be expected from a model that is progressively learning
# to make better predictions; this is in contrast to the overfitting models which 
# demonstrate abnormally good accuracy and specificity right from the start of 
# training). While the performance of this model appears to be worse when compared
# to its counterparts, it is still being considered for further investigation due
# to the fact that it is somewhat unique in having a balanced sensitivity and 
# specificity.


import os
from math import floor

import Bio.SeqIO as SeqIO
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append('../../../src')
from legacy_metrics import *

from pandas.core.common import flatten

model_name = "mobidb-pdb_cnn_9_3_1_2"

class BatchGenerator(keras.utils.Sequence):
    """Label, batch, and pad protein sequence data.

    Only complete batches are returned, so a single epoch may not train on every example."""
    def __init__(self, records, batch_size, sym_codes, shuffle=True):
        self.records = records
        self.indices = np.arange(len(self.records))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctable = {}  # Category table
        for i, sym_code in enumerate(sym_codes):
            self.ctable[sym_code] = i
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
            x[i, :len(syms)] = [self.ctable.get(sym, 0) for sym in syms]
            y[i, :len(syms)] = [int(label) if label in ["0", "1"] else 2 for label in labels]

        sample_weights = np.ones((self.batch_size, max_len))
        sample_weights[y == 1] = 75.0
        sample_weights[y == 2] = 0.0

        x = keras.utils.to_categorical(x, num_classes=len(self.ctable))
        y = keras.utils.to_categorical(y, num_classes=3)
        for i, (syms, _) in enumerate(records):
            x[i, len(syms):, :] = 0
            y[i, len(syms):, :] = 0

        return x, y, sample_weights

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

def decode_sym(batch, sym_codes):
    """Decodes residue symbols in batch of proteins from encoded form."""
    ctable, i = {}, 0
    for sym_code in sym_codes:
        ctable[i] = sym_code
        i += 1

    decoded_sym = []
    for protein in batch:
        decoded_protein = []
        for residue in protein:
            if sum(residue) == 0:
                pass
            else:
                decoded_residue = ctable[np.argmax(residue)]
                decoded_protein.append(decoded_residue)
        decoded_sym.append("".join(decoded_protein))

    return decoded_sym

def decode_label_by_protein(batch_to_be_decoded, original_batch):
    """
    Decodes residue labels in batch of proteins from encoded form.
    Outputs list of str where each str is labels of one protein.

    batch_to_be_decoded:
    batch of labels which want to be decoded

    original_batch:
    original batch of labels generated by BatchGenerator which
    batch_to_be_decoded is derived from (this is needed in order
    to remove unwanted masked values)
    """
    decoded_labels = []
    for x in np.arange(len(original_batch)):
        protein = original_batch[x]
        decoded_protein = []
        for y in np.arange(len(protein)):
            residue = protein[y]
            if sum(residue) == 0:
                pass
            else:
                decoded_residue = np.argmax(batch_to_be_decoded[x][y])
                if decoded_residue == 2:
                    decoded_protein.append("-")
                else:
                    decoded_protein.append(str(decoded_residue))
        decoded_labels.append("".join(decoded_protein))
    return decoded_labels

def decode_label_to_lst(batch_to_be_decoded, original_batch):
    """
    Decodes residue labels in batch of proteins from encoded form.
    Outputs list of int where each int is label for one residue.
    Does not include residues labeled "-" in output.

    batch_to_be_decoded:
    batch of labels which want to be decoded

    original_batch:
    original batch of labels generated by BatchGenerator which
    batch_to_be_decoded is derived from (this is needed in order
    to remove unwanted masked values)
    """
    decoded_labels = []
    for x in np.arange(len(original_batch)):
        protein = original_batch[x]
        for y in np.arange(len(protein)):
            residue = protein[y]
            if sum(residue) == 0:
                pass
            elif np.argmax(residue) == 2:
                pass
            else:
                decoded_residue = np.argmax(batch_to_be_decoded[x][y])
                decoded_labels.append(decoded_residue)

    return decoded_labels

class MaskedConv1D(keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.supports_masking = True

# Parameters
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Load data
train_records = load_data('../../mobidb-pdb_validation/split_data/out/train_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/train_labels.fasta')
validation_records = load_data('../../mobidb-pdb_validation/split_data/out/validation_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/validation_labels.fasta')
test_records = load_data('../../mobidb-pdb_validation/split_data/out/test_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/test_labels.fasta')

# Batch data
train_batches = BatchGenerator(train_records, 32, sym_codes)
validation_batches = BatchGenerator(validation_records, 32, sym_codes)
test_batches = BatchGenerator(test_records, 32, sym_codes)

# Build model
inputs = keras.layers.Input(shape=(None, 20), name='input1')

# 50 kernal head
mask1 = layers.Masking(mask_value=0, name='mask11')(inputs)
conv11 = MaskedConv1D(128, 50, padding='same', activation='relu', name='conv1d11')(mask1)

# 100 kernal head
mask2 = layers.Masking(mask_value=0, name='mask21')(inputs)
conv21 = MaskedConv1D(128, 100, padding='same', activation='relu', name='conv1d21')(mask2)

merged = layers.concatenate([conv11,conv21])

dense1 = layers.Dense(128, activation="relu")(merged)

outputs = layers.Dense(3, activation='softmax', name='output1')(dense1)


model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
model.compile(loss='binary_crossentropy', optimizer='adam', weighted_metrics=['binary_accuracy'], sample_weight_mode="temporal")
model.summary()

num_epochs = 100

# Train model
# Epochs are written explicitly in a training loop since Keras
# does not support generators for calculating validation metrics

histories = []

metrics_df = pd.DataFrame()

for i in range(num_epochs):
    # Fit
    print(f'EPOCH {i}')
    history = model.fit(train_batches, epochs=1)
    histories.append(history)

    # Evaluate
    true_labels = []
    pred_labels = []
    for batch in validation_batches:
        true_labels.append(decode_label_to_lst(batch[1], batch[1]))
        pred_labels.append(decode_label_to_lst(model.predict(batch[0]), batch[1]))
    true_labels = list(flatten(true_labels))
    pred_labels = list(flatten(pred_labels))

    batch_metrics_df = get_binary_metrics(true_labels, pred_labels)
    print("Label Accuracy:", batch_metrics_df.loc[0].at["Accuracy"])

    metrics_df = metrics_df.append(batch_metrics_df, ignore_index = True)

x_axis_ticks = np.arange(num_epochs)
y_axis_ticks = np.arange(11)/10

metrics_fig = metrics_df.plot(figsize = (12, 5), use_index = True, title = model_name + " training metrics",
                              xticks = x_axis_ticks, yticks = y_axis_ticks)

metrics_fig = metrics_fig.legend(bbox_to_anchor=(1, 1))
metrics_fig = metrics_fig.get_figure()

# Save model
if not os.path.exists('out_model/'):
    os.mkdir('out_model/')

model.save('out_model/' + model_name + ".h5")

# Save metrics df
if not os.path.exists('out_metrics/'):
    os.mkdir('out_metrics/')

metrics_df.to_csv("out_metrics/training_metrics.csv", index=False)
