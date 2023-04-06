# software setup:
# python 3.7
# tensorflow 2.1.0
# cuda 10.1
# cudnn 7.6

# Purpose:
# Examine how increasing the epoch to 100 in combination with a x75 disorder weight
# affects the performance. Model was rerun to also save initial weights.

# Architecture:
# disorder weight: x75
# layers: x2 1D conv layers with 128 filter and 20 kernal
# epoch: 100

# Significance:
# When the 3 series of models were being experimented with, there was a concern
# that training was being ended too early, preventing full observation of the
# learning behaviors of the models. The 6 series were an attempt to catch any
# behavior missed in the 3 series by extending the training to 100 epochs instead
# of 50. In general, the behavior of the 6 series mirrored that of their 3 series
# counterparts. The training curves of the 6 series did not seem to show that
# any new learning behavior was being missed and seemed to indicate that the
# training metrics had mostly stabilized at 50 epochs. The 6 series did as a rule
# each have lower sensitivity and higher accuracy, MCC, specificity, precision, 
# and F1 compared to their equivalent 3 series counterpart. Interestingly, this
# seems to be similar to the effect produce by using a lower disorder weight,
# possibly indicating that increasing epochs may increase overfitting (this is
# however only a tentative guess). Like other 6 series models, the behavior of
# this model resembled that of its 3 series counterpart mobidb_pdb_cnn_3_6_1 as
# described above. The training curves of this model closely follow that of 
# mobidb-pdb_cnn_3_6_1 in the beginning and continue the trend of leveling off
# which was seen at 50 epochs in the mobidb-pdb_cnn_3_6_1 training curves. This 
# model was ultimately used along with mobidb-pdb_cnn_3_6_1 (which this model 
# was based on) as the templates upon which further models were developed. 
# Naturally, this meant that the performance of this model also served alongside
# the performance of mobidb-pdb_cnn_3_6_1 as a baseline to which the performances
# of other model architectures were compared to. The reason for choosing this 
# model and mobidb-pdb_cnn_3_6_1 to serve as templates and baselines was because
# x75 disorder weight was the smallest weight tested which seemed to properly 
# counteract overfitting. While other models which implemented even larger 
# disorder weights did not appear to demonstrate any abnormal behavior and 
# did also counteract overfitting, there was a desire to limit the size of 
# the disorder weight as much as possible in order to minimize manipulation 
# of the dataset. This along with the fact that these other models did not 
# appear to demonstrate any significant increases in performance meant that
# they were ultimately passed over for further consideration and development
# in favor of this model and mobidb-pdb_cnn_3_6_1.


import os
from math import floor

import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append('../../../src')
from legacy_metrics import *

from pandas.core.common import flatten

model_name = "mobidb-pdb_cnn_6_2_re"

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
        
# Set random seet
tf.random.set_seed(1)

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
x = layers.Masking(mask_value=0, name='mask1')(inputs)
x = MaskedConv1D(128, 20, padding='same', activation='relu', name='conv1d1')(x)
x = MaskedConv1D(128, 20, padding='same', activation='relu', name='conv1d2')(x)
outputs = layers.Dense(3, activation='softmax', name='output1')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
model.compile(loss='binary_crossentropy', optimizer='adam', weighted_metrics=['binary_accuracy'], sample_weight_mode="temporal")
model.summary()

# Save untrained model
if not os.path.exists('out_model/'):
    os.mkdir('out_model/')
    
model.save('out_model/' + model_name + "_untrained.h5")

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

# Save trained model
model.save('out_model/' + model_name + "_trained.h5")

# Save metrics df
if not os.path.exists('out_metrics/'):
    os.mkdir('out_metrics/')

metrics_df.to_csv("out_metrics/training_metrics.csv", index=False)
