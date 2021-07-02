import os

import Bio.SeqIO as SeqIO
import numpy as np
from tensorflow import keras


def load_data(seqs_path, labels_path):
    """
    Given the path to the fasta files of the sequence and labels
    returns a list of a tuple pair of the accession of sequence/label
    and a string of sequence/string.

    :param seqs_path: file path to fasta file with amino acid sequence data
    :param labels_path: file path to fasta file with binary label data
    :return: [(accession_seq, 'amino_acid_seq')], [(accession_label, 'label')]
            ex: [('QP106', 'AASSSDD'), ...], [('QP106', '00001111000'), ...]
    """

    # Load files
    seqs = []
    for record in SeqIO.parse(seqs_path, 'fasta'):
        accession = record.description.split('|')[0]
        seq = str(record.seq)
        seqs.append((accession, seq))

    labels = []
    for record in SeqIO.parse(labels_path, 'fasta'):
        accession = record.description.split('|')[0]
        label = str(record.seq)
        labels.append((accession, label))

    return seqs, labels


def convert_ohc(seqs):
    """
    Converts the given tuple of the accession and amino acid sequence string into
    a int list using sym_codes and then one hot encoded the int list using keras.util.to_categorical

    :param seqs: list of tuple pair of the accession of the amino acid sequence
    and the string of the amino acid sequence ex: [('QP106', 'AASSSDD'), ...]
    :return: 2D list of each one hot encoded amino acid sequence ex: [[0,0,0,0,0,0,1], ...]
    """
    x = []
    # convert string amino acid string characters into int list
    for _, seq in seqs:
        seq_idx = [sym_codes[sym] for sym in seq]
        x.append(seq_idx)

    # convert list into numpy array and one hot encode int array
    x = np.array(x)
    x = keras.utils.to_categorical(x, num_classes=len(sym_codes))
    return x


# Parameters
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Load data
train_seqs, train_labels = load_data('../inpainting_mobidb/out/unmasked_seq_file.fasta',
                                     '../inpainting_mobidb/out/label.fasta')
validation_seqs, validation_labels = ...
test_seqs, test_labels = ...

# convert character array to numpy array and one hot encode the sequences
x_train = convert_ohc(train_seqs)
...

y_train = np.array([list(label) for _, label in train_labels])
...

# model.fit(input,taget, batcsize, epochs)
# input = x, target = y, batchsize = whatever, epochs = whatever

# x needs to be numpy array for inputs
# 'MDS' -> numpy_array -> int array for chatacters -> [0, 5, 3]
# CATEGORY TABLE
# numerically encode the characters

# one hoch encode
# x = [[1,0,0], [0,1,0], [0,0,1]] to categorical in tensorflow, already made example

# going to be one layer nested further
#
#['MDS', 'SDM']
#[[0, 1, 2]
# [2, 1, 0]]

#[[1, 0, 0], [0, 1, 0], []]

# BEFORE


# label

# convert aa into number

# changing to double array using to categorical keras

#


# training the data
histories = []
validation_metrics = []
for i in range(10):
    # Fit
    print(f'EPOCH {i}')
    history = model.fit(train_batches, epochs=1)
    histories.append(history)

    # Evaluate
    total = 0
    l_count = 0
    for batch in validation_batches:
        true_labels = batch[1]
        pred_labels = model.predict(batch[0])
        for true_label, pred_label in zip(true_labels, pred_labels):
            idxs = true_label.sum(axis=1).astype(bool)
            total += idxs.sum()
            l_count += (np.argmax(true_label[idxs], axis=1) == np.argmax(pred_label[idxs], axis=1)).sum()
    validation_metrics.append({'total': total, 'l_count': l_count})

    print('LABEL ACCURACY:', l_count / total)
    print()

# Save model
if not os.path.exists('out/'):
    os.mkdir('out/')

model.save('out/model')
