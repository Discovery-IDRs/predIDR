import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


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
        seq_idx = [sym_codes.index(sym) for sym in seq]
        x.append(seq_idx)

    # convert list into numpy array and one hot encode int array
    x = np.array(x)
    x = keras.utils.to_categorical(x, num_classes=len(sym_codes))
    return x


# Parameters
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Load data
train_seqs, train_labels = load_data('../inpainting_mobidb/out/train_seq.fasta',
                                     '../inpainting_mobidb/out/train_label.fasta')

valid_seqs, valid_labels = load_data('../inpainting_mobidb/out/validation_seq.fasta',
                                     '../inpainting_mobidb/out/validation_label.fasta')

test_seqs, test_labels = load_data('../inpainting_mobidb/out/test_seq.fasta',
                                     '../inpainting_mobidb/out/test_label.fasta')

# convert character array to numpy array and one hot encode the sequences
x_train = convert_ohc(train_seqs)
x_test = convert_ohc(test_seqs)
x_valid = convert_ohc(valid_seqs)

y_train = np.array([list(label) for _, label in train_labels])
y_test = np.array([list(label) for _, label in test_labels])
y_valid = np.array([list(label) for _, label in valid_labels])

#'batch' size and shape
batch_len = len(x_train)
batch_shape = x_train.shape

# make generative model, architecture used from tensorflow tutorial https://www.tensorflow.org/tutorials/generative/dcgan
def make_generative_model():
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=batch_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generative_model()
noise = tf.random.normal([1, batch_shape])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0,:,:,0], cmap='gray')



#Training Model


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

