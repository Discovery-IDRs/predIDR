# software setup:
# python 3.7
# tensorflow 2.1.0
# cuda 10.1
# cudnn 7.6

# Purpose:
# This was an initial test run of the code for a cnn trained on the
# mobidb-pdb dataset.

# Architecture:
# disorder weight: x1
# layers: x2 1D conv layers with 128 filter and 20 kernel
# epoch: 50

# Significance:
# The performance of this model was initially used as the baseline to which
# the performances of other model architectures were compared to. However,
# when it was determined that increasing disorder weight seemed to prevent
# overfitting, this model was disregarded for further consideration and
# development and the performance of mobidb-pdb_cnn_3_6_1 and mobidb-pdb_cnn_6_2
# were used as baselines instead. This was because the training curves for this
# model appeared to be abnormal and seemed to indicate that possible overfitting
# was occurring.

import os

import pandas as pd
import tensorflow as tf
import src.models.utils as utils

# Parameters
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
weights = {'0': 1, '1': 1}
model_name = 'mobidb-pdb_cnn_1'
num_epochs = 50
batch_size = 32
tf.keras.utils.set_random_seed(1)

# Load data
train_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/train_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/train_labels.fasta')
validation_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/validation_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/validation_labels.fasta')
test_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/test_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/test_labels.fasta')

# Batch data
train_batches = utils.BatchGenerator(train_records, batch_size, alphabet, weights)
validation_batches = utils.BatchGenerator(validation_records, batch_size, alphabet, weights)
test_batches = utils.BatchGenerator(test_records, batch_size, alphabet, weights)

# Build model
inputs = tf.keras.layers.Input(shape=(None, 20), name='input1')
x = tf.keras.layers.Masking(mask_value=0, name='mask1')(inputs)
x = tf.keras.layers.Conv1D(128, 20, padding='same', activation='relu', name='conv1d1')(x)
x = tf.keras.layers.Conv1D(128, 20, padding='same', activation='relu', name='conv1d2')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax', name='output1')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
model.compile(loss='binary_crossentropy', optimizer='adam',
              weighted_metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

if not os.path.exists('out/'):
    os.mkdir('out/')

# Train model
model.save('out/model_untrained.h5')
history = model.fit(train_batches, epochs=num_epochs, validation_data=validation_batches)
model.save('out/model_trained.h5')

df = pd.DataFrame(history.history)
df['epoch'] = df.index + 1  # Add epochs to df
df.to_csv('out/history.tsv', sep='\t', index=False)
