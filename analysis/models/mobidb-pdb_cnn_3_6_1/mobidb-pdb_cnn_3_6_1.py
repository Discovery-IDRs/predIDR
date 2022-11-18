# software setup:
# python 3.7
# tensorflow 2.1.0
# cuda 10.1
# cudnn 7.6

# Purpose:
# Examine the effect increasing the disorder weight to 75 has on performance
# and overfitting.

# Architecture:
# disorder weight: x75
# layers: x2 1D conv layers with 128 filter and 20 kernel
# epoch: 50

# Significance:
# The 3 series of models were an attempt to try using an increase in disorder weight
# to counteract the overfitting issue caused by the imbalance of classes. They appear
# to demonstrate that, as disorder weight increases, accuracy, MCC, specificity,
# precision, and f1 scores decrease while sensitivity scores increase. This particular
# model does appear to follow those trends. Additionally, the 3 series seems to demonstrate
# that the training curves look less and less abnormal as the disorder weight increases
# seeming to indicate that less and less overfitting is occurring. This model demonstrated
# normal looking training curves (note the rapid initial growth of accuracy and specificity
# that eventually levels off, something which would be expected from a model that is
# progressively learning to make better predictions; this is in contrast to the overfitting
# models which demonstrate abnormally good accuracy and specificity right from the start
# of training). This model was ultimately used along with mobidb-pdb_cnn_6_2 (which was itself
# based on this model) as the templates upon which further models were developed. Naturally,
# this meant that the performance of this model also served alongside the performance of
# mobidb-pdb_cnn_6_2 as a baseline to which the performances of other model architectures
# were compared to. The reason for choosing this model and mobidb-pdb_cnn_6_2 to serve as
# templates and baselines was because x75 disorder weight was the smallest weight tested
# which seemed to properly counteract overfitting. While other models which implemented even
# larger disorder weights did not appear to demonstrate any abnormal behavior and did also
# counteract overfitting, there was a desire to limit the size of the disorder weight as
# much as possible in order to minimize manipulation of the dataset. This along with the
# fact that these other models did not appear to demonstrate any significant increases in
# performance meant that they were ultimately passed over for further consideration and
# development in favor of this model and mobidb-pdb_cnn_6_2. Note also that close examination
# of the training curve graphs for models 3.6 through 3.9 seems to show that the increases in
# disorder weight for these models from x50 to x500 appears to simply cause the training curves
# to shift increasingly to the right, seeming to indicate that the increased disorder weights
# simply delay the learning of the models but do not actually change their final performances.

import os

import pandas as pd
import tensorflow as tf
import src.models.utils as utils

# Parameters
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
weights = {'0': 1, '1': 75}
model_name = 'mobidb-pdb_cnn_3_6_1'
num_epochs = 50
batch_size = 32
tf.keras.utils.set_random_seed(1)

# Load data
train_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/train_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/train_labels.fasta')
validation_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/validation_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/validation_labels.fasta')

# Batch data
train_batches = utils.BatchGenerator(train_records, batch_size, alphabet, weights)
validation_batches = utils.BatchGenerator(validation_records, batch_size, alphabet, weights)

# Build model
inputs = tf.keras.layers.Input(shape=(None, 20), name='input1')
x = tf.keras.layers.Conv1D(128, 20, padding='same', activation='relu', name='conv1d1')(inputs)
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
