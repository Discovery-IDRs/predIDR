"""Build and run DCGAN to inpaint target disordered regions using only context."""

import os

import Bio.SeqIO as SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf


# Utility functions
def load_data(seq_path, label_path):
    """Return sequences and labels from FASTA files as arrays.

    :param seq_path: path for FASTA file of amino acid sequences
    :param label_path: path for FASTA file of labels of amino acid sequences where disordered residues are labeled as 1
        and ordered residues are labeled as 0
    :return: array of one hot encoded sequences and array of labels
    """
    ohes = []
    labels = []
    for record_seq, record_label in zip(SeqIO.parse(seq_path, 'fasta'), SeqIO.parse(label_path, 'fasta')):
        # Convert string seq to one-hot-encoded seq
        ohe = seq_to_OHE(str(record_seq.seq))
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


def seq_to_OHE(seq):
    """Return amino acid sequence as one-hot-encoded vector.

    :param seq: amino acid sequence as string
    :return: one-hot-encoded string as 2D array
    """
    ohe = np.array([sym2idx[sym] for sym in seq])
    ohe = tf.keras.utils.to_categorical(ohe, num_classes=len(alphabet), dtype='int32')
    return ohe


def OHE_to_seq(ohe):
    """Return one-hot-encoded vector as amino acid sequence.

    :param ohe: one-hot-encoded string as 2D array
    :return: amino acid sequence as list
    """
    x = np.argmax(ohe, axis=1)
    seq = []
    for idx in x:
        sym = alphabet[idx]
        seq.append(sym)
    return seq


# MODELS
def make_generative_model():
    """Return generative model.

    DCGAN architecture from "Protein Loop Modeling Using Deep Generative Adversarial Network."
    10.1109/ICTAI.2017.00166

    :return: model instance of generative model
    """
    # Convolution
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(180, len(alphabet))))

    model.add(tf.keras.layers.Conv1D(2, 2, strides=1, padding='same', name='C1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(4, 2, strides=1, padding='same', name='C2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(8, 2, strides=1, padding='same', name='C3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(16, 2, strides=1, padding='same', name='C4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Deconvolution
    model.add(tf.keras.layers.Conv1DTranspose(8, 2, strides=1, padding='same', name='D1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(4, 2, strides=1, padding='same', name='D2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(2, 2, strides=1, padding='same', name='D3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Last layer transforms filters to probability classes
    model.add(tf.keras.layers.Conv1DTranspose(len(alphabet), 3, strides=1, padding='same', activation='softmax', name='D4'))

    return model


def make_discriminator_model():
    """Return discriminative model.

    DCGAN architecture from "Protein Loop Modeling Using Deep Generative Adversarial Network."
    10.1109/ICTAI.2017.00166

    :return: model instance of discriminative model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(180, len(alphabet))))

    model.add(tf.keras.layers.Conv1D(8, 4, strides=2, padding='same', name='C1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(4, 4, strides=2, padding='same', name='C2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(2, 4, strides=2, padding='same', name='C3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax', name='dense'))

    return model


# LOSSES
def generator_loss(fake_output, fake_target, real_target, weight):
    """Return generator loss.

    The loss for the generator is made of reconstruction and discriminative terms
      The reconstruction loss measures how well the generator recreates the target from the context
      The discriminative loss measures are well the generator is fooling the discriminator
        Remember we want the generator to recreate the target realistically (rather than taking shortcuts that minimize the reconstruction loss)
        This means if the generator is doing well, the discriminator should think its output is real, i.e. 1
        Hence the y_true for the discriminative loss is a tensor of 1s the length of the batch size

    :param fake_output: discriminator evaluation of whether target from generator is real or not
    :param fake_target: generator derived target from masked sequence
    :param real_target: real data derived target from masked sequence
    :param weight: binary classification of data (1: target 0: context)
    :return: generator loss
    """
    # Make y_true for calculating loss
    true_target = tf.convert_to_tensor(real_target)
    true_output = tf.stack([tf.zeros(fake_output.shape[0]), tf.ones(fake_output.shape[0])], axis=1)

    # Apply losses
    r_loss = cce(true_target, fake_target, weight)  # Reconstruction loss
    d_loss = bce(true_output, fake_output)  # Discriminative loss
    return r_loss + d_loss


def discriminator_loss(real_output, fake_output):
    """Return discriminator loss.

    The loss how well the discriminator can differentiate between the real output from the data and the fake output from
    the generator.

    :param real_output: discriminator evaluation of whether target from real data is real or not
    :param fake_output: discriminator evaluation of whether target from generator is real or not
    :return: discriminator loss
    """
    # Make y_true for calculating loss
    true_real = tf.stack([tf.zeros(real_output.shape[0]), tf.ones(real_output.shape[0])], axis=1)
    true_fake = tf.stack([tf.ones(fake_output.shape[0]), tf.zeros(fake_output.shape[0])], axis=1)

    # Apply losses
    real_loss = bce(true_real, real_output)
    fake_loss = bce(true_fake, fake_output)
    return real_loss + fake_loss


# TRAINING
def train_step(context, target, weight, batch_idx, player_interval):
    """Run one step of training."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_target = target  # For clearer naming :)
        fake_target = generator(context, training=True)
        real_output = discriminator(real_target*weight, training=True)
        fake_output = discriminator(fake_target*weight, training=True)

        if batch_idx % player_interval == 0:
            gen_loss = generator_loss(fake_output, fake_target, real_target, weight)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        else:
            disc_loss = discriminator_loss(real_output, fake_output)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(train_context, train_target, train_weight, valid_context, valid_target, valid_weight,
          epochs, batch_size, seed=None):
    """Run training loop.

    :param train_context: sequence around disordered sequence of interest
    :param train_target: disordered sequence of interest
    :param train_weight: binary classification of data (1: target 0: context)
    :param valid_context: sequence around disordered sequence of interest
    :param valid_target: disordered sequence of interest
    :param valid_weight: binary classification of data (1: target 0: context)
    :param epochs: number of training loops
    :param batch_size: number of examples in batch
    :param seed: value of random seed
    :return: dataframe of losses and accuracy in each epoch of training
    """
    rng = np.random.default_rng(seed)
    batch_num = train_context.shape[0] // batch_size
    indices = np.arange(train_context.shape[0])

    # Training loop
    records = []
    for epoch in range(epochs):
        print(f'EPOCH {epoch} / {epochs}')

        # Update parameters for each batch
        indices_shuffle = rng.permutation(indices)
        for batch_idx in range(batch_num):
            print(f'\r\tBATCH {batch_idx} / {batch_num-1}', end='')  # Use carriage return to move cursor to beginning before printing

            # Get batch
            indices_batch = indices_shuffle[batch_idx*batch_size:(batch_idx+1)*batch_size]
            context_batch = train_context[indices_batch]
            target_batch = train_target[indices_batch]
            weight_batch = train_weight[indices_batch]

            # Run backpropagation on batch
            train_step(context_batch, target_batch, weight_batch, batch_idx, train_interval)

        # Calculate metrics at epoch end
        data_sets = [(train_context, train_target, train_weight, 'train'),
                     (valid_context, valid_target, valid_weight, 'valid')]
        record = {'epoch': epoch}
        for context, target, weight, label in data_sets:
            # Get targets and outputs
            real_target = target
            fake_target = generator(context)
            real_output = discriminator(real_target*weight).numpy()
            fake_output = discriminator(fake_target*weight).numpy()

            # Calculate metrics
            equality_target = np.argmax(real_target*weight, axis=2) == np.argmax(fake_target*weight, axis=2)
            sum_context = np.sum(np.invert(weight) + 2)
            target_length = np.sum(weight)
            accuracy = (np.sum(equality_target) - sum_context) / target_length
            record[label + ' accuracy'] = accuracy
            record[label + ' generator loss'] = generator_loss(fake_output, fake_target, real_target, weight).numpy()
            record[label + ' discriminator loss'] = discriminator_loss(real_output, fake_output).numpy()
        records.append(record)

        # Report metrics
        print()  # Add newline after batch counter
        print('\taccuracy loss:', record['train accuracy'])
        print('\tgenerator loss:', record['train generator loss'])
        print('\tdiscriminator loss:', record['train discriminator loss'])

    return records


# PARAMETERS
batch_size = 30
train_interval = 10
epoch_num = 300
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
sym2idx = {sym: idx for idx, sym in enumerate(alphabet)}
idx2sym = {idx: sym for idx, sym in enumerate(alphabet)}

# MODEL
generator = make_generative_model()
generator.summary()
print()  # Newline after model summary

discriminator = make_discriminator_model()
discriminator.summary()
print()  # Newline after model summary

# LOSS FUNCTIONS
cce = tf.keras.losses.CategoricalCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy()

# OPTIMIZER
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# LOAD DATA AND TRAIN
train_seq_path = '../split_data/out/train_seqs.fasta'
train_label_path = '../split_data/out/train_labels.fasta'
valid_seq_path = '../split_data/out/validation_seqs.fasta'
valid_label_path = '../split_data/out/validation_labels.fasta'

train_seq, train_label = load_data(train_seq_path, train_label_path)
valid_seq, valid_label = load_data(valid_seq_path, valid_label_path)

train_context, train_weight = get_context_weight(train_seq, train_label)
valid_context, valid_weight = get_context_weight(valid_seq, valid_label)
history = train(train_context, train_seq, train_weight, valid_context, valid_seq, valid_weight,
                epoch_num, batch_size, seed=1)

# SAVE DATA
if not os.path.exists("out/"):
    os.mkdir("out/")

pd.DataFrame(history).to_csv('out/metrics.tsv', index=False, sep='\t')
generator.save('out/generator_model.h5')
discriminator.save('out/discriminator_model.h5')

