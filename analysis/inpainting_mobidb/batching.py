import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf

# Parameters
BATCH_NUM = 10
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
sym2idx = {i: sym for i, sym in enumerate(sym_codes)}
train_seq_path = 'out/unmasked_seq_file.fasta'
train_label_path = 'out/label.fasta'


# Utility functions
def load_data(seq_path, label_path):
    """Return sequences and labels from FASTA files as arrays.

    :param seq_path: path for FASTA file of amino acid sequences
    :param label_path: path for FASTA file of labels of amino acid sequences where disordered residues are labeled are labeled as 1 and ordered residues are labeled as 0
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
    :return: target, context and weight according to seq_ohc and label
    """
    weight = np.expand_dims(label, axis=2)
    context = (np.invert(weight) + 2) * ohe

    return context, weight


def seq_to_OHE(seq):
    """Return sequence as one-hot-encoded vector

    :param seq: amino acid sequence as string
    :return: one-hot-encoded string as 2D array
    """
    ohe = np.array([sym_codes.index(sym) for sym in seq])
    ohe = tf.keras.utils.to_categorical(ohe, num_classes=len(sym_codes), dtype='int32')
    return ohe


def OHE_to_seq(ohe):
    pass


# MODELS
# Generator Model
def make_generative_model():
    """Return generative generative model.

    DCGAN architecture from "Protein Loop Modeling Using Deep Generative Adversarial Network."

    :return: model instance of generative model
    """
    # TODO: EXPERIMENT WITH THE RATIO OF FILTERS FROM LAYER TO LAYER AND WINDOW SIZES
    # Convolution
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(180, 20)))

    model.add(tf.keras.layers.Conv1D(8, 3, strides=1, padding='same', name='C1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(16, 3, strides=1, padding='same', name='C2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(32, 3, strides=1, padding='same', name='C3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(64, 3, strides=1, padding='same', name='C4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(128, 3, strides=1, padding='same', name='C5'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(256, 3, strides=1, padding='same', name='C6'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Deconvolution
    model.add(tf.keras.layers.Conv1DTranspose(128, 3, strides=1, padding='same', name='D1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(64, 3, strides=1, padding='same', name='D2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(32, 3, strides=1, padding='same', name='D3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(16, 3, strides=1, padding='same', name='D4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1DTranspose(8, 3, strides=1, padding='same', name='D5'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Last layer transforms filters to probability classes
    model.add(tf.keras.layers.Conv1DTranspose(20, 3, strides=1, padding='same', activation='softmax', name='D6'))

    return model


generator = make_generative_model()
generator.summary()


# Discriminator Model
def make_discriminator_model():
    """Return discriminative model.

    DCGAN architecture from "Protein Loop Modeling Using Deep Generative Adversarial Network."

    :return: model instance of discriminative model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(180, 20)))

    model.add(tf.keras.layers.Conv1D(25, 4, strides=2, padding='same', name='C1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(13, 4, strides=2, padding='same', name='C2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(7, 4, strides=2, padding='same', name='C3'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(4, 4, strides=2, padding='same', name='C4'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax', name='dense'))

    return model


discriminator = make_discriminator_model()
discriminator.summary()


# LOSS FUNCTIONS
cce = tf.keras.losses.CategoricalCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy()


# Generator Loss
def generator_loss(fake_output, fake_target, real_target, weight):
    """Return generator loss.

    The loss for the generator is made of reconstruction and discriminative terms
      The reconstruction loss measures how well the generator recreates the target from the context
      The discriminative loss measures are well the generator is fooling the discriminator
        Remember we want the generator to recreate the target realistically (rather than taking shortcuts that minimize the reconstruction loss)
        This means if the generator is doing well, the discriminator should think its output is real, i.e. 1
        Hence the y_true for the discriminative loss is a tensor of 1s the length of the batch size
    """
    # Make y_true for calculating loss
    true_target = tf.convert_to_tensor(real_target)
    true_output = tf.stack([tf.zeros(fake_output.shape[0]), tf.ones(fake_output.shape[0])], axis=1)

    # Apply losses
    r_loss = cce(true_target, fake_target, weight)  # Reconstruction loss
    d_loss = bce(true_output, fake_output)  # Discriminative loss
    return r_loss + d_loss


# Discriminator Loss
def discriminator_loss(real_output, fake_output):
    """Return discriminator loss."""
    # Make y_true for calculating loss
    true_real = tf.stack([tf.zeros(real_output.shape[0]), tf.ones(real_output.shape[0])], axis=1)
    true_fake = tf.stack([tf.ones(fake_output.shape[0]), tf.zeros(fake_output.shape[0])], axis=1)

    # Apply losses
    real_loss = bce(true_real, real_output)
    fake_loss = bce(true_fake, fake_output)
    return real_loss + fake_loss


# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Training Loop
def train_step(context, target, weight):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_target = target  # For clearer naming :)
        fake_target = generator(context, training=True)
        real_output = discriminator(real_target*weight, training=True)
        fake_output = discriminator(fake_target*weight, training=True)

        gen_loss = generator_loss(fake_output, fake_target, real_target, weight)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(context, target, weight, epochs):
    # Batch data
    context_batch = np.array_split(context, BATCH_NUM)
    target_batch = np.array_split(target, BATCH_NUM)
    weight_batch = np.array_split(weight, BATCH_NUM)

    # Training loop
    for epoch in range(epochs):
        print(f'EPOCH {epoch}')
        for batch, (context, target, weight) in enumerate(zip(context_batch, target_batch, weight_batch)):
            print(f'\tBATCH {batch}')
            train_step(context, target, weight)


# Load data and train
train_seq, train_label = load_data(train_seq_path, train_label_path)
train_context, train_weight = get_context_weight(train_seq, train_label)

train(train_context, train_seq, train_weight, 10)

# Examine output of trained network
real_target = train_seq
fake_target = generator(train_context).numpy()
real_output = discriminator(train_seq*train_weight).numpy()
fake_output = discriminator(fake_target*train_weight).numpy()
