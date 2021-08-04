#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install tensorflow


# In[1]:


import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt


# In[2]:


# Parameters
sym_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# In[3]:


BATCH_SIZE = 10 


# In[4]:


# Helper function for func load_data 
def convert_ohc(seq):
    """
    One hot encodes given amino acid sequence string.
    
    :param seq: string of amino acid sequence 
    :return: 2D array of one hot encoded string 
    
    """
    seq_idx = [sym_codes.index(sym) for sym in seq]
    x = np.array(seq_idx)
    x = keras.utils.to_categorical(x, num_classes=len(sym_codes), dtype='int32')
    
    return x


# In[5]:


def load_data(seqs_path, label_path):
    """
    Loads sequences and lables from fasta files. 
    
    :param seq_path: path for fasta file of amino acid sequences 
    :param label_path: path fasta file of labels of amino acid sequences where disordered residues are labeled are labeled as 1 and ordered residues are labeled as 0
    :return: array all one hot encoded sequences and array of all labels from 
    """
    seq_ohc_lst = []
    label_lst = []
    
    for record_seq, record_label in zip(SeqIO.parse(seqs_path, 'fasta'), SeqIO.parse(label_path, 'fasta')):
        
        # one hot encode each record_seq 
        seq = str(record_seq.seq)
        seq_ohc = convert_ohc(seq)
        seq_ohc_lst.append(seq_ohc)
        
        # expand the dimension of record_label for broadcasting
        label = [int(sym) for sym in record_label]
        label_lst.append(label)
        
    return np.array(seq_ohc_lst), np.array(label_lst)


# In[6]:


def get_weight_target_context(seq_ohc, label):
    """
    Gets the target, context, and weight from one hot encoded sequences and labels. 
    
    :param seq_ohc: one hot ended 2D arrays of sequences 
    :param label: array of labels corresponding to seq_ohc 
    :return: target, context and weight according to seq_ohc and label 
    
    """

    weight = np.expand_dims(label, axis = 2)

    # get the target from the record 
    target = weight*seq_ohc
        
    # get the context from the record (inverted the weight)
    context = (np.invert(weight) + 2)*seq_ohc
    
    return weight, target, context


# # Generator Model

# In[7]:


# make generative model
def make_generative_model():
    """
    Makes generative generative model for DCGAN based off of architecture from "Protein Loop Modeling Using 
    Deep Generative Adversarial Network" paper. 
    
    :return: model instance of generative model 
    
    """
    
    # convolution 
    model = tensorflow.keras.Sequential()
    model.add(keras.Input(shape=((180, 20))))
    
    model.add(keras.layers.Conv1D(8, 3, strides = 1, padding='same', name='first'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(16, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(32, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv1D(64, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(128, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(256, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    # deconvolution 
    model.add(keras.layers.Conv1DTranspose(128, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1DTranspose(64, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1DTranspose(32, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1DTranspose(16, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1DTranspose(8, 3, strides = 1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    #FIXEME: PLAY AROUND WITH THE RATIO OF FILTER 
    
    model.add(keras.layers.Conv1DTranspose(20, 3, strides = 1, padding='same', activation = 'softmax'))

    return model


# In[8]:


generator = make_generative_model()
generator.summary()


# # Discriminator Model

# In[9]:


# make discrimator model
def make_discriminator_model():
    """
    Makes adverserial/discriminative model for DCGAN based off of architecture from "Protein Loop Modeling Using 
    Deep Generative Adversarial Network" paper. 
    
    :return: model instance of discriminative model 
    
    """
    model = tensorflow.keras.Sequential()
    model.add(keras.Input(shape=((180, 20))))
    
    model.add(keras.layers.Conv1D(25, 4, strides = 2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(13, 4, strides = 2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(7, 4, strides = 2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Conv1D(4, 4, strides = 2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation = 'softmax'))
    
    return model


# In[10]:


discriminator = make_discriminator_model()
discriminator.summary()


# # Loss Function

# ## Generator Loss

# In[11]:


cross_entropy = tensorflow.keras.losses.CategoricalCrossentropy()


# In[12]:


def generator_loss(fake_output, generated_target, target):
    # label: data_step[0], target: data_step[1], context: data_step[2]
    return cross_entropy(tensorflow.ones_like(fake_output), fake_output) + cross_entropy(generated_target, target)


# ## Discriminator Loss

# In[13]:


def discriminator_loss(real_output, fake_output, weight):
    real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output, weight)
    fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output, weight)
    total_loss = real_loss + fake_loss
    return total_loss


# # Optimizer

# In[14]:


generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)


# # Training Loop

# In[15]:


def train_step(context, target, weight):

    with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
        generated_target = generator(context, training=True)*weight
        
        real_output = discriminator(target, training=True)
        fake_output = discriminator(generated_target, training=True)
        
        gen_loss = generator_loss(fake_output, generated_target, target)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    #backpropogration 

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[17]:


def train(context, target, weight, epochs):
    
    # batch data 
    context_batch = np.array_split(context, BATCH_SIZE)
    target_batch = np.array_split(target, BATCH_SIZE)
    weight_batch = np.array_split(weight, BATCH_SIZE)
    
    for epoch in range(epochs):

        for context, target, weight in zip(context_batch, target_batch, weight_batch):
            
            train_step(context, target, weight)


# In[18]:


# Load data
train_seq, train_label = load_data('../inpainting_mobidb/out/train_seq.fasta', '../inpainting_mobidb/out/train_label.fasta')
train_weight, train_target, train_context = get_weight_target_context(train_seq, train_label)


#test_data = load_data('../../inpainting_mobidb/out/test_seq.fasta','../../inpainting_mobidb/out/test_label.fasta')

#valid_data = load_data('../../inpainting_mobidb/out/validation_seq.fasta','../../inpainting_mobidb/out/validation_label.fasta')


# In[19]:


contxt = train_context[:5]
trgt = train_target[:5]
wght = train_weight[:5]


# In[25]:


contxt.shape


# In[26]:


trgt.shape


# In[27]:


wght.shape


# In[20]:


generator(contxt, training=True)*wght


# In[21]:


g = make_generative_model()
g.predict(contxt)*wght


# In[24]:


train(train_context, train_target, train_weight, 1)

