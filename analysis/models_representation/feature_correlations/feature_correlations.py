"""Calculate correlations between learned and known features."""

import os

import matplotlib.pyplot as plt
import numpy as np
import src.models.utils as utils
import tensorflow as tf


class MaskedConv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.supports_masking = True


alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
weights = {'0': 1, '1': 1, '-': 0}
batch_size = 32
model_data = [('../../models/mobidb-pdb_cnn_6_1/out_model/mobidb-pdb_cnn_6_1.h5', 'mobidb-pdb_cnn_6_1', 'conv1d2')]

# Load data
train_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/train_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/train_labels.fasta')
validation_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/validation_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/validation_labels.fasta')
test_records = utils.load_data('../../mobidb-pdb_validation/split_data/out/test_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/test_labels.fasta')

# Batch data
train_batches = utils.BatchGenerator(train_records, batch_size, alphabet, weights)
validation_batches = utils.BatchGenerator(validation_records, batch_size, alphabet, weights)
test_batches = utils.BatchGenerator(test_records, batch_size, alphabet, weights)

if not os.path.exists('out/'):
    os.mkdir('out/')

for model_path, model_name, layer_name in model_data:
    out_path = f'out/{model_name}/'
    if not os.path.exists(f'out/{model_name}/'):
        os.mkdir(out_path)

    # Load model and extract last layer
    model = tf.keras.models.load_model(model_path, custom_objects={"MaskedConv1D": MaskedConv1D})
    layer = model.get_layer(layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # Calculate features and plot
    learned_features = []
    for input, _, weight in train_batches:  # Predict method was acting strange, so extract individual batches
        features = feature_extractor(input).numpy()

        # Combine features across examples into single axis and remove padding
        example_num, position_num, feature_num = features.shape
        features = features.reshape((example_num * position_num, feature_num), order='F')  # Fortran style indexing: First index changes fastest which preserves feature layout
        weight = weight.reshape((example_num * position_num))
        features = features[weight == 1, :]  # Drop padding
        learned_features.append(features)

        # Calculate feature maps using known features
    learned_features = np.concatenate(learned_features)
    corr = np.corrcoef(learned_features, rowvar=False)  # Columns are features which is opposite of default

    plt.imshow(corr, vmin=-1, vmax=1)  # Fix colormap to full range of allowed values for correlations
    plt.title('Learned feature correlations')
    plt.xlabel('Learned feature 1')
    plt.ylabel('Learned feature 2')
    plt.colorbar()
    plt.savefig(f'{out_path}/heatmap_corr_learned_features.png')
    plt.close()

# calculate correlations between all pairs of features in the two sets
# visualize it somehow