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
weights = {'0': 1, '1': 2, '-': 3}
batch_size = 32
model_data = [('../../models/mobidb-pdb_cnn_6_1/out_model/mobidb-pdb_cnn_6_1.h5', 'mobidb-pdb_cnn_6_1', 'conv1d2')]

# Load and batch data
records = utils.load_data('../../mobidb-pdb_validation/split_data/out/all_seqs.fasta', '../../mobidb-pdb_validation/split_data/out/all_labels.fasta')
batches = utils.BatchGenerator(records, batch_size, alphabet, weights)

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
    known_features = []
    training_weights = []
    for input, _, training_weight in batches:  # Predict method was acting strange, so extract individual batches
        features = feature_extractor(input).numpy()

        # Combine features across examples into single axis and remove padding
        example_num, position_num, feature_num = features.shape

        features = np.swapaxes(features, 0, 1)  # Put examples on first so reshaped by sequence first
        features = features.reshape((example_num * position_num, feature_num), order='F')  # Fortran style indexing: First index changes fastest which preserves feature layout
        training_weight = np.swapaxes(training_weight, 0, 1)
        training_weight = training_weight.reshape((example_num * position_num), order='F')

        features = features[training_weight != 0, :]  # Drop padding and unknown labels
        learned_features.append(features.transpose())

        # Calculate feature maps using known features
        known_features_example = []
        # for example in input:
        #   convert one-hot-encoded to string
        #   use string to calculate known features
        #   for feature_function in feature_functions
        #       feature = feature_function
        #       drop unknown positions
        #       append to known_features_example
        known_features.append(known_features_example)

        training_weight = training_weight[training_weight != 0]
        training_weights.append(training_weight)

    learned_features = np.concatenate(learned_features, axis=1)
    #known_features = np.concatenate(known_features, axis=1)
    training_weights = np.concatenate(training_weights)

    disorder_slice = training_weights == weights['1']
    order_slice = training_weights == weights['0']
    label_slice = disorder_slice | order_slice

    corr = np.corrcoef(learned_features)
    plt.imshow(corr, vmin=-1, vmax=1)  # Fix colormap to full range of allowed values for correlations
    plt.title('Learned feature correlations: all positions')
    plt.xlabel('Learned feature 1')
    plt.ylabel('Learned feature 2')
    plt.colorbar()
    plt.savefig(f'{out_path}/heatmap_corr_learned_features_all.png')
    plt.close()

    corr = np.corrcoef(learned_features[:, order_slice])
    plt.imshow(corr, vmin=-1, vmax=1)  # Fix colormap to full range of allowed values for correlations
    plt.title('Learned feature correlations: order positions')
    plt.xlabel('Learned feature 1')
    plt.ylabel('Learned feature 2')
    plt.colorbar()
    plt.savefig(f'{out_path}/heatmap_corr_learned_features_order.png')
    plt.close()

    corr = np.corrcoef(learned_features[:, disorder_slice])
    plt.imshow(corr, vmin=-1, vmax=1)  # Fix colormap to full range of allowed values for correlations
    plt.title('Learned feature correlations: disorder positions')
    plt.xlabel('Learned feature 1')
    plt.ylabel('Learned feature 2')
    plt.colorbar()
    plt.savefig(f'{out_path}/heatmap_corr_learned_features_disorder.png')
    plt.close()

    corr = np.corrcoef(learned_features[:, label_slice])
    plt.imshow(corr, vmin=-1, vmax=1)  # Fix colormap to full range of allowed values for correlations
    plt.title('Learned feature correlations: labeled positions')
    plt.xlabel('Learned feature 1')
    plt.ylabel('Learned feature 2')
    plt.colorbar()
    plt.savefig(f'{out_path}/heatmap_corr_learned_features_label.png')
    plt.close()
