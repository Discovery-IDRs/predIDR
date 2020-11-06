"""Calculate features for all sequences in DisProt."""

import os
import features

# Calculate features
disprot_features = []
for file_label in os.listdir('../../disprot_validation/extract_seqs/out/fastas/'):
    with open(f'../../disprot_validation/extract_seqs/out/fastas/{file_label}') as file:
        # Parse FASTA and get sequence
        disprot_id = file_label[:-4]
        seq = ''

        # Apply feature functions to sequence
        seq_features = features.get_features(seq, features.features, 15)

        # Store features in list labeled by DisProt ID and residue ID
        for sym_id, sym_features in enumerate(seq_features):
            disprot_features.append((disprot_id, str(sym_id), sym_features))

# Write features to file
if not os.path.exists('out/'):
    os.mkdir('out/')

with open('out/disprot_features.tsv', 'w') as file:
    feature_labels = list(features.features)
    file.write('\t'.join(['disprot_id', 'sym_id'] + feature_labels) + '\n')
    for disprot_id, sym_id, sym_features in disprot_features:
        values = [str(sym_features[feature_label]) for feature_label in feature_labels]
        file.write('\t'.join([disprot_id, sym_id] + values) + '\n')

"""
DEPENDENCIES
../../disprot_validation/extract_seqs/extract_seq.py
    ../../disprot_validation/extract_seqs/out/fastas/*.fasta
"""