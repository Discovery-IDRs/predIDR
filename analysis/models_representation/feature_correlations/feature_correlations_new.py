"""Calculate correlations between learned and known features."""

import os

# Load models as dictionary of mapping of model name to model
#
# Calculate learned features
# (See old script)
# Load (seq, labels) records (utils.models.load_data)
# Batching via batch generator
# learned features = []
# for stuff in batches:
#     Get learned features in batch (REMEMBER SHUFFLE=FALSE)
#     Append to learned features
# REMEMBER THE ORDER STAYS THE SAME AS IN ORIGINAL RECORDS
#
#
# fasta = read_fasta(path)
# Pull out accessions here
#
# Construct new record objects that contain known and learned features
# records = []
# for accession, (seq, labels), learned_feature in zip(accessions, records, learned_features):
#     Load known features using accession
#     Put everything into a dictionary and append to records
