"""Generate maps of features for each sequence in data set."""

import os
import src.utils as utils

# Write feature functions
# Make dict of functions to iterate over
#
# Load sequence FASTA files
#
# records = []
# for each sequence
#     feature_maps = {}
#     for each feature
#          calculate feature_map
#          feature_maps[feature_label] = feature_map
# Want to know the accession for the sequence to make it the file name
# Want to be able to pull out feature maps in right order
# Store this information in record and feature_maps
# record = (accession, feature_map) ???
#
# Write outputs to file
# for each sequence
#     create a TSV file with columns as features and rows as positions in sequence
#     tab character: '\t'
