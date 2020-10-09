"""Extract reference labels from DisProt JSON file."""

import json
import os


def get_labels(record):
    """Return labels for amino acid sequence where 1 indicates disorder and 0 indicates order."""
    labels = record['length'] * ['0']
    for region in record['disprot_consensus']['structural_state']:
        if region['type'] == 'D':
            start = region['start']
            end = region['end']
            labels[start-1:end] = (end-start+1) * ['1']
    return ''.join(labels)


# Use JSON module to load file
with open('../../../data/DisProt/JSON/DisProt.json') as file:
    data = json.load(file)['data']

# Create output directories
if not os.path.exists('out/fastas/'):
    os.makedirs('out/fastas/')  # Recursive folder creation

fields = ['disprot_id', 'acc', 'name', 'released']

# Create single FASTA file for all sequences
with open('out/disprot_2020_06.fasta', 'w') as file:
    for record in data:
        header = '|'.join([field + ':' + record[field] for field in fields]) + '\n'
        labels = get_labels(record)
        labelstring = '\n'.join(labels[i:i + 80] for i in range(0, len(labels), 80))
        file.write('>' + header + labelstring)

# Create FASTA file for each sequence
for record in data:
    disprot_id = record['disprot_id']
    header = '|'.join([field + ':' + record[field] for field in fields]) + '\n'
    labels = get_labels(record)
    labelstring = '\n'.join(labels[i:i + 80] for i in range(0, len(labels), 80))
    with open(f'out/fastas/{disprot_id}.fasta', 'w') as file:
        file.write('>' + header + labelstring)

"""
DEPENDENCIES
../../../data/DisProt/JSON/DisProt.json
"""