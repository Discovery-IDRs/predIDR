"""Extract reference labels from DisProt JSON file."""

import json
import os


def get_labels(records):
    """Return labels for amino acid sequence where 1 indicates disorder and 0 indicates order."""
    pass


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
        seqstring = ''
        file.write('>' + header + seqstring)

# Create FASTA file for each sequence
for record in data:
    disprot_id = record['disprot_id']
    header = '|'.join([field + ':' + record[field] for field in fields]) + '\n'
    seqstring = ''
    with open(f'out/fastas/{disprot_id}.fasta', 'w') as file:
        file.write('>' + header + seqstring)

"""
DEPENDENCIES
../../../data/DisProt/JSON/DisProt.json
"""