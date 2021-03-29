"""Extract sequences from DisProt JSON file."""

import json
import os

# Use JSON module to load file
with open('../../../data/DisProt/DisProt.json') as file:
    data = json.load(file)['data']

# Create output directories
if not os.path.exists('out/fastas/'):
    os.makedirs('out/fastas/')  # Recursive folder creation

fields = ['disprot_id', 'acc', 'name', 'released']

# Create single FASTA file for all sequences
with open('out/disprot_2020_06.fasta', 'w') as file:
    for record in data:
        header = '|'.join([field + ':' + record[field] for field in fields]) + '\n'
        seqstring = '\n'.join(record['sequence'][i:i+80] for i in range(0, len(record['sequence']), 80)) + '\n'
        file.write('>' + header + seqstring)

# Create FASTA file for each sequence
for record in data:
    disprot_id = record['disprot_id']
    header = '|'.join([field + ':' + record[field] for field in fields]) + '\n'
    seqstring = '\n'.join(record['sequence'][i:i+80] for i in range(0, len(record['sequence']), 80)) + '\n'
    with open(f'out/fastas/{disprot_id}.fasta', 'w') as file:
        file.write('>' + header + seqstring)

"""
DEPENDENCIES
../../../data/DisProt/DisProt.json
"""