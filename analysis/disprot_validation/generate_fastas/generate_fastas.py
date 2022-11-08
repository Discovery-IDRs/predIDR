"""Generate FASTAs from raw DisProt files."""

import json
import os


def get_label(record):
    """Return label for amino acid sequence where 1 indicates disorder and 0 indicates order."""
    label = record['length'] * ['0']
    for region in record['disprot_consensus']['structural_state']:
        if region['type'] == 'D':
            start = region['start']
            end = region['end']
            label[start-1:end] = (end-start+1) * ['1']
    return ''.join(label)


# Use JSON module to load file
with open('../../../data/DisProt/2020_12.json') as file:
    data = json.load(file)['data']

# Create output directory
if not os.path.exists('out/'):
    os.mkdir('out/')

fields = ['disprot_id', 'acc', 'name', 'released']

# Create single FASTA file for all sequences
with open('out/disprot_seqs.fasta', 'w') as seqs_file, open('out/disprot_labels.fasta', 'w') as labels_file:
    for record in data:
        header = '|'.join([field + ':' + record[field] for field in fields])
        seq = record['sequence']
        label = get_label(record)
        seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)])
        seqs_file.write(f'>{header}\n{seqstring}\n')
        labels_file.write(f'>{header}\n{labelstring}\n')

"""
DEPENDENCIES
../../../data/DisProt/2020_12.json
"""