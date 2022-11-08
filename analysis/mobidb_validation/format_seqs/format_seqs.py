"""Format clustered seqs into FASTA files."""

import os

from src.utils import read_fasta

# Set file paths
seqs_path = '../remove_outliers/out/mobidb_seqs.fasta'
labels_path = '../remove_outliers/out/mobidb_labels.fasta'
cluster_path = '../cluster_seqs/out/mobidb.clstr'

# Load FASTA files
seqs_fasta = read_fasta(seqs_path)
labels_fasta = read_fasta(labels_path)

records = {}  # Dictionary of header, seq, header, label keyed by accession
for header, seq in seqs_fasta:
    accession = header.split('|')[0][1:]  # Trim >
    records[accession] = [header, seq]
for header, label in labels_fasta:
    accession = header.split('|')[0][1:]  # Trim >
    records[accession].extend([header, label])

# Read raw cluster output and extract representative protein codes
reps = []
with open(cluster_path) as file:
    for line in file:
        if '*' in line:
            accession = line.split('|')[0].split('>')[1]
            reps.append(accession)

# Make output directories
if not os.path.exists('out/'):
    os.mkdir('out/')
if not os.path.exists('out/seqs/'):
    os.mkdir('out/seqs/')
if not os.path.exists('out/labels/'):
    os.mkdir('out/labels/')

# Write output
with open('out/mobidb_seqs.fasta', 'w') as seqs_file, open('out/mobidb_labels.fasta', 'w') as labels_file:
    for accession in sorted(reps):
        seq_header, seq, label_header, label = records[accession]
        seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)])
        seqs_file.write(f'{seq_header}\n{seqstring}\n')
        labels_file.write(f'{label_header}\n{labelstring}\n')

        with open(f'out/seqs/{accession}.fasta', 'w') as file:
            file.write(f'{seq_header}\n{seqstring}\n')
        with open(f'out/labels/{accession}.fasta', 'w') as file:
            file.write(f'{label_header}\n{labelstring}\n')

"""
DEPENDENCIES
../cluster_seqs/cluster_seqs.py
../remove_outliers/remove_outliers.py
"""