"""Split data in train, validation, and test subsets."""

import math
import os
import random


def load_fasta(path):
    fasta = []
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                header = line
                line = file.readline()

            seqlines = []
            while line and not line.startswith('>'):
                seqlines.append(line.rstrip())
                line = file.readline()
            seq = ''.join(seqlines)
            fasta.append((header, seq))
    return fasta


# Set file paths
seqs_path = '../remove_outliers/out/mobidb_seqs.fasta'
labels_path = '../remove_outliers/out/mobidb_labels.fasta'
cluster_path = '../cluster_seqs/out/mobidb.clstr'

# Load FASTA files
seqs_fasta = load_fasta(seqs_path)
labels_fasta = load_fasta(labels_path)

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

# Data shuffling and splitting
# Set random seed for repeatability and shuffle the data after sorting in-place
random.seed(1)
reps.sort()
random.shuffle(reps)

# Extract by index
train_length = math.ceil(0.8 * len(reps))
test_length = math.ceil(0.1 * len(reps))

train = reps[:train_length]
test = reps[train_length:train_length + test_length]
validation = reps[train_length + test_length:]  # Validation gets remainder if split is not even

if not os.path.exists('out/'):
    os.mkdir('out/')

# Compile all sequences into FASTA files
for name, subset in zip(['train', 'test', 'validation'], [train, test, validation]):
    with open(f'out/{name}_seqs.fasta', 'w') as seqs_file, open(f'out/{name}_labels.fasta', 'w') as labels_file:
        for accession in subset:
            seq_header, seq, label_header, label = records[accession]
            seqstring = '\n'.join([seq[i:i + 80] for i in range(0, len(seq), 80)]) + '\n'
            labelstring = '\n'.join([label[i:i + 80] for i in range(0, len(label), 80)]) + '\n'
            seqs_file.write(seq_header + seqstring)
            labels_file.write(label_header + labelstring)

# Compute statistics for each subset of the split
for name, subset in zip(['train', 'test', 'validation'], [train, test, validation]):
    residue_num = 0
    order_num = 0
    disorder_num = 0
    for accession in subset:
        label = records[accession][3]
        residue_num += len(label)
        order_num += label.count('0')
        disorder_num += label.count('1')

    print(name.upper())
    print('Number of proteins:', len(subset))
    print('Number of residues:', residue_num)
    print('Number of ordered residues:', order_num)
    print('Fraction of ordered residues:', order_num / residue_num)
    print('Number of disordered residues:', disorder_num)
    print('Fraction of disordered residues:', disorder_num / residue_num)
    print()
print('Subsets sum to total protein number:', sum([len(subset) for subset in [train, test, validation]]) == len(reps))

"""
OUTPUT
TRAIN
Number of proteins: 17399
Number of residues: 8234588
Number of ordered residues: 7727846
Fraction of ordered residues: 0.9384617663931699
Number of disordered residues: 506742
Fraction of disordered residues: 0.06153823360683012

TEST
Number of proteins: 2175
Number of residues: 1065462
Number of ordered residues: 1005395
Fraction of ordered residues: 0.9436235173098618
Number of disordered residues: 60067
Fraction of disordered residues: 0.056376482690138174

VALIDATION
Number of proteins: 2174
Number of residues: 1048302
Number of ordered residues: 975730
Fraction of ordered residues: 0.9307718577280212
Number of disordered residues: 72572
Fraction of disordered residues: 0.06922814227197888

Subsets sum to total protein number: True

DEPENDENCIES
../cluster_seqs/cluster_seqs.py
../remove_outliers/remove_outliers.py
"""