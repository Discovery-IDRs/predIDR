"""Split data in train, validation, and test subsets."""

import math
import os
import random

from src.utils import read_fasta

# Set file paths
seqs_path = '../remove_outliers/out/mobidb-pdb_seqs.fasta'
labels_path = '../remove_outliers/out/mobidb-pdb_labels.fasta'
cluster_path = '../cluster_seqs/out/mobidb-pdb.clstr'

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
for name, subset in zip(['all', 'train', 'test', 'validation'], [reps, train, test, validation]):
    with open(f'out/{name}_seqs.fasta', 'w') as seqs_file, open(f'out/{name}_labels.fasta', 'w') as labels_file:
        for accession in subset:
            seq_header, seq, label_header, label = records[accession]
            seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)])
            labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)])
            seqs_file.write(f'{seq_header}\n{seqstring}\n')
            labels_file.write(f'{label_header}\n{labelstring}\n')

# Compute statistics for each subset of the split
for name, subset in zip(['all', 'train', 'test', 'validation'], [reps, train, test, validation]):
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
ALL
Number of proteins: 21560
Number of residues: 10242811
Number of ordered residues: 5942300
Fraction of ordered residues: 0.58014347819168
Number of disordered residues: 601545
Fraction of disordered residues: 0.05872850724278716

TRAIN
Number of proteins: 17248
Number of residues: 8152053
Number of ordered residues: 4725712
Fraction of ordered residues: 0.5796959367168001
Number of disordered residues: 478645
Fraction of disordered residues: 0.05871465752246704

TEST
Number of proteins: 2156
Number of residues: 1069167
Number of ordered residues: 616320
Fraction of ordered residues: 0.5764487680596202
Number of disordered residues: 66602
Fraction of disordered residues: 0.06229335548141684

VALIDATION
Number of proteins: 2156
Number of residues: 1021591
Number of ordered residues: 600268
Fraction of ordered residues: 0.5875815272452478
Number of disordered residues: 56298
Fraction of disordered residues: 0.05510815972341181

Subsets sum to total protein number: True

DEPENDENCIES
../cluster_seqs/cluster_seqs.py
../remove_outliers/remove_outliers.py
"""
