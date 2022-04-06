"""Format clustered seqs into FASTA files."""

import os


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
        seqstring = '\n'.join([seq[i:i+80] for i in range(0, len(seq), 80)]) + '\n'
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'
        seqs_file.write(seq_header + seqstring)
        labels_file.write(label_header + labelstring)

        with open(f'out/seqs/{accession}.fasta', 'w') as file:
            file.write(seq_header + seqstring)
        with open(f'out/labels/{accession}.fasta', 'w') as file:
            file.write(label_header + labelstring)

"""
DEPENDENCIES
../cluster_seqs/cluster_seqs.py
../remove_outliers/remove_outliers.py
"""