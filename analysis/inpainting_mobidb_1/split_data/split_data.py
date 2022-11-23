"""Split extracted disordered regions into train, validation, and test sets"""

import math
import os
import random

from Bio import SeqIO

# Ratio for splitting into validation, test and train
train_ratio = 0.80
test_ratio = 0.10
random.seed(7)

# Load data
fasta_seqs = SeqIO.parse("../extract_data/out/seqs.fasta", 'fasta')
fasta_labels = SeqIO.parse("../extract_data/out/labels.fasta", 'fasta')

# Create dictionary with key-value pair, "accession" : ["amino_acid_sequence", "disorder_labels"]
input_records = {}

# Load amino acid sequences into dictionary
for record in fasta_seqs:
    accession = record.id
    input_records[accession] = str(record.seq)

# Edit dictionary to include amino labels and descriptions
for record in fasta_labels:
    accession = record.id
    input_records[accession] = [input_records[accession], str(record.seq), record.description]

# Data shuffling
accessions = list(input_records)
random.shuffle(accessions)

# Extract by index
train_length = math.ceil(train_ratio*len(accessions))
test_length = math.ceil(test_ratio*len(accessions))

train = accessions[:train_length]
test = accessions[train_length:train_length+test_length]
validation = accessions[train_length+test_length:]  # Validation gets remainder if split is not even

# Create out directory to put FASTA files in
if not os.path.exists("out/"):
    os.mkdir("out/")

data_sets = [('train', train), ('validation', validation), ('test', test)]
for data_label, data_set in data_sets:
    with open(f"out/{data_label}_seqs.fasta", "w") as seq_file, open(f"out/{data_label}_labels.fasta", "w") as labels_file:
        for accession in data_set:
            record = input_records[accession]
            seq, labels, description = record

            seqstring = "\n".join([seq[i:i + 80] for i in range(0, len(seq), 80)])
            seq_file.write(f">{description}\n{seqstring}\n")

            seqstring = "\n".join([labels[i:i + 80] for i in range(0, len(labels), 80)])
            labels_file.write(f">{description}\n{seqstring}\n")
