"""Extract disordered regions within length limits from MobiDB data."""

import os

import scipy.ndimage as ndimage
import numpy as np
from Bio import SeqIO

# Create variables determined from inpainting_exploration.ipynb of upper and lower limit of length of disordered regions
LOWER_LIMIT = 30
UPPER_LIMIT = 90
TOTAL_LENGTH = 180
RANDOM_MASKING_PERCENTAGE = 0.9

# Set path of previously created fasta files of amino acid sequence and labels
fasta_seqs = SeqIO.parse("../../mobidb_validation/format_seqs/out/mobidb_seqs.fasta", "fasta")
fasta_labels = SeqIO.parse("../../mobidb_validation/format_seqs/out/mobidb_labels.fasta", "fasta")

# Create dictionary with key-value pair, "accession" : ["amino_acid_sequence", "disorder_labels"]
input_records = {}

# Load amino acid sequences into dictionary
for record in fasta_seqs:
    accession = record.id.split("|")[0]
    input_records[accession] = str(record.seq)

# Edit dictionary to include amino labels and descriptions
for record in fasta_labels:
    accession = record.id.split("|")[0]
    input_records[accession] = [input_records[accession], str(record.seq)]

# Iterate through all proteins
output_records = []
for accession, record in input_records.items():
    # Find the disordered regions of the protein
    seq, labels = record
    labels = [sym == "1" for sym in labels]

    slices = ndimage.find_objects(ndimage.label(labels)[0])
    for s, in slices:  # Unpack 1-tuple of slice indices
        length = s.stop - s.start



        if LOWER_LIMIT <= length <= UPPER_LIMIT:  # Check if disordered region is of desired length
            # Calculate the length for the context of the disordered region
            context_length = (TOTAL_LENGTH - length) // 2
            remainder_length = (TOTAL_LENGTH - length) % 2
            start = s.start - context_length
            stop = s.stop + context_length + remainder_length

            # Check if there is enough context on both sides of the sequence
            if (start >= 0) and (stop <= len(seq)):
                # Make output seq and labels
                output_seq = seq[start:stop]

                # within disordered region (length of disordered region = length) do random masking
                random_index = np.random.choice(length, int(length*RANDOM_MASKING_PERCENTAGE), replace=False)
                dis_lst = [x for x in length*'1']
                for index in random_index:
                    dis_lst[index] = '0'

                dis_str = "".join(dis_lst)

                output_labels = context_length * "0" + dis_str + (context_length + remainder_length) * "0"
                output_records.append((output_seq, output_labels, accession, (start, stop)))

# Create FASTA files with labels and unmasked amino acid sequences
if not os.path.exists("out/"):
    os.mkdir("out/")

with open("out/seqs.fasta", "w") as file:
    for seq, _, accession, interval in output_records:
        start, stop = interval
        header = f">{accession}|{start}:{stop}"
        seqstring = "\n".join([seq[i:i + 80] for i in range(0, len(seq), 80)])
        file.write(f"{header}\n{seqstring}\n")
with open("out/labels.fasta", "w") as file:
    for _, labels, accession, interval in output_records:
        start, stop = interval
        header = f">{accession}|{start}:{stop}"
        seqstring = "\n".join([labels[i:i + 80] for i in range(0, len(labels), 80)])
        file.write(f"{header}\n{seqstring}\n")

print('Number of input records:', len(input_records))
print('Number of output records:', len(output_records))
print('Number of unique accessions in output records:', len({record[2] for record in output_records}))