"""Remove outliers from data."""

import os
import scipy.ndimage as ndimage

from src.utils import read_fasta

# Load outlier accessions
outliers = {'Q8WZ42', 'O97791', 'G4SLH0'}  # Titin sequences
with open('../mobidb_stats/out/ns_codes.tsv') as file:  # Sequences with non-standard amino acids
    file.readline()  # Skip header
    for line in file:
        fields = line.split('\t')
        outliers.add(fields[0])

# Make output directories
if not os.path.exists('out/'):
    os.mkdir('out/')

# Remove outliers from seqs
fastas = read_fasta('../generate_fastas/out/mobidb_seqs.fasta')
with open('out/mobidb_seqs.fasta', 'w') as file:
    for header, seq in fastas:
        accession = header.split('|')[0][1:]  # Trim >
        if accession not in outliers:
            seqstring = '\n'.join(seq[i:i+80] for i in range(0, len(seq), 80))
            file.write(f'{header}\n{seqstring}\n')

# Remove outliers from labels including flipping labels in short segments
fastas = read_fasta('../generate_fastas/out/mobidb_labels.fasta')
with open('out/mobidb_labels.fasta', 'w') as file:
    for header, seq in fastas:
        accession = header.split('|')[0][1:]  # Trim >
        if accession not in outliers:
            seq1 = [1 if sym == '1' else 0 for sym in seq]
            seq2 = [sym for sym in seq]
            labels = ndimage.label(seq1)[0]
            for s, in ndimage.find_objects(labels):  # Unpack 1-tuple
                if s.stop - s.start < 10:
                    seq2[s.start:s.stop] = (s.stop-s.start) * ['0']
            seq = ''.join(seq2)
            seqstring = '\n'.join(seq[i:i+80] for i in range(0, len(seq), 80))
            file.write(f'{header}\n{seqstring}\n')

"""
NOTES
Titin sequences were removed due to their extreme length compared to other sequences.
Sequences with non-standard amino acids were removed (as reported in mobidb_stats.py).
Short disordered segments (9 aa or smaller) were also removed by flipping their labels from 1s to 0s.
    This was largely motivated by the exclusion such segments from DisProt as well as their tendency to be simple loops
    connecting larger, ordered domains.
    Proteins with no disordered labels following this change were not removed.

DEPENDENCIES
../generate_fastas/generate_fastas.py
    ../generate_fastas/out/*
../mobidb_stats/mobidb_stats.py
    ../mobidb_stats/mobidb_stats/out/ns_codes.tsv
"""