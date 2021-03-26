"""Remove outliers from data."""

import os

import scipy.ndimage as ndimage
from Bio import SeqIO

# Load outlier IDs
outlier_ids = ['Q8WZ42', 'O97791', 'G4SLH0']  # Titin sequences
with open('../mobidb_stats/out/ns_codes.tsv') as file:  # Sequences with non-standard amino acids
    file.readline()  # Skip header
    for line in file:
        fields = line.split('\t')
        outlier_ids.append(fields[0])

# Make output directoies
if not os.path.exists('out/'):
    os.mkdir('out/')
if not os.path.exists('out/Seq/'):
    os.mkdir('out/Seq/')
if not os.path.exists('out/Disorder/'):
    os.mkdir('out/Disorder/')

# Remove outliers from allseq
fastas = SeqIO.parse('../generate_fastas/out/allseq.fasta', 'fasta')
with open('out/allseq.fasta', 'w') as allseq:
    for record in fastas:
        id = record.id.split('|')[0]
        if record.id.split('|')[0] not in outlier_ids:
            # Write to allseq.fasta
            seq = str(record.seq)
            seqstring = '\n'.join(seq[i:i + 80] for i in range(0, len(seq), 80))
            allseq.write('>' + record.description + '\n' + seqstring + '\n')

            # Write to individual fasta
            with open(f'out/Seq/{id}_seq.fasta', 'w') as oneseq:
                oneseq.write('>' + record.description + '\n' + seqstring + '\n')

# Remove outliers alldisorder including flipping labels in short segments
fastas = SeqIO.parse('../generate_fastas/out/alldisorder.fasta', 'fasta')
with open('out/alldisorder.fasta', 'w') as allseq:
    for record in fastas:
        id = record.id.split('|')[0]
        if record.id.split('|')[0] not in outlier_ids:
            # Write to alldisorder.fasta
            seq1 = [int(s) for s in str(record.seq)]
            seq2 = ['0' for _ in range(len(seq1))]
            labels = ndimage.label(seq1)[0]
            for s, in ndimage.find_objects(labels):  # Unpack 1-tuple
                if s.stop - s.start >= 10:
                    seq2[s.start:s.stop] = (s.stop-s.start) * ['1']
            seq2 = ''.join(seq2)
            seqstring = '\n'.join(seq2[i:i + 80] for i in range(0, len(seq2), 80))
            allseq.write('>' + record.description + '\n' + seqstring + '\n')

            # Write to individual fasta
            with open(f'out/Disorder/{id}_disorder.fasta', 'w') as oneseq:
                oneseq.write('>' + record.description + '\n' + seqstring + '\n')

"""
NOTES
Titin sequences were removed due to their extreme length compared to other sequences.
Sequences with non-standard amino acids were removed (as reported in mobidb_stats.py).
Short disordered segments (9 aa or smaller) were also removed by flipping their labels from 1s to 0s.
    This was largely motivated by the exclusion such segments from DisProt as well as their tendency to be simple loops
    connecting larger, ordered domains.
    Proteins with no disordered labels following this change were not removed.
"""