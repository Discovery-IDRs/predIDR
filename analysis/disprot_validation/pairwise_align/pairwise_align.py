"""Pairwise align all sequences in the DisProt database."""

import os
import re
from itertools import combinations

from Bio import SeqIO
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


def get_pident(alignment):
    aligned1, aligned2 = alignment[:2]
    num_sym = 0
    num_ident = 0
    for sym1, sym2 in zip(aligned1, aligned2):
        num_sym += 1
        if sym1 == sym2:
            num_ident += 1
    return 100 * num_ident / num_sym


# Load FASTAs
fastas = {}
for fasta in SeqIO.parse('../extract_seqs/out/disprot_2020_06.fasta', 'fasta'):
    disprot_id = re.match(f'disprot_id:(DP[0-9]+)', fasta.description).group(1)
    seq = str(fasta.seq)
    fastas[disprot_id] = seq

# Align sequences
alignments = []
for fasta1, fasta2 in combinations(fastas.items(), 2):
    # Unpack components of fasta
    disprot_id1, seq1 = fasta1
    disprot_id2, seq2 = fasta2

    # Get alignment with max pident
    alignment = max(pairwise2.align.globalds(seq1, seq2, matlist.blosum62, -11, -1), key=get_pident)
    alignments.append((disprot_id1, disprot_id2, *alignment[:2]))

# Make output directory
if not os.path.exists('out/'):
    os.mkdir('out/')


"""
DEPENDENCIES
../extract_seqs/extract_seqs.py
    ../extract_seqs/out/disprot_2020_06.fasta
"""