"""Remove outliers from data."""

import os
from shutil import copyfile
from Bio import SeqIO

# Load outlier IDs
outlier_ids = ['Q8WZ42', 'O97791', 'G4SLH0']  # Titin sequences
with open('../mobidb_stats/out/ns_codes.tsv') as file:  # Sequences with non-standard amino acids
    file.readline()  # Skip header
    for line in file:
        fields = line.split('\t')
        outlier_ids.append(fields[0])

if not os.path.exists('out/'):
    os.mkdir('out/')

# Remove outliers from allseq and alldisorder
fastas = SeqIO.parse('../generate_fastas/out/allseq.fasta', 'fasta')
with open('out/allseq.fasta', 'w') as file:
    for record in fastas:
        id = record.id.split('|')[0]
        if record.id.split('|')[0] not in outlier_ids:
            seq = str(record.seq)
            seqstring = '\n'.join(seq[i:i + 80] for i in range(0, len(seq), 80))
            file.write('>' + record.description + '\n' + seqstring + '\n')

fastas = SeqIO.parse('../generate_fastas/out/alldisorder.fasta', 'fasta')
with open('out/alldisorder.fasta', 'w') as file:
    for record in fastas:
        id = record.id.split('|')[0]
        if record.id.split('|')[0] not in outlier_ids:
            seq = str(record.seq)
            seqstring = '\n'.join(seq[i:i + 80] for i in range(0, len(seq), 80))
            file.write('>' + record.description + '\n' + seqstring + '\n')

# Remove outliers from individual files
if not os.path.exists('out/Seq/'):
    os.mkdir('out/Seq/')

for file in os.listdir('../generate_fastas/out/Seq/'):
    if file.split('_')[0] not in outlier_ids:
        copyfile('../generate_fastas/out/Seq/' + file, 'out/Seq/' + file)

if not os.path.exists('out/Disorder/'):
    os.mkdir('out/Disorder/')

for file in os.listdir('../generate_fastas/out/Disorder/'):
    if file.split('_')[0] not in outlier_ids:
        copyfile('../generate_fastas/out/Disorder/' + file, 'out/Disorder/' + file)

"""
NOTES
Titin sequences were removed due to their extreme length compared to other sequences.
"""