"""Run IUPRED2a on MobiDB sequences."""

import os
import subprocess

# Make output directories
if not os.path.exists('out/'):
    os.mkdir('out/')
if not os.path.exists('out/raw/'):
    os.mkdir('out/raw/')

score_records, label_records = [], []
for path in sorted(os.listdir('../../mobidb_validation/format_seqs/out/seqs/')):
    # Run IUPRED2a
    accession = path.split('.')[0]
    process = subprocess.run(['python', '../../../bin/iupred2a/iupred2a.py',
                              f'../../mobidb_validation/format_seqs/out/seqs/{path}', 'long'],
                             capture_output=True, text=True, check=True)

    # Write raw output to file
    with open(f'out/raw/{accession}.txt', 'w') as raw:
        raw.write(process.stdout)

    # Parse raw output to scores and labels
    scores, labels = [], []
    for line in process.stdout.split('\n'):
        if line and not line.startswith('#'):  # If not empty and not comment
            value = line.split()[2]
            sym = '1' if float(value) >= 0.5 else '0'
            scores.append(value)
            labels.append(sym)
    scorestring = '\n'.join(scores) + '\n'
    labelstring = '\n'.join([''.join(labels)[i:i+80] for i in range(0, len(labels), 80)]) + '\n'

    # Store strings with header
    with open(f'../../mobidb_validation/format_seqs/out/seqs/{accession}.fasta') as file:
        header = file.readline()
    score_records.append((header, scorestring))
    label_records.append((header, labelstring))

# Write consolidated output to file
with open('out/iupred2a_scores.fasta', 'w') as file:
    for header, scorestring in score_records:
        file.write(header + scorestring)

with open('out/iupred2a_labels.fasta', 'w') as file:
    for header, labelstring in label_records:
        file.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
"""