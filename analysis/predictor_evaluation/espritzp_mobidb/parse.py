"""Parse raw output from Espritz into combined FASTA file."""

import os

score_records, label_records = [], []
paths = [path for path in os.listdir('out/raw/') if path.endswith('espritz')]
for path in sorted(paths):
    # Parse raw output to scores and labels
    accession = path.split('.')[0]
    with open(f'out/raw/{accession}.espritz') as file:
        star_count = 0
        scores, labels = [], []
        for line in file:
            if line.startswith('*'):
                star_count += 1
            elif star_count >= 2:
                value = line.split()[1]
                sym = '1' if float(value) >= 0.2644 else '0'  # Cutoff was pulled from espritz code; unclear if threshold > or >=, so may differ in edge cases
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
with open('out/espritzp_scores.fasta', 'w') as file:
    for header, scorestring in score_records:
        file.write(header + scorestring)

with open('out/espritzp_labels.fasta', 'w') as file:
    for header, labelstring in label_records:
        file.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
./predict.sh
"""