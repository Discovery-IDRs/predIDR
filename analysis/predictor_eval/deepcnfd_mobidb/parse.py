"""Parse raw output from DeepCNF_D into combined FASTA file."""

import os

score_records, label_records = [], []
for path in sorted(os.listdir('out/raw/')):
    # Parse raw output to scores and labels
    accession = path.split('.')[0]
    with open(f'out/raw/{accession}.diso') as file:
        star_count = 0
        scores, labels = [], []
        for line in file:
            if not line.startswith('\n') and not line.startswith('#'):  # If not empty and not comment
                value = line.split()[3]
                sym = '1' if '*' in line else '0'
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
with open('out/deepcnfd_scores.fasta', 'w') as file:
    for header, scorestring in score_records:
        file.write(header + scorestring)

with open('out/deepcnfd_labels.fasta', 'w') as file:
    for header, labelstring in label_records:
        file.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
./predict.sh
"""