"""Parse raw output from Espritz into combined FASTA file."""

import os

with open('out/espritzs_labels.fasta', 'w') as labels:
    paths = [path for path in os.listdir('out/raw/') if path.endswith('espritz')]
    for path in sorted(paths):
        accession = path.split('.')[0]
        with open(f'out/raw/{accession}.espritz') as file:
            star_count = 0
            label = []
            for line in file:
                if line.startswith('*'):
                    star_count += 1
                elif star_count >= 2:
                    sym = '1' if float(line.split()[1]) >= 0.2644 else '0'  # Cutoff was pulled from espritz code; unclear if threshold > or >=, so may differ in edge cases
                    label.append(sym)
            label = ''.join(label)
            labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'

        # Write labels to file
        with open(f'../../disprot_validation/format_seqs/out/seqs/{accession}.fasta') as file:
            header = file.readline()
        labels.write(header + labelstring)

"""
DEPENDENCIES
../../disprot_validation/format_seqs/format_seqs.py
./predict.sh
"""