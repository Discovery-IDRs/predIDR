"""Parse raw output from DeepCNF_D into combined FASTA file."""

import os

with open('out/deepcnfd_labels.fasta', 'w') as labels:
    for path in sorted(os.listdir('out/raw/')):
        accession = path.split('.')[0]
        with open(f'out/raw/{accession}.diso') as file:
            star_count = 0
            label = []
            for line in file:
                if line and not line.startswith('#'):  # If not empty and not comment
                    sym = '1' if '*' in line else '0'
                    label.append(sym)
            label = ''.join(label)
            labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'

        # Write labels to file
        with open(f'../../mobidb_validation/format_seqs/out/seqs/{accession}.fasta') as file:
            header = file.readline()
        labels.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
./predict.sh
"""