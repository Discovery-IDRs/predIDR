"""Run IUPRED2a on MobiDB sequences."""

import os
import subprocess

# Make output directories
if not os.path.exists('out/'):
    os.mkdir('out/')
if not os.path.exists('out/raw/'):
    os.mkdir('out/raw/')

with open('out/iupred2a_labels.fasta', 'w') as labels:
    for path in sorted(os.listdir('../../mobidb_validation/format_seqs/out/seqs/')):
        accession = path.split('.')[0]
        process = subprocess.run(['python', '../../../bin/iupred2a/iupred2a.py',
                                  f'../../mobidb_validation/format_seqs/out/seqs/{path}', 'long'],
                                 capture_output=True, text=True, check=True)

        # Write raw output to file
        with open(f'out/raw/{accession}.txt', 'w') as raw:
            raw.write(process.stdout)

        # Convert raw output to binary labels
        label = []
        for line in process.stdout.split('\n'):
            if line and not line.startswith('#'):  # If not empty and not comment
                score = float(line.split()[2])
                sym = '1' if score >= 0.5 else '0'
                label.append(sym)
        label = ''.join(label)
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'

        # Write labels to file
        with open(f'../../mobidb_validation/format_seqs/out/seqs/{path}') as file:
            header = file.readline()
        labels.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
"""