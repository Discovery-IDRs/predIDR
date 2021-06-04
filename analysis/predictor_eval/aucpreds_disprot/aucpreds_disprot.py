"""Run AUCPreD on DisProt sequences."""

import multiprocessing as mp
import os
import subprocess


def get_record(path):
    accession = path.split('.')[0]
    subprocess.run(f'../../../bin/Predict_Property/AUCpreD.sh -i ../../disprot_validation/format_seqs/out/seqs/{path} -o out/raw/',
                   check=True, shell=True)

    # Convert raw output to binary labels
    with open(f'out/raw/{accession}.diso_noprof') as raw:
        label = []
        for line in raw:
            if line and not line.startswith('#'):  # If not empty and not comment
                sym = '1' if '*' in line else '0'
                label.append(sym)
        label = ''.join(label)
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'

    # Write labels to file
    with open(f'../../disprot_validation/format_seqs/out/seqs/{path}') as file:
        header = file.readline()

    return header, labelstring


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.mkdir('out/')
    if not os.path.exists('out/raw/'):
        os.mkdir('out/raw/')

    with mp.Pool(processes=num_processes) as pool:
        records = pool.map(get_record, os.listdir('../../disprot_validation/format_seqs/out/seqs/'), chunksize=5)

    with open('out/aucpreds_labels.fasta', 'w') as file:
        for header, labelstring in sorted(records):
            file.write(header + labelstring)

"""
DEPENDENCIES
../../disprot_validation/format_seqs/format_seqs.py
"""