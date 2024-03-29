"""Run AUCPreD on MobiDB sequences."""

import multiprocessing as mp
import os
import subprocess


def get_record(path):
    accession = path.split('.')[0]
    subprocess.run(f'../../../bin/Predict_Property/AUCpreD.sh -i ../../mobidb_validation/format_seqs/out/seqs/{path} -o out/raw/',
                   check=True, shell=True)

    # Convert raw output to binary labels
    with open(f'out/raw/{accession}.diso_noprof') as raw:
        scores, labels = [], []
        for line in raw:
            if not line.startswith('#'):
                value = line.split()[3]
                sym = '1' if '*' in line else '0'
                scores.append(value)
                labels.append(sym)
        scorestring = '\n'.join(scores) + '\n'
        labelstring = '\n'.join([''.join(labels)[i:i+80] for i in range(0, len(labels), 80)]) + '\n'

    # Get header
    with open(f'../../mobidb_validation/format_seqs/out/seqs/{path}') as file:
        header = file.readline()

    return header, scorestring, labelstring


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.mkdir('out/')
    if not os.path.exists('out/raw/'):
        os.mkdir('out/raw/')

    with mp.Pool(processes=num_processes) as pool:
        records = pool.map(get_record, os.listdir('../../mobidb_validation/format_seqs/out/seqs/'), chunksize=5)

    with open('out/aucpreds_scores.fasta', 'w') as scores, open('out/aucpreds_labels.fasta', 'w') as labels:
        for header, scorestring, labelstring in sorted(records):
            scores.write(header + scorestring)
            labels.write(header + labelstring)

"""
DEPENDENCIES
../../mobidb_validation/format_seqs/format_seqs.py
"""