"""Run AUCPreD on profiles generated from DisProt sequences."""

import multiprocessing as mp
import os
import subprocess


def get_record(accession):
    subprocess.run(f'../../../bin/Predict_Property/AUCpreD.sh -i ../tgt_disprot/out/{accession}/{accession}.tgt -o out/raw/',
                   check=True, shell=True)

    # Convert raw output to binary labels
    with open(f'out/raw/{accession}.diso_profile') as raw:
        label = []
        for line in raw:
            if not line.startswith('#'):
                sym = '1' if '*' in line else '0'
                label.append(sym)
        label = ''.join(label)
        labelstring = '\n'.join([label[i:i+80] for i in range(0, len(label), 80)]) + '\n'

    # Get header
    with open(f'../tgt_disprot/out/{accession}/{accession}.fasta_raw') as file:
        header = file.readline()

    return header, labelstring


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.mkdir('out/')
    if not os.path.exists('out/raw/'):
        os.mkdir('out/raw/')

    with mp.Pool(processes=num_processes) as pool:
        args = [path for path in os.listdir('../tgt_disprot/out/') if not path.endswith('out')]
        records = pool.map(get_record, args, chunksize=5)

    with open('out/aucpredp_labels.fasta', 'w') as file:
        for header, labelstring in sorted(records):
            file.write(header + labelstring)

"""
DEPENDENCIES
../tgt_disprot/tgt_disprot.py
"""