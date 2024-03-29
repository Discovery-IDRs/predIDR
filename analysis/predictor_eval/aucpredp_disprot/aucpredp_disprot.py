"""Run AUCPreD on profiles generated from DisProt sequences."""

import multiprocessing as mp
import os
import subprocess


def get_record(accession):
    subprocess.run(f'../../../bin/Predict_Property/AUCpreD.sh -i ../tgt_disprot/out/{accession}/{accession}.tgt -o out/raw/',
                   check=True, shell=True)

    # Convert raw output to binary labels
    with open(f'out/raw/{accession}.diso_profile') as raw:
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
    with open(f'../tgt_disprot/out/{accession}/{accession}.fasta_raw') as file:
        header = file.readline()

    return header, scorestring, labelstring


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.mkdir('out/')
    if not os.path.exists('out/raw/'):
        os.mkdir('out/raw/')

    with mp.Pool(processes=num_processes) as pool:
        args = [path for path in os.listdir('../tgt_disprot/out/') if not path.endswith('out')]
        records = pool.map(get_record, args, chunksize=5)

    with open('out/aucpredp_scores.fasta', 'w') as scores, open('out/aucpredp_labels.fasta', 'w') as labels:
        for header, scorestring, labelstring in sorted(records):
            if 'DP02925' in header:
                continue
            scores.write(header + scorestring)
            labels.write(header + labelstring)

"""
NOTES
DP02925 does not complete successfully, returning the error message:

seq_len 7096 not equal to reso_len 4084

The length of the sequence is 7096, so I suspect one portion of the TGT file is poorly formatted or complete. The format
is not documented at all, so it is difficult to say where the error is. It appears it's simply the outputs of search
programs like PSI-BLAST or predictors like DISOPRED concatenated, one after the other.

Unfortunately, the predictor makes empty files rather than terminating, so when the raw outputs are compiled, DP02925
has an "empty" record (header with no corresponding sequence). The empty entries for in the compiled label and score
FASTA files are therefore manually removed.

DEPENDENCIES
../tgt_disprot/tgt_disprot.py
"""