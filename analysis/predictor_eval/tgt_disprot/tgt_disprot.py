"""Make TGT (profile) files for DisProt sequences."""

import multiprocessing as mp
import os
import subprocess


def make_tgt(path):
    accession = path.split('.')[0]
    process = subprocess.run(f'../../../bin/TGT_Package/A3M_TGT_Gen.sh -i ../../disprot_validation/format_seqs/out/seqs/{path} '
                             f'-c 1 -m 2 -o out/{accession}/ &> out/{accession}.out', check=True, shell=True)
    return process


num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])

if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.mkdir('out/')

    with mp.Pool(processes=num_processes) as pool:
        processes = pool.map(make_tgt, os.listdir('../../disprot_validation/format_seqs/out/seqs/'))

"""
DEPENDENCIES
../../disprot_validation/format_seqs/format_seqs.py
"""