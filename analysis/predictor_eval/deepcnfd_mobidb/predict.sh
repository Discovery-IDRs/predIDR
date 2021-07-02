#!/bin/bash
# Run DeepCNF_D on individual FASTA files

# Unpack args
deepcnfd_path=$1
fasta_path=$2
out_path="$(basename $3)"  # Remove trailing / if present

# Run DeepCNF_D
# (Automatically writes output to current directory, so move to out/raw/ to execute)
if [ ! -d "${out_path}/raw/" ]; then
  mkdir "${out_path}/raw/"
fi
cd "${out_path}/raw/"
"../../${deepcnfd_path}" -i "../../${fasta_path}" -s 1
