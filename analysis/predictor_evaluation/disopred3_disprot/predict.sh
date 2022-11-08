#!/bin/bash
# Run DISOPRED3 on individual FASTA files

# Unpack args
disopred_path="$(realpath $1)"
fasta_path="$(realpath $2)"
out_path="$(basename $3)"  # Remove trailing / if present

# Construct names
fasta_name=$(basename "${fasta_path}")
prefix="${fasta_name%.*}"

# Run DISOPRED3
# (Automatically writes temporary files  to current directory, so move to out/raw/${prefix} to execute)
if [ ! -d "${out_path}/raw/${prefix}/" ]; then
  mkdir -p "${out_path}/raw/${prefix}/"
fi
cd "${out_path}/raw/${prefix}/"
cp "${fasta_path}" .
"${disopred_path}" "${fasta_name}"
rm "${fasta_name}"