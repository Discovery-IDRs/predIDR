#!/bin/bash
# Run Espritz on individual FASTA files

# Unpack args
espritz_path=$1
fasta_path=$2
out_path="$(basename $3)"  # Remove trailing / if present

# Construct names
fasta_name=$(basename "${fasta_path}")
prefix="${fasta_name%.*}"
ext="${fasta_name##*.}"

# Run Espritz
# (Espritz is buggy and only runs correctly if it's executed from the folder in which it's stored due to path issues in espritz.pl)
temp_dir="$(realpath "${out_path}/${prefix}")"
mkdir "${temp_dir}"
cp "${fasta_path}" "${temp_dir}"
cd $(dirname "${espritz_path}")
"./$(basename "${espritz_path}")" "${temp_dir}" pD 1
cd -  # Return to previous directory

# Clean up
if [ ! -d "${out_path}/raw/" ]; then
  mkdir "${out_path}/raw/"
fi
mv "${temp_dir}/${prefix}.espritz" "${temp_dir}/${prefix}.espritz.fasta" "${out_path}/raw/"
rm -r "${temp_dir}"
