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
temp_dir="${out_path}/${prefix}"
mkdir "${temp_dir}"
cp "${fasta_path}" "${temp_dir}"
${espritz_path} "${temp_dir}" D 1

# Clean up
if [ ! -d "${out_path}/raw/" ]; then
  mkdir "${out_path}/raw/"
fi
mv "${temp_dir}/${prefix}.espritz" "${temp_dir}/${prefix}.espritz.fasta" "${out_path}/raw/"
rm -r "${temp_dir}"
