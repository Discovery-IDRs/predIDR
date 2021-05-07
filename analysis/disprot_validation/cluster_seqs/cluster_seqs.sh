# Cluster seqs at 40% sequence identity

if [ ! -d out/ ]; then
  mkdir out/
fi

../../../bin/cd-hit -i ../remove_outliers/out/disprot_seqs.fasta -o out/disprot -c 0.4 -n 2 -d 0 -g 1

# NOTES
# c sequence identity threshold
# n word_length (2 as recommended for low identity)
# d full header up to first space in .clstr file description
# g accurate mode: find best matching cluster
#
# DEPENDENCIES
# ../remove_outliers/remove_outliers.py
#     ../remove_outliers/out/disprot_seqs.fasta