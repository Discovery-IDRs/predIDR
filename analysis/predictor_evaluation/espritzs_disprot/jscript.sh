#!/bin/bash
# Key parameters
#SBATCH --account=fc_eisenlab
#SBATCH --partition=savio2
#SBATCH --time=12:00:00
#SBATCH --qos=savio_normal
#
# Process parameters
#SBATCH --nodes=1
#
# Reporting parameters
#SBATCH --job-name=predict
#SBATCH --output=predict.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marcsingleton@berkeley.edu
#
# Command(s) to run:
source /global/home/users/singleton/.bashrc
conda activate predIDR
module load gnu-parallel

# Link to output in scratch
if [ ! -d out ]; then
  out_dir=/global/scratch/users/predIDR/analysis/predictor_eval/espritzs_disprot/out/
  if [ ! -d ${out_dir} ]; then
    mkdir -p ${out_dir}  # -p makes intermediate directory if they do not exist
  fi
  ln -s ${out_dir} out
fi

fasta_dir=../../disprot_validation/format_seqs/out/seqs/
parallel -j $SLURM_CPUS_ON_NODE bash predict.sh ../../../bin/Espritz/espritz.pl $fasta_dir/{} out/ ::: $(ls $fasta_dir)
