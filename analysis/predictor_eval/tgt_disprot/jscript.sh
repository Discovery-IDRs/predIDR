#!/bin/bash
# Key parameters
#SBATCH --account=fc_eisenlab
#SBATCH --partition=savio2
#SBATCH --time=24:00:00
#SBATCH --qos=savio_normal
#
# Process parameters
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#
# Reporting parameters
#SBATCH --job-name=tgt_disprot
#SBATCH --output=tgt_disprot.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marcsingleton@berkeley.edu
#
# Command(s) to run:
# Set current directory and link to output in scratch
if [ ! -d out ]; then
  out_dir=/global/scratch/singleton/predIDR/analysis/predictor_eval/tgt_disprot/out/
  if [ ! -d ${out_dir} ]; then
    mkdir -p ${out_dir}  # -p makes intermediate directory if they do not exist
  fi
  ln -s ${out_dir} out
fi

source /global/home/users/singleton/.bashrc
conda activate predIDR
python tgt_disprot.py
