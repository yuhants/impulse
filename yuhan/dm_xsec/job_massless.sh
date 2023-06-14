#!/bin/bash
#SBATCH -J massless_nanosphere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python rate_massless_mediator.py
