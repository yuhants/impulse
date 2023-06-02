#!/bin/bash
#SBATCH -J nanosphere_rate_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH -t 00:40:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python rate_massive_mediator.py 0.075 5e3 1e-8 1e-3
