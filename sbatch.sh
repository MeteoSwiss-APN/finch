#!/bin/bash -l

#SBATCH --job-name="finch"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

python run.py
