#!/bin/bash -l

#SBATCH --job-name="finch_run"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

srun --exclusive python run.py
